"""Random slopes for the binary, ordered and multinomial logit
(`random={"g": ["1", "x"]}`, random_effects/slopes.py): against R's
GLMMadaptive (adaptive quadrature, a random intercept and slope, with and
without a fixed effect as dummies), against brute-force integration (the
ordered logit; the multinomial's slopes), the multinomial with two levels
against the binary logit, predict (seen and unseen groups, averaged), and
refusals."""

import warnings

import numpy as np
import polars as pl
import pytest
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from logit_stream import logit_stream, mlogit_stream, ologit_stream


def _sim(G=400, sd0=0.8, sd1=0.5, rho=0.3, seed=0, size=8, cells=None, J=2):
    """Groups with a correlated random intercept and slope on x; a binary
    (J = 2) or ordered outcome."""
    rng = np.random.default_rng(seed)
    n_g = 1 + rng.poisson(size - 1, G)
    g = np.repeat(np.arange(G), n_g)
    n = len(g)
    S = np.array([[sd0 ** 2, rho * sd0 * sd1], [rho * sd0 * sd1, sd1 ** 2]])
    a = rng.multivariate_normal([0, 0], S, G)
    x = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, n).astype(float)
    lat = 0.7 * x - 0.4 * x2 + a[g, 0] + a[g, 1] * x + rng.logistic(size=n)
    cols = {}
    if cells:
        cell = rng.integers(0, cells, G)
        lat = lat + rng.normal(0, 0.5, cells)[cell][g]
        cols["cell"] = cell[g]
    cuts = [0.0] if J == 2 else list(np.linspace(-1, 1.2, J - 1))
    y = np.searchsorted(cuts, lat)
    return pl.DataFrame({"y": y, "x": x, "x2": x2, "g": g, **cols})


def _sim_mnl(G=150, seed=0, size=6, J=3):
    rng = np.random.default_rng(seed)
    n_g = 1 + rng.poisson(size - 1, G)
    g = np.repeat(np.arange(G), n_g)
    n = len(g)
    a = rng.normal(0, 0.6, (G, J - 1))
    x = rng.normal(size=n)
    E = np.zeros((n, J))
    for j in range(1, J):
        E[:, j] = -0.3 * j + (0.5 - 0.3 * j) * x + a[g, j - 1] * x
    P = softmax(E, 1)
    y = (np.cumsum(P, 1) < rng.random(n)[:, None]).sum(1)
    return pl.DataFrame({"y": y, "x": x, "g": g})


def _fit(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, verbose=False, random_tol=1e-9, **kw)


# R 4.5.3, GLMMadaptive 0.9.7: mixed_model(y ~ x + x2 [+ factor(cell)], random = ~ x | g,
# family = binomial(), nAGQ = 7, control = list(iter_EM = 100, optimizer = "optim",
# optim_method = "BFGS", tol1 = 1e-8, tol2 = 1e-8, tol3 = 1e-10))
GLMMA = {
    "plain": dict(data=dict(G=1500), b=[-0.01560217, 0.64750149, -0.37867459],
                  se=[0.03478737, 0.02945505, 0.04412458], sd=[0.79290497, 0.53068906],
                  corr=0.36980138, ll=-7617.401055),
    "small": dict(data=dict(G=800, seed=3, size=4), b=[0.04397845, 0.72992083, -0.40242009],
                  se=[0.06393552, 0.05871332, 0.08865144], sd=[0.82175319, 0.54140518],
                  corr=0.35878838, ll=-1997.702591),
    "cells": dict(data=dict(G=700, seed=7, size=6, cells=12), b=[0.63735159, -0.34288142],
                  se=[0.05012046, 0.07664375], sd=[0.85874240, 0.53912413],
                  corr=0.26273180, ll=-2622.910939),
}


@pytest.mark.parametrize("case", list(GLMMA))
def test_binary_matches_glmmadaptive(case):
    ref = GLMMA[case]
    d = _sim(**ref["data"])
    fml = "y ~ x + x2 | cell" if case == "cells" else "y ~ x + x2"
    f = _fit(logit_stream, fml, d, random={"g": ["1", "x"]}, random_nodes=7)
    se = np.array(ref["se"])
    assert np.all(np.abs(f.beta - ref["b"]) < 0.01 * se), (f.beta - ref["b"]) / se
    np.testing.assert_allclose(f.se, se, rtol=2e-3)
    info = f.random_effects["g"]
    np.testing.assert_allclose(info["sd"], ref["sd"], rtol=2e-3)
    assert info["corr"][0, 1] == pytest.approx(ref["corr"], abs=2e-3)
    assert f.loglik == pytest.approx(ref["ll"], abs=2e-3)
    assert info["levels"] == ["(Intercept)", "x"]


def _ordered_brute(d):
    """Minus the exact marginal log-likelihood of the ordered logit with a
    random intercept and slope, by a 40 x 40 Gauss-Hermite grid on the
    prior (tiny groups: the posterior is close to it), in (b, cutpoints,
    Sigma's log-Cholesky)."""
    from numpy.polynomial.hermite_e import hermegauss
    zq, wq = hermegauss(40)
    Z = np.array(np.meshgrid(zq, zq, indexing="ij")).reshape(2, -1).T
    lw = np.log(np.outer(wq, wq).ravel() / wq.sum() ** 2)
    x, x2, y, g = (d[c].to_numpy() for c in ("x", "x2", "y", "g"))
    G = g.max() + 1

    def nll(p):
        b, cuts = p[:2], np.sort(p[2:4])
        L = np.array([[np.exp(p[4]), 0], [p[6], np.exp(p[5])]])
        A = Z @ L.T                                             # points x 2
        eta = (b[0] * x + b[1] * x2)[None, :] + A[:, :1] + A[:, 1:] * x[None, :]
        c = np.r_[-np.inf, cuts, np.inf]
        lp = np.log(expit(c[y + 1][None] - eta) - expit(c[y][None] - eta))
        lg = np.zeros((len(A), G))
        np.add.at(lg.T, g, lp.T)
        return -float(logsumexp(lg + lw[:, None], axis=0).sum())

    return nll


def test_ordered_matches_brute_force():
    d = _sim(G=150, seed=11, size=4, J=3)
    f = _fit(ologit_stream, "y ~ x + x2", d, random={"g": ["1", "x"]}, random_nodes=9)
    S = np.asarray(f.random_effects["g"]["Sigma"])
    L = np.linalg.cholesky(S)
    p = np.r_[f.beta, np.log(np.diag(L)), L[1, 0]]
    nll = _ordered_brute(d)
    r = minimize(nll, p, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-9,
                                                          "maxiter": 20_000})
    assert f.loglik == pytest.approx(-nll(p), abs=2e-3)
    se = np.r_[f.se, 0.1, 0.1, 0.1]
    assert np.all(np.abs(p - r.x) < 0.02 * se), (p - r.x) / se


def test_multinomial_slopes_match_brute_force():
    """Slopes only (no random intercept), J = 3: q = 2 effects, a 2-D grid."""
    from numpy.polynomial.hermite_e import hermegauss
    d = _sim_mnl()
    f = _fit(mlogit_stream, "y ~ x", d, random={"g": "x"}, random_nodes=9)
    zq, wq = hermegauss(24)
    Z = np.array(np.meshgrid(zq, zq, indexing="ij")).reshape(2, -1).T
    lw = np.log(np.outer(wq, wq).ravel() / wq.sum() ** 2)
    x, y, g = (d[c].to_numpy() for c in ("x", "y", "g"))
    G = g.max() + 1

    def nll(p):
        B = p[:4].reshape(2, 2)                         # level x (intercept, x)
        L = np.array([[np.exp(p[4]), 0], [p[6], np.exp(p[5])]])
        A = Z @ L.T
        E = np.zeros((len(A), len(x), 3))
        for j in range(2):
            E[:, :, j + 1] = B[j, 0] + B[j, 1] * x + A[:, j:j + 1] * x[None, :]
        lp = E[:, np.arange(len(x)), y] - logsumexp(E, axis=2)
        lg = np.zeros((len(A), G))
        np.add.at(lg.T, g, lp.T)
        return -float(logsumexp(lg + lw[:, None], axis=0).sum())

    S = np.asarray(f.random_effects["g"]["Sigma"])
    Lc = np.linalg.cholesky(S)
    p = np.r_[f.beta, np.log(np.diag(Lc)), Lc[1, 0]]
    r = minimize(nll, p, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8,
                                                          "maxiter": 5_000})
    assert f.loglik == pytest.approx(-nll(p), abs=2e-3)
    assert nll(p) <= r.fun + 1e-3
    se = np.r_[f.se, 0.1, 0.1, 0.1]
    assert np.all(np.abs(p - r.x) < 0.03 * se), (p - r.x) / se
    assert f.random_effects["g"]["levels"] == ["1:x", "2:x"]


def test_two_level_multinomial_is_the_binary_logit():
    d = _sim(G=300, seed=5)
    b = _fit(logit_stream, "y ~ x + x2", d, random={"g": ["1", "x"]})
    m = _fit(mlogit_stream, "y ~ x + x2", d, random={"g": ["1", "x"]})
    np.testing.assert_allclose(m.beta, b.beta, atol=1e-6)
    np.testing.assert_allclose(m.random_effects["g"]["Sigma"], b.random_effects["g"]["Sigma"],
                               atol=1e-6)
    assert m.loglik == pytest.approx(b.loglik, abs=1e-6)


def test_predict_and_average():
    d = _sim(G=300, seed=6)
    f = _fit(logit_stream, "y ~ x + x2", d, random={"g": ["1", "x"]})
    r = f.ranef().collect()
    assert r.columns == ["g", "re_g", "re_g:x", "sd_g", "sd_g:x", "n_obs"]
    row = r.filter(pl.col("g") == 4)
    new = pl.DataFrame({"x": [0.5, -1.0], "x2": [1.0, 0.0], "g": [4, 10 ** 6]})
    eta = f.beta[0] + new["x"].to_numpy() * f.beta[1] + new["x2"].to_numpy() * f.beta[2]
    a0, a1 = row["re_g"][0], row["re_g:x"][0]
    p = f.predict(new, type="response").collect()["prediction"].to_numpy()
    np.testing.assert_allclose(p, expit(eta + np.r_[a0 + a1 * 0.5, 0.0]), rtol=1e-10)
    # averaged: over N(z'mode, z'Cz) for the seen group, N(0, z'Sigma z) for the new one
    C = f._spec["ranef_cov"]["g"][r["g"].to_list().index(4)]
    S = np.asarray(f.random_effects["g"]["Sigma"])
    zz = np.linspace(-12, 12, 4001)
    w = np.exp(-zz * zz / 2)
    w /= w.sum()
    want = []
    for i, (zi, mu, V) in enumerate([(np.r_[1, 0.5], a0 + a1 * 0.5, C),
                                     (np.r_[1, -1.0], 0.0, S)]):
        s = float(np.sqrt(zi @ V @ zi))
        want.append(w @ expit(eta[i] + mu + s * zz))
    pa = f.predict(new, type="response", effects="average").collect()["prediction"].to_numpy()
    np.testing.assert_allclose(pa, want, rtol=1e-7)
    # the multinomial's averaged probabilities sum to 1
    m = _fit(mlogit_stream, "y ~ x", _sim_mnl(seed=2), random={"g": ["1", "x"]})
    pm = m.predict(pl.DataFrame({"x": [0.3], "g": [10 ** 6]}), effects="average").collect()
    assert pm.select(pl.sum_horizontal(pl.col("^prob_.*$"))).item() == pytest.approx(1.0)


def test_draws():
    """Each group's draw given its rows: the resampling's posterior moments
    against a dense grid (binary), the two-level multinomial's draws the
    binary's, and the parameter draws' spread the standard errors."""
    d = _sim(G=300, seed=6)
    f = _fit(logit_stream, "y ~ x + x2", d, random={"g": ["1", "x"]})
    one = f.draw_ranef(seed=2)
    groups = f.ranef().collect()["g"].to_list()
    S = np.asarray(f.random_effects["g"]["Sigma"])
    Li = np.linalg.inv(S)
    gr = np.linspace(-4.5, 4.5, 451)
    A0, A1 = np.meshgrid(gr, gr, indexing="ij")
    for g in (4, 11, 27):
        sub = d.filter(pl.col("g") == g)
        x, x2, y = (sub[c].to_numpy() for c in ("x", "x2", "y"))
        eta = f.beta[0] + f.beta[1] * x + f.beta[2] * x2
        lp = -0.5 * (Li[0, 0] * A0 ** 2 + 2 * Li[0, 1] * A0 * A1 + Li[1, 1] * A1 ** 2)
        for i in range(len(y)):
            p = expit(eta[i] + A0 + A1 * x[i])
            lp += np.log(np.where(y[i] == 1, p, 1 - p))
        W = np.exp(lp - lp.max())
        W /= W.sum()
        mean = np.array([(W * A0).sum(), (W * A1).sum()])
        sd = np.sqrt(np.array([(W * A0 ** 2).sum(), (W * A1 ** 2).sum()]) - mean ** 2)
        k = groups.index(g)
        # the quadrature's conditional sds in ranef(), exactly; the resampling's mean
        # within its Monte Carlo error
        row = f.ranef().collect().row(k, named=True)
        np.testing.assert_allclose([row["sd_g"], row["sd_g:x"]], sd, rtol=1e-3)
        ess = one.diagnostics["ranef_ess"][k]
        assert np.all(np.abs(one.diagnostics["ranef_mean"][k] - mean) < 4 * sd / np.sqrt(ess))
    m = _fit(mlogit_stream, "y ~ x + x2", d, random={"g": ["1", "x"]})
    a = f.draw_ranef(seed=5).ranef().collect()
    b = m.draw_ranef(seed=5).ranef().collect()
    np.testing.assert_allclose(b["re_g:1"].to_numpy(), a["re_g"].to_numpy(), atol=1e-4)
    np.testing.assert_allclose(b["re_g:1:x"].to_numpy(), a["re_g:x"].to_numpy(), atol=1e-4)
    ds = f.draw_parameters(200, seed=3)
    B = np.array([x.beta for x in ds])
    np.testing.assert_allclose(B.std(0), f.se, rtol=0.2)
    sds = np.array([x.random_effects["g"]["sd"] for x in ds])
    np.testing.assert_allclose(sds.std(0), f.random_effects["g"]["sd_se"], rtol=0.3)
    with pytest.raises(ValueError, match="draw_ranef"):
        ds[0].ranef()
    # with ranef=True: the draw's slopes, new groups from N(0, its Sigma); predict uses them
    new = pl.DataFrame({"x": [0.0, 1.0], "x2": [0.0, 0.0], "g": [4, 10 ** 6]})
    dr = f.draw_parameters(2, seed=4, ranef=True, newdata=new)
    t = dr[0].ranef().collect()
    assert t.filter(pl.col("g") == 10 ** 6)["n_obs"][0] == 0
    re = dict(zip(t["g"], zip(t["re_g"], t["re_g:x"])))
    eta = dr[0].beta[0] + new["x"].to_numpy() * dr[0].beta[1]
    p = dr[0].predict(new, type="response").collect()["prediction"].to_numpy()
    want = expit(eta + np.array([re[4][0], re[10 ** 6][0] + re[10 ** 6][1]]))
    np.testing.assert_allclose(p, want, rtol=1e-10)


def test_refusals():
    d = _sim(G=60, seed=1)
    with pytest.raises(ValueError, match="adaptive quadrature"):
        logit_stream("y ~ x", d, random={"g": ["1", "x"]}, random_method="joint-mode",
                     verbose=False)
    with pytest.raises(ValueError, match="not covariates"):
        logit_stream("y ~ x", d, random={"g": ["1", "x2"]}, verbose=False)
    with pytest.raises(ValueError, match="distinct strings"):
        logit_stream("y ~ x", d, random={"g": ["1", "1"]}, verbose=False)
    with pytest.raises(ValueError, match="points per group"):
        mlogit_stream("y ~ x + x2", _sim(G=60, seed=1, J=4), random={"g": ["1", "x", "x2"]},
                      verbose=False)
    # random={"g": "1"} is the random intercept
    a = _fit(logit_stream, "y ~ x", d, random={"g": "1"})
    b = _fit(logit_stream, "y ~ x", d, random="g")
    np.testing.assert_allclose(a.beta, b.beta)
