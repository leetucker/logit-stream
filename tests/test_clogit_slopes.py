"""Random slopes by region in the conditional logit (`clogit_stream(...,
random={"cz": "dist"})`, alt_slopes.py): the fit against exact maximum
likelihood (a dense grid around each region's mode) and, where the
regions' slopes do not factor (the alternatives' regions, two slopes),
against an independent Laplace approximation maximized numerically;
predict (seen and unseen regions, effects="average"), the draws, the
calibration and assignment, the clustered variance, the likelihood-ratio
test and refusals."""

import warnings

import numpy as np
import polars as pl
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from logit_stream import clogit_stream, lr_test

A_TRUE = np.array([0.0, 0.5, -0.3, 0.2])


def _sim(R=12, n_per=300, sd=0.5, seed=0, x2=False, alt_level=False):
    """Choice sets of four common alternatives (constants A_TRUE); utility
    (-1 + b_g) dist [+ (0.6 + c_g) x2]; each set's region (or each
    alternative's)."""
    rng = np.random.default_rng(seed)
    J = 4
    B = rng.normal(0, sd, (R, 2))
    rows = []
    for c in range(R * n_per):
        dist = rng.exponential(1.0, J)
        z2 = rng.normal(size=J)
        gr = rng.integers(0, R, J) if alt_level else np.full(J, c % R)
        u = ((-1 + B[gr, 0]) * dist + A_TRUE + ((0.6 + B[gr, 1]) * z2 if x2 else 0)
             + rng.gumbel(size=J))
        ch = (u == u.max()).astype(int)
        for j in range(J):
            rows.append((c, j, int(gr[j]), dist[j], z2[j], int(ch[j])))
    return pl.DataFrame(rows, schema=["case", "alt", "cz", "dist", "x2", "chosen"], orient="row")


def _arrays(df, zs, xs):
    df = df.sort("case", "alt")
    n, J = df["case"].n_unique(), df["alt"].n_unique()
    X = np.stack([df[x].to_numpy().reshape(n, J) for x in xs], -1)
    Z = np.stack([df[z].to_numpy().reshape(n, J) for z in zs], -1)
    y = df["chosen"].to_numpy().reshape(n, J).argmax(1)
    g = df["cz"].to_numpy().reshape(n, J)
    return X, Z, y, g


def _laplace_nll(df, zs, xs):
    """Minus Laplace's approximation of the marginal log-likelihood, in
    (beta, the constants less the first's, Sigma's log-Cholesky), from the
    joint mode of the slopes by Newton's method (any regions)."""
    X, Z, y, g = _arrays(df, zs, xs)
    n, J, K = X.shape
    q = Z.shape[2]
    R = g.max() + 1

    def nll(p):
        b, al = p[:K], np.r_[0, p[K:K + J - 1]]
        ps = p[K + J - 1:]
        Lc = np.diag(np.exp(ps[:q]))
        if q == 2:
            Lc[1, 0] = ps[2]
        S = Lc @ Lc.T
        Lam = np.linalg.inv(S)
        V = X @ b + al
        B = np.zeros((R, q))
        idx = np.arange(n)[:, None], np.arange(J)[None, :]
        Y = np.zeros((n, J))
        Y[np.arange(n), y] = 1

        def h(B):
            U = V + np.einsum("njq,njq->nj", Z, B[g])
            return ((U[np.arange(n), y] - logsumexp(U, 1)).sum()
                    - 0.5 * np.einsum("gi,ij,gj->", B, Lam, B))

        for _ in range(200):
            U = V + np.einsum("njq,njq->nj", Z, B[g])
            P = softmax(U, 1)
            # gradient and information in the slopes, region-major
            D = np.zeros((n, J, R, q))
            D[idx[0], idx[1], g] = Z
            D = D.reshape(n, J, R * q)
            Dbar = np.einsum("nj,njk->nk", P, D)
            gr = np.einsum("nj,njk->k", Y - P, D) - (B @ Lam).ravel()
            Dt = D - Dbar[:, None, :]
            H = np.einsum("nj,njk,njl->kl", P, Dt, Dt) + np.kron(np.eye(R), Lam)
            step = np.linalg.solve(H, gr).reshape(R, q)
            t, h0 = 1.0, h(B)
            while h(B + t * step) < h0 - 1e-12 and t > 1e-8:
                t /= 2
            B = B + t * step
            if np.abs(step).max() < 1e-11:
                break
        U = V + np.einsum("njq,njq->nj", Z, B[g])
        P = softmax(U, 1)
        D = np.zeros((n, J, R, q))
        D[idx[0], idx[1], g] = Z
        D = D.reshape(n, J, R * q)
        Dt = D - np.einsum("nj,njk->nk", P, D)[:, None, :]
        H = np.einsum("nj,njk,njl->kl", P, Dt, Dt) + np.kron(np.eye(R), Lam)
        U = V + np.einsum("njq,njq->nj", Z, B[g])
        ll = (U[np.arange(n), y] - logsumexp(U, 1)).sum()
        return -(ll - 0.5 * np.einsum("gi,ij,gj->", B, Lam, B)
                 - 0.5 * R * np.linalg.slogdet(S)[1] - 0.5 * np.linalg.slogdet(H)[1])

    return nll


def _exact_nll(df, npts=801, width=10.0):
    """Minus the exact marginal log-likelihood, one slope (dist) on the
    set's region: a dense grid around each region's mode."""
    X, _, y, g = _arrays(df, ["dist"], ["dist"])
    X = X[..., 0]
    g = g[:, 0]
    R = g.max() + 1
    idx = [np.flatnonzero(g == r) for r in range(R)]

    def nll(p):
        b0, al, t2 = p[0], np.r_[0, p[1:4]], np.exp(2 * p[4])
        tot = 0.0
        for r in range(R):
            Xr, yr = X[idx[r]], y[idx[r]]
            V = b0 * Xr + al
            bb = 0.0
            for _ in range(60):
                P = softmax(V + bb * Xr, 1)
                xb = (P * Xr).sum(1)
                gr = (Xr[np.arange(len(yr)), yr] - xb).sum() - bb / t2
                H = ((P * Xr * Xr).sum(1) - xb ** 2).sum() + 1 / t2
                bb += gr / H
                if abs(gr) < 1e-12:
                    break
            grid = bb + np.linspace(-width, width, npts) / np.sqrt(H)
            U = V[None] + grid[:, None, None] * Xr[None]
            lp = (U[:, np.arange(len(yr)), yr] - logsumexp(U, axis=2)).sum(1)
            lp += -grid ** 2 / (2 * t2) - 0.5 * np.log(2 * np.pi * t2)
            tot += logsumexp(lp) + np.log(grid[1] - grid[0])
        return -tot

    return nll


def _params(f, q=1):
    """The fit's (beta, constants less alternative 0's, Sigma's log-Cholesky)."""
    fx = f.fixef().collect().sort("alt")["fe_alt"].to_numpy()
    S = np.asarray(f.random_effects["cz"]["Sigma"])
    Lc = np.linalg.cholesky(S)
    ps = np.log(np.diag(Lc)) if q == 1 else np.r_[np.log(np.diag(Lc)), Lc[1, 0]]
    return np.r_[f.beta, fx[1:] - fx[0], ps]


@pytest.fixture(scope="module")
def chooser():
    df = _sim(R=8, n_per=250)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = clogit_stream("chosen ~ dist | alt", df, case="case", random={"cz": "dist"},
                          verbose=False)
    return df, f


def test_matches_exact_maximum_likelihood(chooser):
    """The set's region, one slope: the estimates are exact ML's (a dense
    grid) to Laplace's accuracy, and the log-likelihood Laplace's."""
    df, f = chooser
    nll = _exact_nll(df, npts=301, width=8.0)
    p = _params(f)
    r = minimize(nll, p, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8,
                                                          "maxiter": 5_000})
    se = np.r_[f.se, 0.05, 0.05, 0.05, 0.1]
    assert np.all(np.abs(p - r.x) < 0.005 * se), (p - r.x) / se
    assert f.random_effects["cz"]["sd"][0] == pytest.approx(np.exp(r.x[4]), rel=2e-3)
    assert f.loglik == pytest.approx(-r.fun, abs=0.02)
    # Laplace's own maximum: to the optimizer's accuracy
    lap = _laplace_nll(df, ["dist"], ["dist"])
    assert lap(p) <= lap(r.x) + 1e-6
    assert f.loglik == pytest.approx(-lap(p), abs=1e-5)
    assert f.random_effects["cz"]["region_of"] == "choice set"


@pytest.mark.parametrize("case", ["alternative", "two_slopes"])
def test_matches_laplace_where_regions_mix(case):
    """The alternatives' regions (a set mixes regions), and two slopes:
    the fit is the maximum of the Laplace approximation, computed
    independently with all the regions' slopes jointly."""
    if case == "alternative":
        df = _sim(R=5, n_per=200, seed=3, alt_level=True)
        zs, xs, rnd = ["dist"], ["dist"], {"cz": "dist"}
        fml = "chosen ~ dist | alt"
    else:
        df = _sim(R=6, n_per=250, seed=4, x2=True)
        zs, xs, rnd = ["dist", "x2"], ["dist", "x2"], {"cz": ["dist", "x2"]}
        fml = "chosen ~ dist + x2 | alt"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = clogit_stream(fml, df, case="case", random=rnd, verbose=False, random_tol=1e-9)
    nll = _laplace_nll(df, zs, xs)
    p = _params(f, q=len(zs))
    r = minimize(nll, p, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-11,
                                                          "maxiter": 40_000})
    assert nll(p) <= r.fun + 1e-6
    assert f.loglik == pytest.approx(-nll(p), abs=1e-5)
    assert np.all(np.abs(p - r.x) < 2e-3), p - r.x
    if case == "alternative":
        assert f.random_effects["cz"]["region_of"] == "alternative"


def test_without_absorbed_constants(chooser):
    df, f = chooser
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = clogit_stream("chosen ~ dist + C(alt)", df, case="case", random={"cz": "dist"},
                          verbose=False)
    assert g.beta[0] == pytest.approx(f.beta[0], abs=1e-6)
    assert g.random_effects["cz"]["sd"][0] == pytest.approx(f.random_effects["cz"]["sd"][0],
                                                             rel=1e-5)
    assert g.loglik == pytest.approx(f.loglik, abs=1e-5)


def test_predict_and_average(chooser):
    df, f = chooser
    r = f.ranef().collect()
    assert r.columns == ["cz", "re_cz:dist", "sd_cz:dist", "n_obs", "n_chosen"]
    one = df.filter(pl.col("case") == 7)
    fx = dict(zip(*f.fixef().collect().select("alt", "fe_alt").to_numpy().T))
    al = np.array([fx[a] for a in one["alt"]])
    re = dict(zip(r["cz"], r["re_cz:dist"]))
    sd = dict(zip(r["cz"], r["sd_cz:dist"]))
    g = one["cz"][0]
    d = one["dist"].to_numpy()
    p = f.predict(one).collect()["prob"].to_numpy()
    np.testing.assert_allclose(p, softmax((f.beta[0] + re[g]) * d + al), rtol=1e-10)
    # a region the fit did not see: no slope deviation
    p0 = f.predict(one.with_columns(pl.lit(999).alias("cz"))).collect()["prob"].to_numpy()
    np.testing.assert_allclose(p0, softmax(f.beta[0] * d + al), rtol=1e-10)
    # averaged over the region's conditional distribution, and the prior's
    z = np.linspace(-10, 10, 4001)
    w = np.exp(-z * z / 2)
    w /= w.sum()
    tau = f.random_effects["cz"]["sd"][0]
    for gg, mu, s in [(g, re[g], sd[g]), (999, 0.0, tau)]:
        pa = f.predict(one.with_columns(pl.lit(gg).alias("cz")), effects="average") \
            .collect()["prob"].to_numpy()
        want = w @ softmax((f.beta[0] + mu + s * z)[:, None] * d[None] + al[None], axis=1)
        np.testing.assert_allclose(pa, want, rtol=1e-6)


def test_draws(chooser):
    df, f = chooser
    new = df.filter(pl.col("case") < 40).with_columns(pl.col("cz") + 100)
    draws = f.draw_parameters(300, seed=2, newdata=new)
    info = f._spec["draw"]
    C = np.array([np.r_[d.beta, d.ranef().collect().sort("cz")["re_cz:dist"].to_numpy()[:8]]
                  for d in draws])
    V = info["V"]
    np.testing.assert_allclose(C.std(0), np.sqrt(np.diag(V)), rtol=0.15)
    assert np.mean(C, 0) == pytest.approx(info["coef"], abs=4 * np.sqrt(np.diag(V)).max()
                                          / np.sqrt(300))
    # new regions: N(0, the draw's Sigma), different in each draw
    t = [d.ranef().collect().filter(pl.col("cz") >= 100) for d in draws[:2]]
    assert t[0].height == 8 and not t[0]["re_cz:dist"].equals(t[1]["re_cz:dist"])
    assert t[0]["sd_cz:dist"][0] == pytest.approx(draws[0].random_effects["cz"]["sd"][0])
    sds = np.array([d.random_effects["cz"]["sd"][0] for d in draws])
    assert sds.std() == pytest.approx(f.random_effects["cz"]["sd_se"][0], rel=0.25)
    # the draw's predictions use its slopes; calibration and assignment run under them
    tot = df.group_by("alt").agg(pl.col("chosen").sum().cast(pl.Float64).alias("n"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = draws[0].predict(df.drop("chosen"), calibrate={"alt": tot}).collect()
        gap = c.group_by("alt").agg(pl.col("prob").sum()).join(tot, on="alt") \
            .select((pl.col("prob") - pl.col("n")).abs().max()).item()
        assert gap < 1e-6
        a = draws[0].sample_assignments(df.drop("chosen"), {"alt": tot}, seed=1).collect()
    got = a.group_by("alt").agg(pl.col("assigned").sum()).join(tot, on="alt")
    assert (got["assigned"] == got["n"]).all()


def test_inference_and_lr_test(chooser):
    df, f = chooser
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = clogit_stream("chosen ~ dist | alt", df, case="case", verbose=False)
        cl = clogit_stream("chosen ~ dist | alt", df, case="case", random={"cz": "dist"},
                           vcov={"CRV1": "cz"}, verbose=False)
    t = lr_test(plain, f)
    assert t["boundary"] and t["df"] == 1 and t["p"] < 1e-10
    assert cl.beta == pytest.approx(f.beta)
    assert cl.se[0] > 0 and cl.se[0] != pytest.approx(f.se[0], rel=1e-3)
    # beta's iid standard error near that of the region slopes' spread: with
    # 8 regions the mean slope is known to about sd / sqrt(8)
    tau = f.random_effects["cz"]["sd"][0]
    assert f.se[0] == pytest.approx(np.sqrt(tau ** 2 / 8), rel=0.35)
    v = f.varcorr()
    assert v["term"].to_list() == ["sd(dist)"]


def test_refusals(chooser):
    df, _ = chooser
    kw = dict(case="case", verbose=False)
    with pytest.raises(ValueError, match="grouping: covariate"):
        clogit_stream("chosen ~ dist | alt", df, random="cz", **kw)
    with pytest.raises(ValueError, match="not covariates"):
        clogit_stream("chosen ~ dist | alt", df, random={"cz": "x2"}, **kw)
    with pytest.raises(ValueError, match="heteroskedasticity"):
        clogit_stream("chosen ~ dist | alt", df, random={"cz": "dist"}, vcov="hetero", **kw)
    with pytest.raises(ValueError, match="whole regions"):
        clogit_stream("chosen ~ dist | alt", df, random={"cz": "dist"}, vcov={"CRV1": "case"},
                      **kw)
    with pytest.raises(ValueError, match="laplace"):
        clogit_stream("chosen ~ dist | alt", df, random={"cz": "dist"},
                      random_method="adaptive", **kw)
    with pytest.raises(ValueError, match="both absorbed"):
        clogit_stream("chosen ~ dist | cz", df, random={"cz": "dist"}, **kw)
