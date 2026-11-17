"""Adaptive Gauss-Hermite quadrature for one scalar random intercept
(``random_method="adaptive"``) and the random intercepts' conditional sds
in `ranef`: one node is Laplace's approximation; the pass's gradient and
information are the derivatives of its value (nodes held); the fit is the
maximum likelihood, checked against brute-force integration and
optimization (coefficients, cutpoints, a fixed effect, sd, log-likelihood
and standard errors); invariances; `logit_stream(random=)`; refusals."""

import _adaptive_helpers as H
import numpy as np
import polars as pl
import pytest
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from test_random import _binary_data, _ordered_data

from logit_stream import logit_stream, mlogit_stream, ologit_stream


def _small(J, seed=0, G=300, cells=None, geom=0.4):
    """Tiny groups (1 + Geometric(geom) rows), sd 0.9, optionally within
    `cells` levels of a fixed effect."""
    rng = np.random.default_rng(seed)
    N = 1 + rng.geometric(geom, G)
    g = np.repeat(np.arange(G), N)
    n = len(g)
    x1, x2 = rng.normal(size=n), rng.binomial(1, 0.4, n).astype(float)
    lat = 0.7 * x1 - 0.4 * x2 + rng.normal(0, 0.9, G)[g] + rng.logistic(size=n)
    cols = {}
    if cells:
        cell = rng.integers(0, cells, G)[g]
        lat = lat + rng.normal(0, 0.5, cells)[cell]
        cols["cell"] = cell
    y = np.searchsorted([0.0] if J == 2 else [-1.0, 0.2, 1.4], lat)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g, **cols})


# ---------------------------------------------------------------- brute force
_Z = np.linspace(-9, 9, 721)
_LW = -0.5 * _Z * _Z - 0.5 * np.log(2 * np.pi) + np.log(_Z[1] - _Z[0])


def _brute_nll(p, X, y, g, J, cell=None, n_cells=0):
    """Minus the marginal log-likelihood by a dense grid, in (slopes,
    cutpoints, cell effects less the first's, log sd); with cells, the
    first cutpoint is the first cell's location."""
    K = X.shape[1]
    b, cuts = p[:K], p[K:K + J - 1]
    if np.any(np.diff(cuts) <= 0):
        return np.inf
    eta = X @ b
    if cell is not None:
        eta = eta - np.r_[0.0, p[K + J - 1:K + J - 1 + n_cells - 1]][cell]
    tau = np.exp(p[-1])
    cp = np.r_[-np.inf, cuts, np.inf]
    A = eta[:, None] + tau * _Z[None, :]
    lp = np.log(np.clip(expit(cp[y + 1][:, None] - A) - expit(cp[y][:, None] - A), 1e-300,
                        None))
    F = np.zeros((g.max() + 1, len(_Z)))
    np.add.at(F, g, lp)
    return -logsumexp(F + _LW[None, :], axis=1).sum()


def _brute(df, J, cells=0):
    X = df.select("x1", "x2").to_numpy()
    y, g = df["y"].to_numpy(), df["g"].to_numpy()
    cell = df["cell"].to_numpy() if cells else None
    args = (X, y, g, J, cell, cells)
    p0 = np.r_[0.5, -0.3, np.linspace(-1, 1, J - 1) if J > 2 else [0.0],
               np.zeros(max(cells - 1, 0)), np.log(0.8)]
    r = minimize(_brute_nll, p0, args=args, method="BFGS", options={"gtol": 1e-7})
    p = r.x
    # the inverse of the Hessian by central differences, for the standard errors
    h, n = 1e-4, len(p)
    Hm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            Hm[i, j] = Hm[j, i] = (_brute_nll(p + ei + ej, *args) - _brute_nll(p + ei - ej, *args)
                                   - _brute_nll(p - ei + ej, *args)
                                   + _brute_nll(p - ei - ej, *args)) / (4 * h * h)
    return p, -r.fun, np.sqrt(np.diag(np.linalg.inv(Hm)))


# ---------------------------------------------------------------- the pass
@pytest.mark.parametrize("which", ["binary", "ordered"])
def test_one_node_is_laplace(which):
    """At the Laplace estimates, with the nodes at the modes and the
    conditional sds, the one-node rule's value is the Laplace approximation;
    more nodes converge."""
    df = _binary_data() if which == "binary" else _ordered_data()
    re, drv, fit = H.setup("y ~ x1 + x2", df, "g")
    ad = H.adaptive(re, drv, 1)
    b, al, nu = ad.unpack(fit.theta)
    mode = re._theta_of(fit.theta, ad.f)[:, 0]
    sd = re.conditional_sd(fit.theta)[0][:, 0]
    # (the Laplace fit records its value at its last round's start, a step of
    # ~1e-7 in the log-likelihood before the final estimates)
    assert ad.evaluate(b, al, nu, mode, sd).ll == pytest.approx(fit.laplace_loglik, abs=1e-6)
    v = [H.adaptive(re, drv, k).evaluate(b, al, nu, mode, sd).ll for k in (9, 15, 25)]
    assert abs(v[0] - v[2]) < 1e-5 and abs(v[1] - v[2]) < 1e-8


@pytest.mark.parametrize("J,fml", [(2, "y ~ x1 + x2"), (4, "y ~ x1 + x2"),
                                   (2, "y ~ x1 + x2 | cell"), (4, "y ~ x1 + x2 | cell")])
def test_gradient_and_information_are_the_derivatives(J, fml):
    """With the nodes held, the pass's gradient and information (b, the
    fixed effect's levels, log sd) are the first and second derivatives of
    its value: central differences."""
    re, drv, fit = H.setup(fml, _small(J, G=200, cells=6 if "cell" in fml else None), "g",
                           laplace=False)
    ad = H.adaptive(re, drv, 7)
    b, al, nu = ad.unpack(fit.theta)
    mode = re._theta_of(fit.theta, ad.f)[:, 0]
    sd = re.conditional_sd(fit.theta)[0][:, 0]
    P, idx = len(b), ad._alpha_free()
    na = len(idx)

    def ev(v):
        aa = None if al is None else al.copy()
        if na:
            aa[idx] = v[P:P + na]
        return ad.evaluate(v[:P], aa, v[-1], mode, sd)

    def grad(E):
        return np.r_[E.gb, E.ga[idx] if na else [], E.gnu]

    v0 = np.r_[b, al[idx] if na else [], nu]
    E = ev(v0)
    n = len(v0)
    Jm = np.zeros((n, n))
    Jm[:P, :P], Jm[:P, -1], Jm[-1, :P], Jm[-1, -1] = E.Jbb, E.Jbn, E.Jbn, E.Jnn
    if na:
        Jm[P:P + na, P:P + na] = np.diag(E.Jaa[idx])
        Jm[:P, P:P + na], Jm[P:P + na, :P] = E.Jba[idx].T, E.Jba[idx]
        Jm[P:P + na, -1] = Jm[-1, P:P + na] = E.Jan[idx]
    h = 1e-5
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        Ep, Em = ev(v0 + e), ev(v0 - e)
        assert grad(E)[i] == pytest.approx((Ep.ll - Em.ll) / (2 * h), abs=1e-6)
        np.testing.assert_allclose(Jm[:, i], -(grad(Ep) - grad(Em)) / (2 * h), atol=1e-6)


def test_needs_the_rows_grouped():
    re, drv, _ = H.setup("y ~ x1 + x2", _small(2, G=50), "g", laplace=False)
    re.p0.fe_sorted = False
    with pytest.raises(ValueError, match="one block"):
        H.adaptive(re, drv, 9)


# ---------------------------------------------------------------- the fit
@pytest.mark.parametrize("J", [2, 4])
def test_is_the_maximum_likelihood(J):
    """Tiny groups, where Laplace's approximation is biased: the estimates,
    sd, log-likelihood and standard errors are brute force's (a dense grid,
    optimized by scipy, the Hessian by differences)."""
    df = _small(J)
    p, ll, se_b = _brute(df, J)
    if J == 2:
        f = logit_stream("y ~ x1 + x2", df, random="g", random_method="adaptive",
                         verbose=False, iwls_tol=1e-12)
        est, se = np.r_[f.beta[1:], -f.beta[0]], np.r_[f.se[1:], f.se[0]]
    else:
        f = ologit_stream("y ~ x1 + x2", df, random="g", random_method="adaptive",
                          verbose=False, tol=1e-12)
        est, se = np.asarray(f.beta), np.asarray(f.se)
    re = f.random_effects["g"]
    assert re["method"] == "adaptive" and re["nodes"] == 9
    assert np.all(np.abs(est - p[:-1]) < 1e-4 * se)
    assert re["sd"][0] == pytest.approx(np.exp(p[-1]), rel=1e-5)
    assert f.loglik == pytest.approx(ll, abs=1e-5)
    np.testing.assert_allclose(se, se_b[:-1], rtol=2e-3)
    assert re["sd_se"][0] == pytest.approx(np.exp(p[-1]) * se_b[-1], rel=2e-3)
    # and Laplace's is visibly lower with groups this small
    lap = (logit_stream if J == 2 else ologit_stream)("y ~ x1 + x2", df, random="g",
                                                      verbose=False)
    assert lap.random_effects["g"]["sd"][0] < re["sd"][0]


def test_within_a_fixed_effect_is_the_maximum_likelihood():
    """Random groups within the levels of an absorbed fixed effect (the
    binary logit, ``y ~ x | cell``): the slopes, the cell effects (relative
    to the first) and the sd are brute force's."""
    df = _small(2, seed=3, cells=4)
    p, ll, _ = _brute(df, 2, cells=4)
    f = logit_stream("y ~ x1 + x2 | cell", df, random="g", random_method="adaptive",
                     verbose=False, iwls_tol=1e-12)
    np.testing.assert_allclose(f.beta, p[:2], atol=1e-4 * float(np.min(f.se)))
    # brute force's cell terms enter the linear predictor as minus p[3:6]
    a = f.fixef("cell").collect().sort("cell")["fe_cell"].to_numpy()
    np.testing.assert_allclose((a - a[0])[1:], -p[3:6], atol=1e-4)
    assert f.random_effects["g"]["sd"][0] == pytest.approx(np.exp(p[-1]), rel=1e-5)
    # (an sd near 1 with tiny groups: 9 nodes are 1e-5 off in the log-likelihood,
    # 25 agree with brute force to 1e-10)
    assert f.loglik == pytest.approx(ll, abs=5e-5)


def test_nodes_offset_and_large_groups():
    """More nodes change nothing that matters; a constant offset shifts the
    cutpoints only; with large groups the fit is close to Laplace's."""
    df = _small(4, seed=5).with_columns(off=pl.lit(0.3))
    f9 = ologit_stream("y ~ x1 + x2", df, random="g", random_method="adaptive",
                       verbose=False, tol=1e-12)
    f15 = ologit_stream("y ~ x1 + x2", df, random="g", random_method="adaptive",
                        random_nodes=15, verbose=False, tol=1e-12)
    assert np.all(np.abs(f9.beta - f15.beta) < 1e-3 * f9.se)
    assert f9.random_effects["g"]["sd"][0] == pytest.approx(
        f15.random_effects["g"]["sd"][0], rel=1e-4)
    fo = ologit_stream("y ~ x1 + x2", df, random="g", random_method="adaptive",
                       offset="off", verbose=False, tol=1e-12)
    np.testing.assert_allclose(fo.beta[:2], f9.beta[:2], atol=1e-7)
    np.testing.assert_allclose(fo.beta[2:], f9.beta[2:] + 0.3, atol=1e-7)
    assert fo.random_effects["g"]["sd"][0] == pytest.approx(f9.random_effects["g"]["sd"][0],
                                                            rel=1e-7)
    # about 80 rows a group: within a few hundredths of a standard error
    big = _binary_data()
    lap = logit_stream("y ~ x1 + x2", big, random="g", verbose=False, iwls_tol=1e-12)
    ad = logit_stream("y ~ x1 + x2", big, random="g", random_method="adaptive",
                      verbose=False, iwls_tol=1e-12)
    assert np.all(np.abs(ad.beta - lap.beta) < 0.05 * lap.se)
    assert ad.random_effects["g"]["sd"][0] == pytest.approx(lap.random_effects["g"]["sd"][0],
                                                            rel=0.01)
    assert ad.loglik >= lap.loglik - 0.5


# ---------------------------------------------------------------- logit_stream, ranef
def test_logit_stream_random_matches_lme4_nagq1():
    """logit_stream(random=) fits the binary logit's random intercepts with
    this package's engine: lme4's nAGQ = 1 fit (as test_random's
    two-level mlogit_stream), Intercept first, no small-sample factor."""
    f = logit_stream("y ~ x1 + x2", _binary_data(), random="g", verbose=False, iwls_tol=1e-12,
                     random_tol=1e-8)
    b = np.array([-0.480425667054, 0.799880277051, -0.395924654280])
    se = np.array([0.09757510799, 0.03724080413, 0.06809067919])
    assert f.coefnames == ["Intercept", "x1", "x2"]
    assert np.all(np.abs(f.beta - b) < 0.002 * se)
    np.testing.assert_allclose(f.se, se, rtol=0.01)
    assert f.random_effects["g"]["sd"][0] == pytest.approx(0.6814538185, rel=1e-4)
    assert f.loglik == pytest.approx(-2815.36912312, abs=1e-3)


@pytest.mark.parametrize("method", ["laplace", "adaptive"])
def test_ranef_conditional_sd(method):
    """`ranef` has each group's conditional sd: for the binary logit,
    (sum of the rows' p (1 - p) at the mode + 1 / sigma^2)^-1/2; `predict`
    is unchanged by the column."""
    df = _binary_data()
    f = logit_stream("y ~ x1 + x2", df, random="g", random_method=method, verbose=False,
                     iwls_tol=1e-12)
    r = f.ranef().collect().sort("g")
    assert r.columns == ["g", "re_g", "sd_g", "n_obs"]
    b, s2 = f.beta, f.random_effects["g"]["Sigma"][0][0]
    g = df["g"].to_numpy()
    eta = b[0] + b[1] * df["x1"].to_numpy() + b[2] * df["x2"].to_numpy()
    p = expit(eta + r["re_g"].to_numpy()[g])
    sd = 1 / np.sqrt(np.bincount(g, p * (1 - p)) + 1 / s2)
    np.testing.assert_allclose(r["sd_g"].to_numpy(), sd, rtol=1e-6 if method == "laplace"
                               else 1e-4)
    pr = f.predict(df, type="link").collect()["prediction"].to_numpy()
    np.testing.assert_allclose(pr, eta + r["re_g"].to_numpy()[g], atol=1e-10)


def test_ranef_conditional_sd_multinomial():
    """Each non-base level's conditional sd; 0 for the base."""
    df = _ordered_data().with_columns(pl.col("y").cast(pl.Utf8))
    f = mlogit_stream("y ~ x1", df, random="g", verbose=False)
    r = f.ranef().collect()
    sds = [c for c in r.columns if c.startswith("sd_g:")]
    assert len(sds) == 4 and (r[sds[0]] == 0).all()
    assert all((r[c] > 0).all() for c in sds[1:])
    assert all((r[c] < np.sqrt(f.random_effects["g"]["Sigma"][i][i])).all()
               for i, c in enumerate(sds[1:]))


def test_refusals():
    df = _small(2, G=60)
    with pytest.raises(ValueError, match="logit_stream"):
        mlogit_stream("y ~ x1", df, random="g", random_method="adaptive", verbose=False)
    with pytest.raises(ValueError, match="at least 5"):
        logit_stream("y ~ x1", df, random="g", random_method="adaptive", random_nodes=3,
                     verbose=False)
    with pytest.raises(ValueError, match="iid"):
        logit_stream("y ~ x1", df, random="g", random_method="adaptive",
                     vcov={"CRV1": "g"}, verbose=False)
    with pytest.raises(ValueError, match="one random grouping"):
        ologit_stream("y ~ x1", df.with_columns(h=pl.col("g") // 3), random=["g", "h"],
                      random_method="adaptive", verbose=False)
    with pytest.raises(ValueError, match="probit"):
        from logit_stream import probit_stream
        probit_stream("y ~ x1", df, random="g", verbose=False)


# ---------------------------------------------------------------- edge cases
def test_starts_at_the_variance_floor_and_leaves_it():
    """A start with the variance at its floor (as a singular joint-mode fit
    gives) still reaches the maximum when the data have group variance."""
    df = _small(2)
    p, _, _ = _brute(df, 2)
    re, drv, fit = H.setup("y ~ x1 + x2", df, "g", laplace=False)
    re.set_covs([np.array([[1e-6]])])
    res = H.adaptive(re, drv, 9).fit(fit.theta, 1e-10)
    assert res["nu_free"] and np.exp(res["nu"]) == pytest.approx(np.exp(p[-1]), rel=1e-5)


def test_no_group_variance():
    """Groups that differ no more than chance: the variance at the floor, a
    warning, finite standard errors for the coefficients, and the reports."""
    rng = np.random.default_rng(11)         # (a seed whose estimate is at the floor)
    g = np.repeat(np.arange(400), 3)
    x1 = rng.normal(size=len(g))
    y = (rng.random(len(g)) < expit(0.2 + 0.7 * x1)).astype(int)
    df = pl.DataFrame({"y": y, "x1": x1, "g": g})
    with pytest.warns(UserWarning, match="variance of zero"):
        f = logit_stream("y ~ x1", df, random="g", random_method="adaptive", verbose=False)
    assert f.random_effects["g"]["sd"][0] < 0.01
    assert np.all(np.isfinite(f.se))
    f.summary_text()
    f.varcorr()


def test_dropped_cells_and_weights():
    """A cell whose outcome never varies is dropped (separation), and the
    fit is brute force's on the other cells; frequency weights of 2 are
    the rows twice."""
    df = _small(2, seed=4, cells=4)
    df = df.with_columns(pl.when(pl.col("cell") == 3).then(0).otherwise("y").alias("y"))
    with pytest.warns(UserWarning, match="dropped"):
        f = logit_stream("y ~ x1 + x2 | cell", df, random="g", random_method="adaptive",
                         verbose=False, iwls_tol=1e-12)
    kept = df.filter(pl.col("cell") < 3)
    p, ll, _ = _brute(kept, 2, cells=3)
    np.testing.assert_allclose(f.beta, p[:2], atol=1e-4 * float(np.min(f.se)))
    assert f.random_effects["g"]["sd"][0] == pytest.approx(np.exp(p[-1]), rel=1e-5)
    assert f.loglik == pytest.approx(ll, abs=5e-5)
    small = _small(2, seed=6, G=150)
    w = logit_stream("y ~ x1 + x2", small.with_columns(w=pl.lit(2.0)), random="g",
                     random_method="adaptive", weights="w", weights_type="fweights",
                     verbose=False, iwls_tol=1e-12)
    d = logit_stream("y ~ x1 + x2", pl.concat([small, small]).sort("g"), random="g",
                     random_method="adaptive", verbose=False, iwls_tol=1e-12)
    np.testing.assert_allclose(w.beta, d.beta, atol=1e-8)
    assert w.random_effects["g"]["sd"][0] == pytest.approx(d.random_effects["g"]["sd"][0],
                                                           rel=1e-8)


def test_predict_within_a_fixed_effect():
    """The linear predictor is the covariates', the cell's fixed effect and
    the group's random effect; the fixed effect's column is its own."""
    df = _small(2, seed=7, cells=4)
    f = logit_stream("y ~ x1 + x2 | cell", df, random="g", random_method="adaptive",
                     verbose=False)
    fe = dict(f.fixef("cell").collect().select("cell", "fe_cell").iter_rows())
    ra = dict(f.ranef("g").collect().select("g", "re_g").iter_rows())
    out = f.predict(df, type="link").collect()
    xb = f.beta[0] * df["x1"].to_numpy() + f.beta[1] * df["x2"].to_numpy()
    want = xb + np.array([fe[c] for c in df["cell"]]) + np.array([ra[g] for g in df["g"]])
    np.testing.assert_allclose(out["prediction"].to_numpy(), want, atol=1e-10)
