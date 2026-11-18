"""`draw_parameters` and the fixed effects' standard errors in `fixef()`:
the draws' covariance is the inverse information with the fixed effect's
levels included, checked against fits with the levels as dummy variables
(no random intercepts) and against brute-force integration and a numerical
Hessian (an adaptive random intercept within a fixed effect); the sampler
reproduces that covariance; a drawn result's predict, fixef and
random_effects are the draw's; refusals."""

import numpy as np
import polars as pl
import pytest
from test_random_adaptive import _brute, _small

from logit_stream import logit_stream, mlogit_stream, ologit_stream


def _cells(seed=0, n=6000, C=8, J=2):
    """Rows in C cells, with a covariate whose mean differs by cell (so the
    cell effects and the slope are correlated)."""
    rng = np.random.default_rng(seed)
    cell = rng.integers(0, C, n)
    x = rng.normal(size=n) + 2.0 + 0.3 * cell
    lat = rng.normal(0, 1, C)[cell] + 0.7 * x - 1.4 + rng.logistic(size=n)
    y = np.searchsorted([0.0] if J == 2 else [-0.8, 0.4, 1.5], lat)
    return pl.DataFrame({"y": y, "x": x, "cell": cell})


def _dummy_cov(fd, C, prefix="C(cell)[T.", slope="x"):
    """The covariance of (slope, the C cell levels) from a fit with an
    intercept and C - 1 cell dummies: level c = Intercept + dummy c."""
    names = fd.coefnames
    T = np.zeros((C + 1, len(names)))
    T[0, names.index(slope)] = 1
    for c in range(C):
        T[1 + c, names.index("Intercept")] = 1
        if c:
            T[1 + c, names.index(f"{prefix}{c}]")] = 1
    return T @ np.asarray(fd.vcov) @ T.T, T @ np.asarray(fd.beta)


def _implied_fe(f):
    """The joint covariance of (b, the kept levels) that the fixed-effect
    draws follow (one parameter per level)."""
    info = f._spec["draw"]
    from logit_stream.draws import _iid_factor
    c = _iid_factor(f, info)
    V = np.asarray(f.vcov)
    keep = info["keep"]
    BD = info["BD"][keep, 0, :]
    own = (info["S"][keep, 0, 0] * c) ** 2
    return V, np.diag(own) + BD @ V @ BD.T, -BD @ V


@pytest.mark.parametrize("vcov", ["iid", {"CRV1": "cell"}])
def test_fixed_effect_matches_dummies(vcov):
    df = _cells()
    f = logit_stream("y ~ x | cell", df, verbose=False, vcov="iid")
    fd = logit_stream("y ~ x + C(cell)", df, verbose=False)
    Vd, est = _dummy_cov(fd, 8)
    V, Vaa, Vab = _implied_fe(f)
    np.testing.assert_allclose(V[0, 0], Vd[0, 0], rtol=1e-6)
    np.testing.assert_allclose(Vaa, Vd[1:, 1:], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(Vab[:, 0], Vd[1:, 0], rtol=1e-6)
    t = f.fixef().collect().sort("cell")
    assert t.columns == ["cell", "fe_cell", "se_cell", "n_obs"]
    np.testing.assert_allclose(t["fe_cell"], est[1:], atol=1e-8)
    np.testing.assert_allclose(t["se_cell"], np.sqrt(np.diag(Vd)[1:]), rtol=1e-6)
    if vcov != "iid":
        # with a robust vcov the slopes' part of each level's variance is the robust one
        fr = logit_stream("y ~ x | cell", df, verbose=False, vcov=vcov)
        _, Vaa_r, _ = _implied_fe(fr)
        se_r = fr.fixef().collect().sort("cell")["se_cell"].to_numpy()
        np.testing.assert_allclose(se_r, np.sqrt(np.diag(Vaa_r)), rtol=1e-10)
        assert not np.allclose(se_r, t["se_cell"].to_numpy(), rtol=1e-3)


def test_ordered_fixed_effect_matches_dummies():
    """The ordered logit within a fixed effect (its first cutpoint fixed at 0,
    the levels carry the location) against cutpoints plus dummies."""
    df = _cells(J=4, seed=1)
    f = ologit_stream("y ~ x | cell", df, verbose=False)
    fd = ologit_stream("y ~ x + C(cell)", df, verbose=False)
    # dummies: level c's location is -cut_1 + ... ; compare differences from cell 0
    a = f.fixef().collect().sort("cell")
    names = fd.coefnames
    d = np.array([0.0] + [fd.beta[names.index(f"C(cell)[T.{c}]")] for c in range(1, 8)])
    np.testing.assert_allclose(a["fe_cell"].to_numpy() - a["fe_cell"][0], d, atol=1e-6)
    # the variance of a level's difference from cell 0
    V, Vaa, _ = _implied_fe(f)
    for c in range(1, 8):
        i = names.index(f"C(cell)[T.{c}]")
        v = Vaa[c, c] + Vaa[0, 0] - 2 * Vaa[c, 0]
        assert v == pytest.approx(float(fd.vcov[i, i]), rel=1e-5)


def test_multinomial_fixed_effect():
    """Per-level blocks (J - 1 parameters each): the reference level stays 0
    in every draw with a null standard error; the free levels' standard
    errors are the draws' spread."""
    rng = np.random.default_rng(2)
    n, C = 4000, 6
    cell = rng.integers(0, C, n)
    x = rng.normal(size=n)
    u = rng.normal(0, 0.7, (C, 3))[cell] + np.c_[np.zeros(n), 0.5 * x, -0.4 * x]
    y = np.argmax(u + rng.gumbel(size=(n, 3)), axis=1)
    df = pl.DataFrame({"y": y, "x": x, "cell": cell})
    f = mlogit_stream("y ~ x | cell", df, verbose=False)
    t = f.fixef().collect().sort("cell")
    assert t["se_cell:0"].is_null().all() and (t["se_cell:1"] > 0).all()
    draws = f.draw_parameters(1500, seed=3)
    A = np.stack([d.fixef().collect().sort("cell")["fe_cell:1"].to_numpy() for d in draws])
    assert all((d.fixef().collect()["fe_cell:0"] == 0).all() for d in draws[:5])
    np.testing.assert_allclose(A.std(axis=0, ddof=1), t["se_cell:1"], rtol=0.08)
    np.testing.assert_allclose(A.mean(axis=0), t["fe_cell:1"], atol=4 * t["se_cell:1"].max()
                               / np.sqrt(1500))


def test_sampler_reproduces_the_covariance():
    """Monte Carlo: the draws' mean and covariance of (slope, levels), and
    with the covariate's cell means far from 0 the levels move with the
    slope (each cell's level at its mean covariate is better determined
    than the level itself)."""
    df = _cells(seed=4)
    f = logit_stream("y ~ x | cell", df, verbose=False)
    V, Vaa, Vab = _implied_fe(f)
    n = 3000
    draws = f.draw_parameters(n, seed=5)
    b = np.array([d.beta[0] for d in draws])
    A = np.stack([d.fixef().collect().sort("cell")["fe_cell"].to_numpy() for d in draws])
    t = f.fixef().collect().sort("cell")
    se = t["se_cell"].to_numpy()
    assert abs(b.mean() - f.beta[0]) < 4 * np.sqrt(V[0, 0] / n)
    np.testing.assert_allclose(A.mean(axis=0), t["fe_cell"], atol=4 * se.max() / np.sqrt(n))
    np.testing.assert_allclose(b.std(ddof=1), np.sqrt(V[0, 0]), rtol=0.06)
    np.testing.assert_allclose(A.std(axis=0, ddof=1), se, rtol=0.06)
    r = [np.corrcoef(A[:, c], b)[0, 1] for c in range(8)]
    r0 = Vab[:, 0] / np.sqrt(np.diag(Vaa) * V[0, 0])
    np.testing.assert_allclose(r, r0, atol=0.05)
    assert max(r0) < -0.5                         # strongly negative: x's mean is far from 0
    xbar = df.group_by("cell").agg(pl.col("x").mean()).sort("cell")["x"].to_numpy()
    lev = A + np.outer(b, xbar)
    assert np.all(lev.std(axis=0) < 0.9 * A.std(axis=0))


def _implied_adaptive(f):
    """The joint covariance of (b, nu, the free levels) that the adaptive
    draws follow."""
    info = f._spec["draw"]
    V = info["V"]
    M = info["Jea"].T / info["Daa"][:, None]
    return V, np.diag(1 / info["Daa"]) + M @ V @ M.T, -M @ V, info


def test_adaptive_within_a_fixed_effect_matches_brute_force():
    """(slopes, cell levels, log sd): their joint covariance against the
    inverse of brute force's numerical Hessian (brute force's cells are the
    first cell's location and the others' differences from it)."""
    df = _small(2, seed=3, cells=4)
    p, _, _ = _brute(df, 2, cells=4)
    f = logit_stream("y ~ x1 + x2 | cell", df, random="g", random_method="adaptive",
                     verbose=False, iwls_tol=1e-12)
    # brute force's covariance, again (as _brute computes it), in its parameters
    from test_random_adaptive import _brute_nll
    X = df.select("x1", "x2").to_numpy()
    args = (X, df["y"].to_numpy(), df["g"].to_numpy(), 2, df["cell"].to_numpy(), 4)
    h, k = 1e-4, len(p)
    Hm = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
            Hm[i, j] = Hm[j, i] = (_brute_nll(p + ei + ej, *args) - _brute_nll(p + ei - ej, *args)
                                   - _brute_nll(p - ei + ej, *args)
                                   + _brute_nll(p - ei - ej, *args)) / (4 * h * h)
    Vp = np.linalg.inv(Hm)
    # to (b1, b2, a_0..a_3, nu): a_c = -cut - d_c (d_0 = 0)
    T = np.zeros((7, k))
    T[0, 0] = T[1, 1] = 1
    for c in range(4):
        T[2 + c, 2] = -1
        if c:
            T[2 + c, 2 + c] = -1
    T[6, -1] = 1
    Vb = T @ Vp @ T.T
    V, Vaa, Vat, info = _implied_adaptive(f)
    order = np.argsort(f.fixef().collect()["cell"].to_numpy())
    assert list(info["idx"]) == [0, 1, 2, 3] and info["nu_free"]
    rel = dict(rtol=3e-3, atol=0)
    np.testing.assert_allclose(V, Vb[np.ix_([0, 1, 6], [0, 1, 6])], **rel)
    np.testing.assert_allclose(Vaa[np.ix_(order, order)], Vb[2:6, 2:6], **rel)
    np.testing.assert_allclose(Vat[order], Vb[2:6][:, [0, 1, 6]], rtol=3e-3, atol=1e-7)
    t = f.fixef().collect().sort("cell")
    np.testing.assert_allclose(t["se_cell"], np.sqrt(np.diag(Vb)[2:6]), rtol=2e-3)
    # the coefficients' vcov is the same (marginal) covariance
    np.testing.assert_allclose(np.asarray(f.vcov), V[:2, :2], rtol=1e-8)


def test_adaptive_draws():
    df = _small(2, seed=5, cells=5, G=600)
    f = logit_stream("y ~ x1 + x2 | cell", df, random="g", random_method="adaptive",
                     verbose=False)
    V, Vaa, Vat, info = _implied_adaptive(f)
    n = 3000
    draws = f.draw_parameters(n, seed=7)
    th = np.array([np.r_[d.beta, np.log(d.random_effects["g"]["sd"][0])] for d in draws])
    A = np.stack([d.fixef().collect()["fe_cell"].to_numpy() for d in draws])
    np.testing.assert_allclose(np.cov(th.T), V, rtol=0.1, atol=0.05 * np.sqrt(np.outer(
        np.diag(V), np.diag(V))).max())
    np.testing.assert_allclose(A.std(axis=0, ddof=1), np.sqrt(np.diag(Vaa)), rtol=0.06)
    r = [np.corrcoef(A[:, c], th[:, 2])[0, 1] for c in range(5)]
    r0 = Vat[:, 2] / np.sqrt(np.diag(Vaa) * V[2, 2])
    np.testing.assert_allclose(r, r0, atol=0.05)
    # the draw's sd and Sigma, and its predict: the drawn coefficients and cell
    # level, without the group intercepts
    d = draws[0]
    sd = d.random_effects["g"]["sd"][0]
    assert d.random_effects["g"]["Sigma"][0, 0] == pytest.approx(sd * sd)
    assert sd != f.random_effects["g"]["sd"][0]
    p = d.predict(df.head(20), type="link").collect()
    lev = dict(zip(*d.fixef().collect().select("cell", "fe_cell").to_numpy().T))
    want = df.head(20).select("x1", "x2").to_numpy() @ d.beta + np.array(
        [lev[c] for c in df.head(20)["cell"]])
    np.testing.assert_allclose(p["prediction"], want, rtol=1e-12)
    with pytest.raises(ValueError, match="does not carry the group intercepts"):
        d.ranef()
    with pytest.raises(ValueError, match="already a draw"):
        d.draw_parameters(1)
    assert "se_cell" not in d.fixef().collect().columns


def test_without_a_fixed_effect():
    df = _cells(seed=6)
    f = logit_stream("y ~ x + C(cell)", df, verbose=False, vcov="hetero")
    n = 3000
    B = np.stack([d.beta for d in f.draw_parameters(n, seed=8)])
    np.testing.assert_allclose(B.mean(axis=0), f.beta, atol=4 * float(np.max(f.se)) / np.sqrt(n))
    np.testing.assert_allclose(np.cov(B.T), f.vcov, rtol=0.12,
                               atol=0.06 * float(np.max(np.diag(f.vcov))))
    # an adaptive random intercept without a fixed effect: the slopes and log sd
    g = logit_stream("y ~ x1 + x2", _small(2, seed=1), random="g", random_method="adaptive",
                     verbose=False)
    d = g.draw_parameters(2, seed=1)
    assert len(d[0].beta) == 3 and d[0].fe_names == []


def test_seed_and_refusals():
    df = _cells(seed=7)
    f = logit_stream("y ~ x | cell", df, verbose=False)
    a = [d.beta[0] for d in f.draw_parameters(3, seed=11)]
    b = [d.beta[0] for d in f.draw_parameters(3, seed=np.random.default_rng(11))]
    assert a == b and len(set(a)) == 3
    with pytest.raises(ValueError, match="positive integer"):
        f.draw_parameters(0)
    lap = logit_stream("y ~ x1 + x2", _small(2), random="g", verbose=False)
    with pytest.raises(ValueError, match="random_method='adaptive'"):
        lap.draw_parameters(1)
