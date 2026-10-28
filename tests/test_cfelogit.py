"""Chamberlain's conditional fixed-effects logit (`cfelogit_stream`)."""

import numpy as np
import polars as pl
import pytest
from scipy.special import gammaln

from logit_stream import ConditionalFELogitResult, cfelogit_stream


def _panel(N=800, seed=5):
    """Unequal panels (2 to 9 periods), units with every share of 1s (so
    some are fitted as their complement), a regressor correlated with the
    unit effect, and a unit-level one."""
    rng = np.random.default_rng(seed)
    T = rng.integers(2, 10, N)
    i = np.repeat(np.arange(N), T)
    a = rng.normal(0, 1.5, N)[i]
    n = len(i)
    x1 = rng.normal(size=n) + 0.5 * a
    x2 = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2 + a)))).astype(int)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "z": rng.normal(size=N)[i],
                         "i": i}).sample(fraction=1.0, shuffle=True, seed=1)


@pytest.fixture(scope="module")
def panel():
    return _panel()


def test_matches_statsmodels(panel):
    cm = pytest.importorskip("statsmodels.discrete.conditional_models")
    f = cfelogit_stream("y ~ x1 + x2 | i", panel, verbose=False, tol=1e-14)
    s = cm.ConditionalLogit(panel["y"].to_numpy(), panel.select("x1", "x2").to_numpy(),
                            groups=panel["i"].to_numpy()).fit(disp=0, method="newton",
                                                              tol=1e-12, maxiter=100)
    np.testing.assert_allclose(f.beta, s.params, atol=1e-9)
    np.testing.assert_allclose(f.se, s.bse, rtol=1e-8)
    assert f.loglik == pytest.approx(s.llf, abs=1e-8)
    assert isinstance(f, ConditionalFELogitResult)


def test_units_and_null(panel):
    f = cfelogit_stream("y ~ x1 + x2 | i", panel, verbose=False)
    per = panel.group_by("i").agg(pl.col("y").sum().alias("k"), pl.len().alias("T"))
    live = per.filter((pl.col("k") > 0) & (pl.col("k") < pl.col("T")))
    assert f.n_units == live.height == f.n_obs
    assert f.n_units_dropped == per.height - live.height
    assert f.n_rows == live["T"].sum()
    T, k = live["T"].to_numpy().astype(float), live["k"].to_numpy().astype(float)
    ll0 = -np.sum(gammaln(T + 1) - gammaln(k + 1) - gammaln(T - k + 1))
    assert f.loglik_null == pytest.approx(ll0)
    assert 0 < f.pseudo_r2 < 1


def test_unit_level_covariate_is_dropped(panel):
    with pytest.warns(UserWarning, match="collinear"):
        f = cfelogit_stream("y ~ x1 + x2 + z | i", panel, verbose=False, tol=1e-14)
    g = cfelogit_stream("y ~ x1 + x2 | i", panel, verbose=False, tol=1e-14)
    assert f.coefnames == ["x1", "x2"]
    np.testing.assert_allclose(f.beta, g.beta, atol=1e-10)


def test_weights_and_offset(panel):
    two = panel.with_columns(f=pl.lit(2.0))
    a = cfelogit_stream("y ~ x1 + x2 | i", panel, verbose=False, tol=1e-14)
    b = cfelogit_stream("y ~ x1 + x2 | i", two, weights="f", weights_type="fweights",
                        verbose=False, tol=1e-14)
    assert b.loglik == pytest.approx(2 * a.loglik)
    np.testing.assert_allclose(b.se, a.se / np.sqrt(2), rtol=1e-8)
    c = cfelogit_stream("y ~ x1 + x2 | i", panel, offset=pl.col("x2") * 0.5, verbose=False,
                        tol=1e-14)
    assert c.coef()["x2"] == pytest.approx(a.coef()["x2"] - 0.5, abs=1e-8)
    xb = c.predict(panel.head(3)).collect()["xb"].to_numpy()
    np.testing.assert_allclose(xb, (c.beta[0] * panel["x1"][:3] + (c.beta[1] + 0.5)
                                    * panel["x2"][:3]).to_numpy())


def test_restrictions(panel):
    with pytest.raises(ValueError, match="one fixed effect"):
        cfelogit_stream("y ~ x1", panel, verbose=False)
    f = cfelogit_stream("y ~ x1 | i", panel, verbose=False)
    with pytest.raises(ValueError, match="type='link'"):
        f.predict(panel.head(3), type="response")
    with pytest.raises(ValueError, match="no residuals"):
        f.resid()
    with pytest.raises(ValueError, match="CRV3"):
        cfelogit_stream("y ~ x1 | i", panel, vcov={"CRV3": "i"}, verbose=False)
