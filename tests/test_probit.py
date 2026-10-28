"""The ordered probit (`oprobit_stream`) and the binary probit wrapper."""

from types import SimpleNamespace

import hdfe_stream.core.gpu as gpu
import numpy as np
import polars as pl
import pytest

from logit_stream import OrderedProbitResult, oprobit_stream, probit_stream


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(2)
    n = 3000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    g = rng.integers(0, 15, n)
    y = np.searchsorted([-0.6, 0.3, 1.1],
                        0.6 * x1 - 0.4 * x2 + rng.normal(0, 0.5, 15)[g] + rng.normal(size=n))
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})


def test_matches_statsmodels(data):
    sm_ord = pytest.importorskip("statsmodels.miscmodels.ordinal_model")
    f = oprobit_stream("y ~ x1 + x2", data, verbose=False, tol=1e-14)
    X = data.select("x1", "x2").to_numpy()
    m = sm_ord.OrderedModel(data["y"].to_numpy(), X, distr="probit").fit(
        method="bfgs", disp=0, gtol=1e-10, maxiter=5000)
    np.testing.assert_allclose(f.beta[:2], m.params[:2], atol=1e-7)
    np.testing.assert_allclose(f.cutpoints, m.model.transform_threshold_params(m.params)[1:-1],
                               atol=1e-7)
    np.testing.assert_allclose(f.se[:2], m.bse[:2], rtol=1e-5)
    assert isinstance(f, OrderedProbitResult) and f.model == "ordered probit"
    p = f.predict(data.head(5)).collect().select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_fixed_effect_equals_its_indicators(data):
    a = oprobit_stream("y ~ x1 + x2 | g", data, verbose=False, tol=1e-14)
    b = oprobit_stream("y ~ x1 + x2 + C(g)", data, verbose=False, tol=1e-14)
    for k in ("x1", "x2"):
        assert a.coef()[k] == pytest.approx(b.coef()[k], abs=1e-8)
        assert a.se[a.coefnames.index(k)] == pytest.approx(b.se[b.coefnames.index(k)], rel=1e-6)
    assert a.loglik == pytest.approx(b.loglik, abs=1e-7)
    # the null with only the fixed effect is the fit with only its indicators
    c = oprobit_stream("y ~ C(g)", data, verbose=False, tol=1e-14)
    assert a.loglik_fe_null == pytest.approx(c.loglik, abs=1e-6)


def test_on_the_device(data, monkeypatch):
    monkeypatch.setattr(gpu, "cuda_backend",
                        lambda: SimpleNamespace(xp=np, asnumpy=np.asarray, name="numpy"))
    a = oprobit_stream("y ~ x1 + x2", data, verbose=False, tol=1e-14)
    b = oprobit_stream("y ~ x1 + x2", data, verbose=False, tol=1e-14, device="cuda",
                       block_mb=0.05)
    np.testing.assert_allclose(b.beta, a.beta, atol=1e-9)
    np.testing.assert_allclose(b.se, a.se, rtol=1e-8)


def test_binary_probit_is_feglm(data):
    sm = pytest.importorskip("statsmodels.api")
    yb = (data["y"] > 1).cast(pl.Int64)
    f = probit_stream("yb ~ x1 + x2", data.with_columns(yb=yb), verbose=False, iwls_tol=1e-14)
    s = sm.Probit(yb.to_numpy().astype(float),
                  sm.add_constant(data.select("x1", "x2").to_numpy())).fit(disp=0, tol=1e-12)
    np.testing.assert_allclose(f.beta, s.params, atol=1e-8)
    assert f.family == "probit"


def test_restrictions(data):
    with pytest.raises(ValueError, match="not the probit"):
        oprobit_stream("y ~ x1", data, random="g", verbose=False)
    with pytest.raises(ValueError, match="canonical"):
        oprobit_stream("y ~ x1", data, firth=True, verbose=False)


@pytest.mark.parametrize("fn,distr", [("ologit", "logit"), ("oprobit", "probit")])
def test_ordered_residuals(data, fn, distr):
    """resid_prob is statsmodels' (Li and Shepherd's probability-scale
    residual); resid_gen, the score in xb, is orthogonal to the covariates
    at the estimates."""
    sm_ord = pytest.importorskip("statsmodels.miscmodels.ordinal_model")
    import logit_stream as ls
    f = getattr(ls, f"{fn}_stream")("y ~ x1 + x2", data, verbose=False, tol=1e-14)
    r = f.resid().collect().sort("row_id")
    m = sm_ord.OrderedModel(data["y"].to_numpy(), data.select("x1", "x2").to_numpy(),
                            distr=distr).fit(method="bfgs", disp=0, gtol=1e-10, maxiter=5000)
    np.testing.assert_allclose(r["resid_prob"].to_numpy(), m.resid_prob, atol=1e-8)
    assert r["resid_prob"].min() >= -1 and r["resid_prob"].max() <= 1
    for v in ("x1", "x2"):
        assert abs(float((r["resid_gen"] * data[v]).sum())) < 1e-6
