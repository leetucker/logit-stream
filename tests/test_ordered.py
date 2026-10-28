"""ologit_stream against statsmodels' OrderedModel."""

import numpy as np
import polars as pl
import pytest
from statsmodels.miscmodels.ordinal_model import OrderedModel

from logit_stream import OrderedLogitResult, ologit_stream


def _sm_cuts(params, k):
    th = params[k:]
    return np.r_[th[0], th[0] + np.cumsum(np.exp(th[1:]))]


@pytest.fixture(scope="module")
def sm_fit(ord_data):
    X = ord_data.select("x1", "x2").to_numpy()
    return OrderedModel(ord_data["y"].to_numpy(), X, distr="logit").fit(
        method="bfgs", disp=0, gtol=1e-10, maxiter=5000)


def test_matches_statsmodels(ord_data, sm_fit):
    r = ologit_stream("y ~ x1 + x2", ord_data, verbose=False, tol=1e-14)
    assert isinstance(r, OrderedLogitResult)
    assert r.coefnames == ["x1", "x2", "cut1", "cut2", "cut3"]
    np.testing.assert_allclose(r.beta[:2], sm_fit.params[:2], rtol=1e-6)
    np.testing.assert_allclose(r.cutpoints, _sm_cuts(sm_fit.params, 2), rtol=1e-6)
    np.testing.assert_allclose(r.se[:2], sm_fit.bse[:2], rtol=1e-5)
    assert r.loglik == pytest.approx(sm_fit.llf, rel=1e-10)
    assert r.levels == [0, 1, 2, 3]


def test_robust_slopes_match_statsmodels(ord_data):
    """The slopes' block of the sandwich does not depend on how the
    cutpoints are parameterized, so it can be compared with statsmodels'
    (whose cutpoints are increments on the log scale)."""
    X = ord_data.select("x1", "x2").to_numpy()
    m = OrderedModel(ord_data["y"].to_numpy(), X, distr="logit").fit(
        method="bfgs", disp=0, gtol=1e-10, maxiter=5000, cov_type="HC0")
    r = ologit_stream("y ~ x1 + x2", ord_data, verbose=False, tol=1e-14, vcov="hetero")
    N = ord_data.height
    np.testing.assert_allclose(r.se[:2], m.bse[:2] * np.sqrt(N / (N - 1)), rtol=1e-5)


def test_enum_order_and_levels(ord_data):
    lab = ["low", "mid", "high", "top"]
    df = ord_data.with_columns(pl.col("y").replace_strict(range(4), lab).cast(pl.Enum(lab))
                               .alias("ye"),
                               pl.col("y").replace_strict(range(4), lab).alias("ys"))
    a = ologit_stream("ye ~ x1 + x2", df, verbose=False, tol=1e-14)
    b = ologit_stream("y ~ x1 + x2", df, verbose=False, tol=1e-14)
    assert a.levels == lab
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    # strings sort alphabetically unless levels= says otherwise
    c = ologit_stream("ys ~ x1 + x2", df, verbose=False, tol=1e-14, levels=lab)
    np.testing.assert_allclose(c.beta, b.beta, rtol=1e-10)
    rev = ologit_stream("y ~ x1 + x2", df, verbose=False, tol=1e-14, levels=[3, 2, 1, 0])
    np.testing.assert_allclose(rev.beta[:2], -b.beta[:2], rtol=1e-9)


def test_intercept_is_collinear_with_cutpoints(ord_data):
    df = ord_data.with_columns(pl.lit(1.0).alias("one"))
    with pytest.warns(UserWarning, match="collinear"):
        r = ologit_stream("y ~ x1 + x2 + one", df, verbose=False)
    assert r.collin_vars == ["one"]


def test_predict_and_resid(ord_data):
    r = ologit_stream("y ~ x1 + x2", ord_data, verbose=False, tol=1e-14)
    p = r.predict(ord_data.head(30)).collect()
    P = p.select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(P.sum(axis=1), 1)
    xb = r.predict(ord_data.head(30), type="link").collect()["xb"].to_numpy()
    np.testing.assert_allclose(xb, ord_data.head(30).select(
        r.beta[0] * pl.col("x1") + r.beta[1] * pl.col("x2")).to_series().to_numpy())
    from scipy.special import expit
    np.testing.assert_allclose(P[:, 0], expit(r.cutpoints[0] - xb), rtol=1e-12)
    cls = r.predict(ord_data.head(30), type="class").collect()["predicted"].to_list()
    assert cls == [r.levels[i] for i in P.argmax(axis=1)]
    res = r.resid().collect()
    assert res.columns[:3] == ["row_id", "y", "xb"] and res.height == ord_data.height


def test_weights(ord_data):
    df = ord_data.head(2000)
    rep = df.select(pl.all().repeat_by(pl.col("w").cast(pl.Int64))).explode(pl.all())
    a = ologit_stream("y ~ x1 + x2", df, verbose=False, tol=1e-14, weights="w",
                      weights_type="fweights")
    b = ologit_stream("y ~ x1 + x2", rep, verbose=False, tol=1e-14)
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-9)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-8)


def test_cluster(ord_data):
    r = ologit_stream("y ~ x1 + x2", ord_data, verbose=False, vcov={"CRV1": "cl"})
    assert r.vcov_type == "CRV1:cl" and r.n_clusters == {"cl": 30}
    assert np.all(r.se > 0)
