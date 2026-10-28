"""mlogit_stream against statsmodels and scikit-learn, and its options."""

import json

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from logit_stream import LogitMulti, MultinomialLogitResult, mlogit_stream

FML = "y ~ x1 + x2 + C(g)"


def _design(df):
    g = df["g"].to_numpy()
    return sm.add_constant(np.column_stack([df["x1"].to_numpy(), df["x2"].to_numpy()]
                                           + [(g == k).astype(float) for k in (1, 2, 3)]))


@pytest.fixture(scope="module")
def fit(mnl_data):
    return mlogit_stream(FML, mnl_data, verbose=False, tol=1e-14)


@pytest.fixture(scope="module")
def sm_fit(mnl_data):
    return sm.MNLogit(mnl_data["yi"].to_numpy(), _design(mnl_data)).fit(disp=0, tol=1e-12,
                                                                         maxiter=200)


def test_matches_statsmodels(fit, sm_fit):
    assert isinstance(fit, MultinomialLogitResult)
    np.testing.assert_allclose(fit.beta, sm_fit.params.T.ravel(), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(fit.se, sm_fit.bse.T.ravel(), rtol=1e-7)
    assert fit.loglik == pytest.approx(sm_fit.llf, rel=1e-12)
    assert fit.loglik_null == pytest.approx(sm_fit.llnull, rel=1e-8)
    assert fit.pseudo_r2 == pytest.approx(sm_fit.prsquared, rel=1e-7)
    assert fit.aic == pytest.approx(sm_fit.aic, rel=1e-10)
    assert fit.bic == pytest.approx(sm_fit.bic, rel=1e-10)
    assert fit.levels == ["a", "b", "c", "d"] and fit.base == "a"
    assert fit.coefnames[:2] == ["b:Intercept", "b:x1"]
    assert fit.converged


def test_matches_sklearn(mnl_data, fit):
    from sklearn.linear_model import LogisticRegression
    X = _design(mnl_data)[:, 1:]
    try:
        sk = LogisticRegression(C=np.inf, tol=1e-12, max_iter=10_000)
        sk.fit(X, mnl_data["yi"].to_numpy())
    except (TypeError, ValueError):
        sk = LogisticRegression(penalty=None, tol=1e-12, max_iter=10_000)
        sk.fit(X, mnl_data["yi"].to_numpy())
    W = np.column_stack([sk.intercept_, sk.coef_])          # J x (1 + k), symmetric
    rel = (W[1:] - W[0]).ravel()
    np.testing.assert_allclose(fit.beta, rel, rtol=1e-5, atol=1e-6)
    probs = fit.predict(mnl_data).collect().select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(probs, sk.predict_proba(X), atol=1e-6)


def test_two_levels_is_binary_logit(mnl_data):
    from sklearn.linear_model import LogisticRegression
    df = mnl_data.filter(pl.col("yi") <= 1)
    r = mlogit_stream("yi ~ x1 + x2", df, verbose=False, tol=1e-14)
    m = sm.Logit(df["yi"].to_numpy(), sm.add_constant(df.select("x1", "x2").to_numpy())) \
        .fit(disp=0, tol=1e-12)
    np.testing.assert_allclose(r.beta, m.params, rtol=1e-8)
    np.testing.assert_allclose(r.se, m.bse, rtol=1e-8)
    # the L2 penalty is scikit-learn's with C = 1 / l2 (the intercept unpenalized)
    rp = mlogit_stream("yi ~ x1 + x2", df, verbose=False, tol=1e-14, l2=50.0)
    sk = LogisticRegression(C=1 / 50.0, tol=1e-12, max_iter=10_000).fit(
        df.select("x1", "x2").to_numpy(), df["yi"].to_numpy())
    np.testing.assert_allclose(rp.beta, np.r_[sk.intercept_, sk.coef_.ravel()], rtol=1e-6)
    assert np.all(np.abs(rp.beta[1:]) < np.abs(r.beta[1:]))
    assert rp.l2 == 50.0


def test_robust_and_clustered(mnl_data):
    r = mlogit_stream(FML, mnl_data, verbose=False, tol=1e-14, vcov="hetero", cluster="cl")
    X, y, N = _design(mnl_data), mnl_data["yi"].to_numpy(), mnl_data.height
    h = sm.MNLogit(y, X).fit(disp=0, tol=1e-12, cov_type="HC0")
    np.testing.assert_allclose(r.se, h.bse.T.ravel() * np.sqrt(N / (N - 1)), rtol=1e-7)
    c = sm.MNLogit(y, X).fit(disp=0, tol=1e-12, cov_type="cluster",
                             cov_kwds={"groups": mnl_data["cl"].to_numpy(),
                                       "use_correction": False})
    G = 30
    rc = r.with_vcov({"CRV1": "cl"})
    np.testing.assert_allclose(rc.se, c.bse.T.ravel() * np.sqrt(G / (G - 1)), rtol=1e-7)
    assert rc.n_clusters == {"cl": 30} and rc.df_t == 29
    assert set(r.all_vcovs) == {"iid", "hetero", "CRV1:cl"}
    with pytest.raises(KeyError):
        r.with_vcov({"CRV1": "g"})


def test_frequency_weights_are_replicated_rows(mnl_data):
    df = mnl_data.head(1500)
    rep = df.select(pl.all().repeat_by(pl.col("w").cast(pl.Int64))).explode(pl.all())
    a = mlogit_stream("y ~ x1 + x2", df, verbose=False, tol=1e-14, weights="w",
                      weights_type="fweights", vcov="hetero")
    b = mlogit_stream("y ~ x1 + x2", rep, verbose=False, tol=1e-14, vcov="hetero")
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-9)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-8)
    assert a.n_obs == b.n_obs == int(df["w"].sum())
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)
    aw = mlogit_stream("y ~ x1 + x2", df, verbose=False, tol=1e-14, weights="w")
    np.testing.assert_allclose(aw.beta, a.beta, rtol=1e-9)
    assert aw.n_obs == df.height
    assert aw.weights == "w" and aw.weights_type == "aweights"


def test_storage_options_agree(mnl_data, fit):
    for kw in ({"storage": "memory"}, {"block_mb": 0.05, "chunk_mb": 0.02},
               {"storage": "memory", "block_mb": 0.05}):
        r = mlogit_stream(FML, mnl_data, verbose=False, tol=1e-14, **kw)
        np.testing.assert_allclose(r.beta, fit.beta, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(r.se, fit.se, rtol=1e-10)
    small = mlogit_stream(FML, mnl_data, verbose=False, tol=1e-14, block_mb=0.05)
    assert small.diagnostics["store"]["blocks"] >= 5
    r32 = mlogit_stream(FML, mnl_data, verbose=False, tol=1e-14, dtype="float32")
    np.testing.assert_allclose(r32.beta, fit.beta, rtol=1e-4, atol=1e-6)


def test_sources(tmp_path, mnl_data, fit):
    path = tmp_path / "d.parquet"
    mnl_data.write_parquet(path)
    for src in (str(path), path, pl.scan_parquet(path), mnl_data.lazy()):
        r = mlogit_stream(FML, src, verbose=False, tol=1e-14)
        np.testing.assert_allclose(r.beta, fit.beta, rtol=1e-10)


def test_levels_and_base(mnl_data):
    r = mlogit_stream("y ~ x1 + x2", mnl_data, verbose=False, levels=["c", "a", "b", "d"])
    assert r.base == "c" and r.coefnames[0] == "a:Intercept"
    p = r.predict(mnl_data.head(5)).collect()
    assert p.columns[-4:] == ["prob_c", "prob_a", "prob_b", "prob_d"]
    with pytest.raises(ValueError, match="not in levels"):
        mlogit_stream("y ~ x1", mnl_data, verbose=False, levels=["a", "b", "c"])
    with pytest.raises(ValueError, match="never occur"):
        mlogit_stream("y ~ x1", mnl_data, verbose=False, levels=["a", "b", "c", "d", "e"])
    # integer outcome, numeric order
    ri = mlogit_stream("yi ~ x1 + x2", mnl_data, verbose=False)
    assert ri.levels == [0, 1, 2, 3]


def test_missing_values_and_sample(mnl_data):
    df = mnl_data.with_columns(
        pl.when(pl.int_range(pl.len()) < 25).then(None).otherwise(pl.col("x1")).alias("x1"),
        pl.when(pl.int_range(pl.len()).is_between(25, 34)).then(float("nan"))
          .otherwise(pl.col("x2")).alias("x2"))
    r = mlogit_stream("y ~ x1 + x2", df, verbose=False)
    assert r.n_obs == df.height - 35
    s = r.sample().collect()
    assert s.columns[0] == "row_id" and s.height == df.height
    assert s["in_sample"].sum() == r.n_obs
    assert s.filter(~pl.col("in_sample"))["row_id"].to_list() == list(range(35))
    assert set(s["dropped_because"].drop_nulls().unique()) == {"missing"}
    res = r.resid().collect()
    assert res.height == r.n_obs and res["row_id"].min() == 35


def test_collinear_covariate_is_dropped(mnl_data):
    df = mnl_data.with_columns((pl.col("x1") + 2 * pl.col("x2")).alias("x3"))
    with pytest.warns(UserWarning, match="collinear"):
        r = mlogit_stream("y ~ x1 + x2 + x3", df, verbose=False, tol=1e-14)
    base = mlogit_stream("y ~ x1 + x2", df, verbose=False, tol=1e-14)
    assert r.collin_vars == ["x3"]
    np.testing.assert_allclose(r.beta, base.beta, rtol=1e-8)
    assert "b:x3" not in r.coefnames
    # predictions use the kept columns
    np.testing.assert_allclose(r.predict(df.head(20)).collect()["prob_b"].to_numpy(),
                               base.predict(df.head(20)).collect()["prob_b"].to_numpy(),
                               rtol=1e-8)


def test_predict_and_resid(mnl_data, fit):
    p = fit.predict(mnl_data.head(50)).collect()
    probs = p.select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert p.columns[:len(mnl_data.columns)] == mnl_data.columns
    c = fit.predict(mnl_data.head(50), type="class").collect()
    assert c["predicted"].to_list() == [fit.levels[i] for i in probs.argmax(axis=1)]
    lk = fit.predict(mnl_data.head(50), type="link").collect()
    np.testing.assert_allclose(np.exp(lk["xb_b"].to_numpy()),
                               p["prob_b"].to_numpy() / p["prob_a"].to_numpy(), rtol=1e-10)
    nd = mnl_data.head(3).with_columns(pl.lit(None, pl.Float64).alias("x1"))
    assert fit.predict(nd).collect()["prob_a"].null_count() == 3
    r = fit.resid().collect()
    assert r.columns == ["row_id", "y", "prob_a", "prob_b", "prob_c", "prob_d",
                         "resid_a", "resid_b", "resid_c", "resid_d"]
    np.testing.assert_allclose(r.select(pl.col("^resid_.*$")).to_numpy().sum(axis=1), 0,
                               atol=1e-12)
    # the scores are zero at the optimum: the residuals sum to zero by level
    np.testing.assert_allclose(r.select(pl.col("^resid_.*$")).sum().to_numpy(), 0, atol=1e-6)
    assert fit.predict(prefix="p").collect().columns[-1] == "p_d"


def test_reports(fit):
    d = json.loads(fit.summary_json(coefficients=True))
    assert d["model"] == "multinomial logit" and d["base"] == "a"
    assert d["levels"] == ["a", "b", "c", "d"]
    assert len(d["coefficients"]) == len(fit.coefnames)
    assert d["newton"]["converged"] is True
    text = fit.summary_text()
    assert "Multinomial logit" in text and "pseudo R2" in text
    t = fit.tidy()
    assert t.columns == ["Coefficient", "Estimate", "Std. Error", "t value", "Pr(>|t|)"]
    ci = fit.confint(keep="x1")
    assert ci.height == 3
    w = fit.wald_test(keep=":x2$")
    assert w["distribution"] == "chi2" and w["df1"] == 3
    m = fit.coef_matrix()
    assert m.columns == ["Coefficient", "b", "c", "d"] and m.height == 6


def test_multiple_estimation(mnl_data):
    r = mlogit_stream("y ~ csw(x1, x2)", mnl_data, verbose=False)
    assert isinstance(r, LogitMulti) and len(r) == 2
    assert [m.fml for m in r] == ["y ~ x1", "y ~ x1 + x2"]
    assert r.tidy().columns[0] == "fml"
    assert set(r.summary_dict()) == {"y ~ x1", "y ~ x1 + x2"}


def test_no_intercept(mnl_data):
    r = mlogit_stream("y ~ x1 + x2 - 1", mnl_data, verbose=False)
    assert not any("Intercept" in c for c in r.coefnames)
