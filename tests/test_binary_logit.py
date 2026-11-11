"""`logit_stream` fits a binary logit with this package's Newton engine (the
two-level ordered logit, its cutpoint the intercept) and reports it with
hdfe-stream's conventions; `feglm_stream` is the reference."""

import numpy as np
import polars as pl
import pytest
from hdfe_stream import HDFEResult, feglm_stream

from logit_stream import BinaryLogitResult, logit_stream

TOL = dict(iwls_tol=1e-13, iwls_maxiter=100)


@pytest.fixture(scope="module")
def data():
    r = np.random.default_rng(7)
    n = 6000
    df = pl.DataFrame({"x1": r.normal(size=n), "x2": r.normal(size=n),
                       "g": r.integers(0, 40, n), "h": r.integers(0, 12, n),
                       "c": r.integers(0, 150, n), "w": r.uniform(.5, 2, n),
                       "fw": r.integers(1, 4, n), "off": r.normal(scale=.1, size=n)})
    eta = df["x1"].to_numpy() * .8 - .4 * df["x2"].to_numpy() + (df["g"].to_numpy() % 7 - 3) * .2
    y = (r.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    df = df.with_columns(pl.Series("y", y))
    # two fixed-effect levels whose outcome never varies, and a singleton level
    df = df.with_columns(pl.when(pl.col("g") == 0).then(0).when(pl.col("g") == 1).then(1)
                         .otherwise(pl.col("y")).alias("y"))
    return pl.concat([df, df.head(1).with_columns(pl.lit(999, dtype=pl.Int64).alias("g"))])


def both(fml, df, tmp_path, **kw):
    new = logit_stream(fml, df, workdir=tmp_path, verbose=False, **kw)
    old = feglm_stream(fml, df, "logit", workdir=tmp_path, verbose=False, **TOL, **kw)
    assert isinstance(new, BinaryLogitResult) and isinstance(old, HDFEResult)
    return new, old


CASES = {
    "plain": ("y ~ x1 + x2", {}),
    "hetero": ("y ~ x1 + x2", {"vcov": "hetero"}),
    "cluster": ("y ~ x1 + x2", {"vcov": {"CRV1": "c"}}),
    "categorical": ("y ~ x1 + C(h)", {}),
    "offset": ("y ~ x1 + x2", {"offset": "off"}),
    "fe": ("y ~ x1 + x2 | g", {}),
    "fe hetero": ("y ~ x1 + x2 | g", {"vcov": "hetero"}),
    "fe cluster": ("y ~ x1 + x2 | g", {"vcov": {"CRV1": "c"}}),
    "fe cluster nested": ("y ~ x1 + x2 | g", {"vcov": {"CRV1": "g"}}),
    "fe two-way cluster": ("y ~ x1 + x2 | g", {"vcov": {"CRV1": "c+g"}}),
    "fe aweights": ("y ~ x1 + x2 | g", {"weights": "w"}),
    "fe fweights": ("y ~ x1 + x2 | g", {"weights": "fw", "weights_type": "fweights"}),
    "fe offset": ("y ~ x1 + x2 | g", {"offset": "off"}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_matches_feglm_stream(case, data, tmp_path):
    fml, kw = CASES[case]
    new, old = both(fml, data, tmp_path, **kw)
    assert new.coefnames == old.coefnames
    assert np.allclose(new.beta, old.beta, rtol=0, atol=1e-9)
    assert new.n_obs == old.n_obs and new.k_fe == old.k_fe
    assert new.collin_vars == old.collin_vars
    assert new.vcov_type == old.vcov_type
    for key in new.all_vcovs:
        V, df = old.all_vcovs[key]
        assert np.allclose(new.all_vcovs[key][0], V, rtol=0, atol=1e-5 * np.abs(V).max()), key
        assert new.all_vcovs[key][1] == df, key
    for stat in ("loglik", "loglik_null", "pseudo_r2", "aic", "bic", "deviance"):
        assert np.isclose(getattr(new, stat), getattr(old, stat), rtol=1e-8), stat
    if new.fe_names:
        assert np.isclose(new.pseudo_r2_within, old.pseudo_r2_within, rtol=1e-8, equal_nan=True)
        one = lambda d: {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in d.items()}
        assert one(new.n_levels) == one(old.n_levels)


def test_estimates_are_at_the_maximum(data, tmp_path):
    """The Newton decrement is the stopping rule: the default tolerance gives
    the estimates to machine precision (hdfe-stream's rule, a deviance
    change of 1e-8, stops about 1e-7 away)."""
    new = logit_stream("y ~ x1 + x2", data, workdir=tmp_path, verbose=False)
    tight = feglm_stream("y ~ x1 + x2", data, "logit", workdir=tmp_path, verbose=False, **TOL)
    assert np.allclose(new.beta, tight.beta, rtol=0, atol=1e-12)
    assert new.diagnostics["irls"]["converged"]
    assert new.diagnostics["irls"]["decrement"] < 1e-8


@pytest.mark.parametrize("fml", ["y ~ x1 + x2", "y ~ x1 + x2 | g"])
def test_predict_resid_fixef_sample(fml, data, tmp_path):
    new, old = both(fml, data, tmp_path)
    a, b = new.resid().collect().sort("row_id"), old.resid().collect().sort("row_id")
    for c in ("y", "eta", "fitted", "resid"):
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy(), atol=1e-9), c
    assert np.allclose(a["resid_working"].to_numpy(), b["resid_working"].to_numpy(), atol=1e-5)
    for kind in ("link", "response"):
        x = new.predict(data, type=kind).collect()["prediction"].to_numpy()
        y = old.predict(data, type=kind).collect()["prediction"].to_numpy()
        assert np.array_equal(np.isnan(x), np.isnan(y))
        assert np.allclose(x, y, atol=1e-9, equal_nan=True)
    cls = new.predict(data, type="class").collect()["prediction"]
    prob = new.predict(data, type="response").collect()["prediction"]
    ok = prob.is_not_null()
    assert (cls.filter(ok) == (prob.filter(ok) > .5).cast(pl.Int64)).all()
    assert cls.filter(~ok).is_null().all()
    sa, sb = new.sample().collect(), old.sample().collect()
    assert sa.columns == sb.columns and sa["in_sample"].equals(sb["in_sample"])
    if new.fe_names:
        fa, fb = new.fixef("g").collect().sort("g"), old.fixef("g").collect().sort("g")
        assert fa["g"].equals(fb["g"]) and np.allclose(fa["fe_g"], fb["fe_g"], atol=1e-9)
        assert fa["n_obs"].equals(fb["n_obs"])


def test_intercept_comes_first_with_its_sign(data, tmp_path):
    new = logit_stream("y ~ x1 + x2", data, workdir=tmp_path, verbose=False)
    assert new.coefnames == ["Intercept", "x1", "x2"]
    assert np.isclose(new.cutpoints[0], -new.beta[0])
    assert new.depvar == "y" and new.family == "logit"
    assert np.isclose(new.deviance, -2 * new.loglik)


def test_routes_to_hdfe_stream_where_this_engine_cannot_fit(data, tmp_path):
    kw = dict(workdir=tmp_path, verbose=False)
    assert isinstance(logit_stream("y ~ x1 | g + h", data, **kw), HDFEResult)
    assert isinstance(logit_stream("y ~ x1 | g", data, bias_correction=True, **kw), HDFEResult)
    assert isinstance(logit_stream("y ~ x1 | g", data, separation_check=False, **kw), HDFEResult)
    # options with no counterpart here are accepted and ignored
    r = logit_stream("y ~ x1 | g", data, iwls_tol=1e-9, iwls_maxiter=30, save_resid=False,
                     within_pseudo_r2=True, fe_dof="pyfixest", **kw)
    assert isinstance(r, BinaryLogitResult)
    with pytest.raises(Exception, match="CRV3"):
        logit_stream("y ~ x1", data, vcov={"CRV3": "c"}, **kw)


def test_outcome_must_be_zero_or_one(data, tmp_path):
    with pytest.raises(ValueError, match="0s and 1s"):
        logit_stream("fw ~ x1", data, workdir=tmp_path, verbose=False)


def test_several_models_in_one_formula(data, tmp_path):
    r = logit_stream("y ~ x1 + sw(x2, h)", data, workdir=tmp_path, verbose=False)
    assert len(r) == 2 and all(isinstance(m, BinaryLogitResult) for m in r)


def test_clusters_are_counted_among_the_rows_used(data, tmp_path):
    """A cluster all of whose rows a fixed effect's separation dropped is not
    one: G, and so the factor G / (G - 1) and the degrees of freedom, count
    the clusters of the rows used."""
    new, old = both("y ~ x1 + x2 | g", data, tmp_path, vcov={"CRV1": "g"})
    assert new.n_clusters == {k: v[0] for k, v in old.n_clusters.items()}
    assert new.all_vcovs["CRV1:g"][1] == old.all_vcovs["CRV1:g"][1]
