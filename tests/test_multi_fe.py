"""The multinomial logit with several absorbed fixed effects, fitted as a
Poisson regression on the data expanded by outcome level (multi_fe.py)."""

import numpy as np
import polars as pl
import pytest

from logit_stream import MultinomialLogitResult, mlogit_stream


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 3000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    g, h = rng.integers(0, 30, n), rng.integers(0, 12, n)
    a, b = rng.normal(0, .5, 30), rng.normal(0, .5, 12)
    E = np.column_stack([np.zeros(n), 0.3 + 0.8 * x1 + a[g] + b[h], -0.4 + 0.5 * x2 - a[g]])
    P = np.exp(E)
    P /= P.sum(1, keepdims=True)
    y = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g, "h": h,
                         "w": rng.uniform(0.5, 2, n), "f": rng.integers(1, 4, n),
                         "c": rng.integers(0, 40, n)})


KW = dict(verbose=False, tol=1e-12)


@pytest.mark.parametrize("extra", [{}, {"weights": "w"},
                                   {"weights": "f", "weights_type": "fweights"}])
def test_equals_one_fixed_effect_and_indicators(data, extra):
    """The second fixed effect as indicators, the first absorbed (the
    native fit), is the same model: the same estimates, log-likelihood,
    every variance and the number of parameters."""
    # (the robust variance is not available with frequency weights: test_restrictions)
    v = "iid" if extra.get("weights_type") == "fweights" else "hetero"
    a = mlogit_stream("y ~ x1 + x2 + C(h) | g", data, vcov=v, cluster="c", **KW, **extra)
    b = mlogit_stream("y ~ x1 + x2 | g + h", data, vcov=v, cluster="c", **KW, **extra)
    assert isinstance(b, MultinomialLogitResult) and b.diagnostics["route"] == "poisson"
    assert b.coefnames == ["1:x1", "1:x2", "2:x1", "2:x2"]
    ia = [a.coefnames.index(c) for c in b.coefnames]
    np.testing.assert_allclose(b.beta, a.beta[ia], atol=1e-7)
    assert b.loglik == pytest.approx(a.loglik, abs=1e-7)
    assert b.loglik_null == pytest.approx(a.loglik_null, abs=1e-8)
    assert b.n_obs == a.n_obs
    for k in ["iid", "CRV1:c"] + (["hetero"] if v == "hetero" else []):
        np.testing.assert_allclose(b.all_vcovs[k][0], a.all_vcovs[k][0][np.ix_(ia, ia)],
                                   rtol=1e-5, err_msg=k)
    assert b.all_vcovs["CRV1:c"][1] == a.all_vcovs["CRV1:c"][1]
    assert b.n_params == a.n_params and b.aic == pytest.approx(a.aic, abs=1e-6)


def test_clustered_by_a_fixed_effect(data):
    a = mlogit_stream("y ~ x1 + x2 + C(h) | g", data, vcov={"CRV1": "g"}, **KW)
    b = mlogit_stream("y ~ x1 + x2 | g + h", data, vcov={"CRV1": "g"}, **KW)
    ia = [a.coefnames.index(c) for c in b.coefnames]
    np.testing.assert_allclose(b.vcov, a.vcov[np.ix_(ia, ia)], rtol=1e-5)


def test_separated_cells_warning(data):
    """A group without some outcome level: the warning is about (group,
    level) cells, not the Poisson regression's expanded rows."""
    d = data.filter(~((pl.col("g") == 0) & (pl.col("y") == 2)))
    with pytest.warns(UserWarning, match=r"1 \(g, outcome level\) cells") as rec:
        b = mlogit_stream("y ~ x1 + x2 | g + h", d, **KW)
    assert not any("__ls_" in str(w.message) for w in rec)
    assert b.fixef("g").filter(pl.col("g") == 0).collect()["fe_g:2"].is_null().all()


def test_predictions(data):
    a = mlogit_stream("y ~ x1 + x2 + C(h) | g", data, **KW)
    b = mlogit_stream("y ~ x1 + x2 | g + h", data, **KW)
    pa = a.predict(data.head(50)).collect().select(pl.col("^prob_.*$")).to_numpy()
    pb = b.predict(data.head(50)).collect().select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(pb, pa, atol=1e-7)
    assert set(b.fixef("g").collect_schema().names()) >= {"g", "fe_g:0", "fe_g:1", "fe_g:2"}
    # unseen groups get nulls
    new = data.head(3).with_columns(pl.lit(99).alias("h"))
    assert b.predict(new).collect()["prob_1"].null_count() == 3


def test_poisson_options_pass_through(data):
    """fepois_stream's own options reach it; the route's own are refused."""
    a = mlogit_stream("y ~ x1 + x2 | g + h", data, **KW)
    b = mlogit_stream("y ~ x1 + x2 | g + h", data, poisson_options={"solver": "stream_cg"},
                      **KW)
    np.testing.assert_allclose(b.beta, a.beta, atol=1e-6)
    with pytest.raises(ValueError, match="cannot set"):
        mlogit_stream("y ~ x1 | g + h", data, poisson_options={"vcov": "iid"}, verbose=False)
    with pytest.raises(TypeError):
        mlogit_stream("y ~ x1 | g + h", data, poisson_options={"bogus": 1}, verbose=False)
    with pytest.raises(ValueError, match="several fixed effects"):
        from logit_stream import ologit_stream
        ologit_stream("y ~ x1 | g", data, poisson_options={}, verbose=False)


def test_device_is_fepois_streams(data, monkeypatch):
    """device="cuda" moves fepois_stream's fixed-effect solve (numpy stand-in
    for CuPy, as hdfe-stream's tests do)."""
    from types import SimpleNamespace

    import hdfe_stream.core.gpu as gpu
    import scipy.sparse as sp
    monkeypatch.setattr(gpu, "cuda_backend",
                        lambda: SimpleNamespace(xp=np, asnumpy=np.asarray, name="numpy",
                                                csr=lambda S: sp.csr_matrix(S, copy=True)))
    a = mlogit_stream("y ~ x1 + x2 | g + h", data, **KW)
    b = mlogit_stream("y ~ x1 + x2 | g + h", data, device="cuda", **KW)
    np.testing.assert_allclose(b.beta, a.beta, atol=1e-6)


def test_restrictions(data):
    with pytest.raises(ValueError, match="not available"):
        mlogit_stream("y ~ x1 | g + h", data, l2=1.0, verbose=False)
    with pytest.raises(ValueError, match="frequency weights"):
        mlogit_stream("y ~ x1 | g + h", data, vcov="hetero", weights="f",
                      weights_type="fweights", verbose=False)
    with pytest.raises(ValueError, match="one absorbed"):
        from logit_stream import ologit_stream
        ologit_stream("y ~ x1 | g + h", data, verbose=False)
