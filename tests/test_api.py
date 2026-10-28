"""Binary logit, options, errors, reporting and housekeeping."""

import logging
import os

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

import logit_stream
from logit_stream import clogit_stream, etable, mlogit_stream, ologit_stream
from logit_stream import logit_stream as logit
from logit_stream.newton import _Driver


def test_public_api():
    for name in logit_stream.__all__:
        assert hasattr(logit_stream, name)
    assert logit_stream.__version__


def test_binary_logit_is_hdfe_stream(mnl_data):
    from hdfe_stream import HDFEResult
    df = mnl_data.with_columns((pl.col("yi") == 1).cast(pl.Float64).alias("yb"))
    r = logit("yb ~ x1 + x2", df, verbose=False, iwls_tol=1e-12)
    assert isinstance(r, HDFEResult) and r.family == "logit"
    m = sm.Logit(df["yb"].to_numpy(), sm.add_constant(df.select("x1", "x2").to_numpy())) \
        .fit(disp=0, tol=1e-12)
    np.testing.assert_allclose(r.beta, m.params, rtol=1e-8)
    f = logit("yb ~ x1 + x2 | g", df, verbose=False)
    assert f.fixef("g").collect().height == 4
    # the same model from the multinomial estimator (its own conventions)
    two = mlogit_stream("yb ~ x1 + x2", df, verbose=False, tol=1e-14)
    np.testing.assert_allclose(two.beta, r.beta, rtol=1e-7)


def test_vcov_errors(mnl_data):
    with pytest.raises(ValueError, match="CRV3 is available"):
        mlogit_stream("y ~ x1 | g", mnl_data, verbose=False, vcov={"CRV3": "cl"})
    with pytest.raises(ValueError, match="one-way"):
        mlogit_stream("y ~ x1", mnl_data, verbose=False, vcov={"CRV3": "cl+g"})
    with pytest.raises(ValueError, match="weights_type"):
        mlogit_stream("y ~ x1", mnl_data, verbose=False, weights="w", weights_type="pweights")
    with pytest.raises(ValueError, match="positive"):
        mlogit_stream("y ~ x1", mnl_data.with_columns(pl.col("w") - 1), weights="w",
                      verbose=False)
    with pytest.raises(ValueError, match="storage"):
        mlogit_stream("y ~ x1", mnl_data, verbose=False, storage="gpu")
    with pytest.raises(ValueError, match="IV"):
        mlogit_stream("y ~ 1 | g | x1 ~ x2", mnl_data, verbose=False)
    with pytest.raises(ValueError, match="only one value"):
        mlogit_stream("y ~ x1", mnl_data.filter(pl.col("y") == "a"), verbose=False)
    with pytest.raises(ValueError, match="case= is for"):
        mlogit_stream("y ~ x1", mnl_data, verbose=False, case="g")


def test_clustering_by_interaction(mnl_data):
    r = mlogit_stream("y ~ x1", mnl_data, verbose=False, vcov={"CRV1": "g^cl"})
    assert r.n_clusters == {"g^cl": mnl_data.select("g", "cl").unique().height}


def test_fe_not_for_clogit(choice_data):
    with pytest.raises(ValueError, match="not available for the conditional"):
        clogit_stream("chosen ~ cost | alt", choice_data, case="case", verbose=False)


def test_gpu_engine_needs_cudf_polars(mnl_data):
    import importlib.util
    if importlib.util.find_spec("cudf_polars") is not None:
        pytest.skip("cudf-polars is installed")
    with pytest.raises(ImportError, match="cudf-polars"):
        mlogit_stream("y ~ x1", mnl_data, verbose=False, polars_engine="gpu")


def test_run_directory_is_removed(tmp_path, mnl_data):
    r = mlogit_stream("y ~ x1", mnl_data, verbose=False, workdir=tmp_path)
    assert r.diagnostics["store"]["bytes"] > 0 and r.diagnostics["disk_peak_gb"] > 0
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("hdfe_run_")]
    # keep_intermediates leaves it
    mlogit_stream("y ~ x1", mnl_data, verbose=False, workdir=tmp_path, keep_intermediates=True)
    assert [p for p in tmp_path.iterdir() if p.name.startswith("hdfe_run_")]


def test_workdir_environment(tmp_path, mnl_data, monkeypatch):
    monkeypatch.setenv("LOGIT_STREAM_WORKDIR", str(tmp_path))
    mlogit_stream("y ~ x1", mnl_data, verbose=False, keep_intermediates=True)
    assert any(p.name.startswith("hdfe_run_") for p in tmp_path.iterdir())


def test_logger(mnl_data, caplog):
    logger = logging.getLogger("lstest")
    with caplog.at_level(logging.INFO, logger="lstest"):
        r = mlogit_stream("y ~ x1", mnl_data, logger=logger)
        r.summary(logger=logger)
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("Newton 1" in m for m in msgs) and any("Multinomial logit" in m for m in msgs)


def test_verbose_prints(mnl_data, capsys):
    mlogit_stream("y ~ x1", mnl_data, verbose=True)
    assert "[logit " in capsys.readouterr().out


def test_maxiter_warns(mnl_data):
    with pytest.warns(UserWarning, match="did not converge"):
        r = mlogit_stream("y ~ x1 + x2", mnl_data, verbose=False, maxiter=1)
    assert not r.converged and not r.diagnostics["newton"]["converged"]


def test_separation_warns():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = np.where(x > 0.5, 2, rng.integers(0, 2, 400))
    df = pl.DataFrame({"y": y, "x": x, "z": rng.normal(size=400)})
    with pytest.warns(UserWarning):
        mlogit_stream("y ~ x + z", df.with_columns((pl.col("x") > 0.5).cast(pl.Float64)
                                                   .alias("x")), verbose=False)
    # a ridge penalty keeps the estimates finite
    r = mlogit_stream("y ~ x + z", df.with_columns((pl.col("x") > 0.5).cast(pl.Float64)
                                                   .alias("x")), verbose=False, l2=1.0)
    assert r.converged and np.all(np.abs(r.beta) < 25)


def test_etable(mnl_data, ord_data):
    m = mlogit_stream("y ~ x1 + x2", mnl_data, verbose=False)
    o = ologit_stream("y ~ x1 + x2", ord_data, verbose=False)
    t = etable([m, o], type="df")
    assert t.shape[1] == 2


def test_collinear_helper():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    X = np.column_stack([X, X[:, 0] - X[:, 2], np.zeros(200)])
    keep = _Driver.collinear(X.T @ X)
    assert keep.tolist() == [True, True, True, True, False, False]
    keep = _Driver.collinear(X.T @ X, n_protected=1)
    assert keep.tolist() == [True, True, True, False, False]


def test_with_summary_and_pyfixest(mnl_data):
    r = mlogit_stream("y ~ x1", mnl_data, verbose=False)
    pf = r.to_pyfixest()
    assert pf._method == "feglm" and pf._loglik == r.loglik
    assert os.path.basename(__file__)
