"""An absorbed fixed effect gives exactly the fit with its indicators."""

import numpy as np
import polars as pl
import pytest

from logit_stream import mlogit_stream, ologit_stream


def _match(fe, dummies):
    idx = [dummies.coefnames.index(c) for c in fe.coefnames if c in dummies.coefnames]
    return idx


@pytest.mark.parametrize("vcov", ["iid", "hetero", "CRV1:cl"])
def test_multinomial_fe_equals_dummies(fe_data, vcov):
    a = mlogit_stream("y ~ x1 + x2 | g", fe_data, verbose=False, tol=1e-14, vcov=vcov)
    b = mlogit_stream("y ~ x1 + x2 + C(g)", fe_data, verbose=False, tol=1e-14, vcov=vcov)
    idx = _match(a, b)
    assert len(idx) == len(a.coefnames) == 6
    np.testing.assert_allclose(a.beta, b.beta[idx], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(a.se, b.se[idx], rtol=1e-7)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)
    assert a.n_params == b.n_params
    pa = a.predict(fe_data).collect().select(pl.col("^prob_.*$")).to_numpy()
    pb = b.predict(fe_data).collect().select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(pa, pb, atol=1e-7)


@pytest.mark.parametrize("vcov", ["iid", "hetero", "CRV1:cl"])
def test_ordered_fe_equals_dummies(fe_data, vcov):
    a = ologit_stream("yo ~ x1 + x2 | g", fe_data, verbose=False, tol=1e-14, vcov=vcov)
    b = ologit_stream("yo ~ x1 + x2 + C(g)", fe_data, verbose=False, tol=1e-14, vcov=vcov)
    assert a.coefnames == ["x1", "x2", "cut2", "cut3"]
    np.testing.assert_allclose(a.beta[:2], b.beta[:2], rtol=1e-8)
    np.testing.assert_allclose(a.se[:2], b.se[:2], rtol=1e-7)
    cb = b.beta[-3:]
    np.testing.assert_allclose(a.cutpoints, cb - cb[0], rtol=1e-8, atol=1e-10)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)
    pa = a.predict(fe_data).collect().select(pl.col("^prob_.*$")).to_numpy()
    pb = b.predict(fe_data).collect().select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(pa, pb, atol=1e-9)


def test_fixef_table(fe_data):
    a = mlogit_stream("y ~ x1 + x2 | g", fe_data, verbose=False, tol=1e-14)
    t = a.fixef("g").collect()
    assert t.columns == ["g", "fe_g:0", "fe_g:1", "fe_g:2", "fe_g:3", "n_obs"]
    assert t.height == 40 and t["n_obs"].sum() == fe_data.height
    assert (t["fe_g:0"] == 0).all()
    assert a.fixef().collect().equals(t)
    with pytest.raises(KeyError):
        a.fixef("h")
    o = ologit_stream("yo ~ x1 + x2 | g", fe_data, verbose=False)
    assert o.fixef("g").collect().columns == ["g", "fe_g", "n_obs"]
    assert a.fe_names == ["g"] and a.n_levels == {"g": 40}


def test_levels_absent_from_a_group(fe_data):
    """A group that never has some level gives it probability zero, and the
    fit is the one without those (group, level) cells."""
    df = fe_data.filter(~((pl.col("g") < 5) & (pl.col("y") == 2)))
    a = mlogit_stream("y ~ x1 + x2 | g", df, verbose=False, tol=1e-14)
    assert a.converged
    t = a.fixef("g").collect().sort("g")
    assert t.filter(pl.col("g") < 5)["fe_g:2"].null_count() == 5
    p = a.predict(df.filter(pl.col("g") < 5)).collect()
    assert (p["prob_2"] == 0).all()
    np.testing.assert_allclose(p.select(pl.col("^prob_.*$")).to_numpy().sum(axis=1), 1)
    # the base level absent from a group: that group's own reference takes over
    df0 = fe_data.filter(~((pl.col("g") == 7) & (pl.col("y") == 0)))
    b = mlogit_stream("y ~ x1 + x2 | g", df0, verbose=False, tol=1e-14)
    t0 = b.fixef("g").collect().filter(pl.col("g") == 7)
    assert t0["fe_g:0"].is_null().all() and t0["fe_g:1"].item() == 0


def test_separated_groups_are_dropped(fe_data):
    df = fe_data.with_columns(pl.when(pl.col("g") == 3).then(1).otherwise(pl.col("y"))
                              .alias("y"),
                              pl.when(pl.col("g") == 4).then(0).otherwise(pl.col("yo"))
                              .alias("yo"))
    n3 = df.filter(pl.col("g") == 3).height
    with pytest.warns(UserWarning, match="never varies"):
        a = mlogit_stream("y ~ x1 + x2 | g", df, verbose=False)
    assert a.n_obs == df.height - n3
    assert a.diagnostics["separation"]["levels"] == 1
    s = a.sample().collect()
    assert s.filter(pl.col("g") == 3)["dropped_because"].unique().to_list() == ["separation"]
    assert a.resid().collect().height == a.n_obs
    assert 3 not in a.fixef().collect()["g"].to_list()
    assert a.predict(df.filter(pl.col("g") == 3).head(3)).collect()["prob_0"].null_count() == 3
    with pytest.warns(UserWarning, match="never varies"):
        o = ologit_stream("yo ~ x1 + x2 | g", df, verbose=False)
    assert o.n_obs == df.height - df.filter(pl.col("g") == 4).height


def test_group_level_covariate_is_collinear(fe_data):
    df = fe_data.with_columns((pl.col("g") * 0.1).alias("gx"))
    with pytest.warns(UserWarning, match="collinear"):
        a = mlogit_stream("y ~ x1 + gx | g", df, verbose=False)
    assert a.collin_vars == ["gx"]


def test_one_fixed_effect_only(fe_data):
    """The ordered logit absorbs one; the multinomial any number (through
    the Poisson regression: test_multi_fe.py)."""
    with pytest.raises(ValueError, match="one absorbed fixed effect"):
        ologit_stream("y ~ x1 | g + cl", fe_data, verbose=False)
    f = mlogit_stream("y ~ x1 | g + cl", fe_data, verbose=False)
    assert f.diagnostics["route"] == "poisson"


def test_interacted_fixed_effect(fe_data):
    df = fe_data.with_columns((pl.col("g") % 4).alias("g1"), (pl.col("g") // 4).alias("g2"))
    a = mlogit_stream("y ~ x1 + x2 | g1^g2", df, verbose=False, tol=1e-14)
    b = mlogit_stream("y ~ x1 + x2 | g", df, verbose=False, tol=1e-14)
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    assert a.fixef().collect().columns[:2] == ["g1", "g2"]
