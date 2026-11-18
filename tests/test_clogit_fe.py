"""Absorbed alternative-specific constants in the conditional logit
(`clogit_stream("chosen ~ x | estab")`, alt_fe.py): one and two dimensions,
both solvers ("blocks", "contraction") against the same model with the
constants as dummy variables; the automatic choice; the reported
constants; predict; separation; refusals."""

import warnings

import numpy as np
import polars as pl
import pytest

from logit_stream import alt_fe, clogit_stream


def _data(seed=1, F=50, female=False):
    """Workers choosing among their firm's 2 to 6 establishments, by
    distance and each establishment's constant; with `female`, also a
    county x sex constant (the second dimension, which varies within an
    establishment across workers and links firms in a county)."""
    rng = np.random.default_rng(seed)
    rows, estab = [], 0
    cs = rng.normal(0, 0.7, (4, 2))                    # county x sex
    for f in range(F):
        ne = int(rng.integers(2, 7))
        ids = np.arange(estab, estab + ne)
        estab += ne
        delta = rng.normal(0, 1, ne)
        county = rng.integers(0, 4, ne)
        ex, ey = rng.normal(size=(2, ne))
        for i in range(int(rng.integers(5, 41))):
            hx, hy = rng.normal(size=2)
            sex = int(rng.integers(0, 2))
            dist = np.hypot(ex - hx, ey - hy)
            u = -1.2 * dist + delta + (cs[county, sex] if female else 0) + rng.gumbel(size=ne)
            ch = (u == u.max()).astype(int)
            for k in range(ne):
                rows.append((f"{f}_{i}", int(ids[k]), f, int(county[k]), sex,
                             f"{county[k]}_{sex}", float(dist[k]), int(ch[k])))
    return pl.DataFrame(rows, schema=["case", "estab", "firm", "county", "sex", "cs", "dist",
                                      "chosen"], orient="row")


def _dummies(df, dims, vcov="iid"):
    """The same model with the constants as dummies (a reference level
    dropped per firm, and the redundant ones by the fit's collinearity
    check)."""
    first = df.group_by("firm").agg(pl.col("estab").min().alias("e0"))
    est = (df.join(first, on="firm").filter(pl.col("estab") != pl.col("e0"))["estab"]
           .unique().sort().to_list())
    cols = [(pl.col("estab") == e).cast(pl.Float64).alias(f"d{e}") for e in est]
    names = [f"d{e}" for e in est]
    if "cs" in dims:
        levels = df["cs"].unique().sort().to_list()[1:]
        cols += [(pl.col("cs") == c).cast(pl.Float64).alias(f"c{c}") for c in levels]
        names += [f"c{c}" for c in levels]
    dd = df.with_columns(cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # (separation of the dummies)
        return clogit_stream("chosen ~ dist + " + " + ".join(names), dd, case="case",
                             vcov=vcov, verbose=False)


@pytest.mark.parametrize("dims", [["estab"], ["estab", "cs"]])
def test_both_solvers_are_the_dummy_fit(dims):
    df = _data(seed=1 if len(dims) == 1 else 2, female=len(dims) > 1)
    fml = "chosen ~ dist | " + " + ".join(dims)
    ref = _dummies(df, dims)
    for solver in ("blocks", "contraction"):
        f = clogit_stream(fml, df, case="case", fe_solver=solver, verbose=False)
        assert f.converged
        np.testing.assert_allclose(f.beta, ref.beta[:1], rtol=1e-6)
        np.testing.assert_allclose(np.sqrt(np.diag(f.vcov)), np.sqrt(np.diag(ref.vcov))[:1],
                                   rtol=1e-5)
        # (the dummies' never-chosen establishments drift toward -inf, to ~1e-8)
        assert abs(f.loglik - ref.loglik) < 1e-6
    if len(dims) == 1:
        # the constants: within a firm, their differences are the dummies'
        fx = f.fixef().collect().drop_nulls("fe_estab")
        b = dict(zip(ref.coefnames, ref.beta))
        dum = {e: b.get(f"d{e}", 0.0) for e in fx["estab"].to_list()}
        j = fx.join(df.select("estab", "firm").unique(), on="estab")
        j = j.with_columns(pl.Series("ref", [dum[e] for e in j["estab"]]))
        sep = set(df.group_by("estab").agg(pl.col("chosen").sum()).filter(
            pl.col("chosen") == 0)["estab"].to_list())
        j = j.filter(~pl.col("estab").is_in(sep))
        d1 = j.with_columns((pl.col("fe_estab") - pl.col("fe_estab").first().over("firm"))
                            .alias("a"), (pl.col("ref") - pl.col("ref").first().over("firm"))
                            .alias("b"))
        np.testing.assert_allclose(d1["a"].to_numpy(), d1["b"].to_numpy(), atol=1e-5)
        assert (fx["fe_estab"].abs().min() == 0)              # a reference per firm


def test_auto_and_the_reported_groups(monkeypatch):
    df = _data(seed=3)
    f = clogit_stream("chosen ~ dist | estab", df, case="case", verbose=False)
    info = f.diagnostics["alternative_constants"]
    # the groups are the firms (less those all of whose workers chose one establishment)
    one = (df.filter(pl.col("chosen") == 1).group_by("firm").agg(pl.col("estab").n_unique())
           .filter(pl.col("estab") == 1).height)
    assert info["fe_solver"] == "blocks" and info["groups"] == df["firm"].n_unique() - one
    monkeypatch.setattr(alt_fe, "BLOCK_MAX", 1)
    g = clogit_stream("chosen ~ dist | estab", df, case="case", verbose=False)
    assert g.diagnostics["alternative_constants"]["fe_solver"] == "contraction"
    np.testing.assert_allclose(f.beta, g.beta, rtol=1e-7)
    assert f.n_levels == {"estab": df["estab"].n_unique()}


def test_predict_and_separation():
    df = _data(seed=4)
    never = df.group_by("estab").agg(pl.col("chosen").sum()).filter(pl.col("chosen") == 0)
    f = clogit_stream("chosen ~ dist | estab", df, case="case", verbose=False)
    # dropped: the never chosen, and an establishment chosen by all its firm's workers
    # (its constant at +infinity; its firm's sets go)
    every = (df.filter(pl.col("chosen") == 1).group_by("firm").agg(pl.col("estab").n_unique())
             .filter(pl.col("estab") == 1).height)
    assert f.diagnostics["separation"]["levels"] == never.height + every
    p = f.predict(df).collect()
    # never-chosen establishments: null (probability 0); the rest sum to one per set
    assert p.filter(pl.col("estab").is_in(never["estab"].to_list()))["prob"].null_count() \
        == p.filter(pl.col("estab").is_in(never["estab"].to_list())).height
    # (a dropped firm's sets have only nulls)
    s = p.group_by("case").agg(pl.col("prob").sum(), pl.col("prob").count().alias("n")) \
        .filter(pl.col("n") > 0)
    np.testing.assert_allclose(s["prob"].to_numpy(), 1.0, rtol=1e-10)
    # the probabilities by hand from the coefficients and the constants
    fx = f.fixef().collect().select("estab", "fe_estab")
    q = (df.join(fx, on="estab").with_columns(
        (f.beta[0] * pl.col("dist") + pl.col("fe_estab")).alias("v"))
        .drop_nulls("v").with_columns((pl.col("v").exp() / pl.col("v").exp().sum()
                                       .over("case")).alias("q")))
    m = p.join(q.select("case", "estab", "q"), on=["case", "estab"])
    np.testing.assert_allclose(m["prob"].to_numpy(), m["q"].to_numpy(), rtol=1e-10)


def test_refusals():
    df = _data(seed=5, F=10)
    with pytest.raises(ValueError, match="fe_solver"):
        clogit_stream("chosen ~ dist | estab", df, case="case", fe_solver="dense",
                      verbose=False)


def test_standard_errors_and_draws_of_the_constants():
    """One dimension: each constant's standard error (against its group's
    reference) is the dummy fit's for the same difference, and the draws
    follow it."""
    df = _data(seed=6, F=30)
    f = clogit_stream("chosen ~ dist | estab", df, case="case", verbose=False)
    ref = _dummies(df, ["estab"])
    fx = f.fixef().collect()
    assert "se_estab" in fx.columns
    # the dummy fit's variance of (d_j - d_r), r the reference (fe 0)
    names = list(ref.coefnames)
    V = np.asarray(ref.vcov)
    firm = dict(df.select("estab", "firm").unique().iter_rows())
    refs = {firm[e]: e for e, v in zip(fx["estab"], fx["fe_estab"]) if v == 0}
    sep = set(df.group_by("estab").agg(pl.col("chosen").sum())
              .filter(pl.col("chosen") == 0)["estab"].to_list())
    got, want = [], []
    for e, se in zip(fx["estab"], fx["se_estab"]):
        r = refs.get(firm[e])
        if se is None or e in sep or r is None or r in sep or e == r:
            continue
        v = np.zeros(len(names))
        if f"d{e}" in names:
            v[names.index(f"d{e}")] += 1
        if f"d{r}" in names:
            v[names.index(f"d{r}")] -= 1
        got.append(se)
        want.append(np.sqrt(v @ V @ v))
    assert len(got) > 50
    np.testing.assert_allclose(got, want, rtol=1e-4)
    # the draws: the constants' spread is their standard error, the reference stays 0
    ds = f.draw_parameters(2000, seed=1)
    A = np.array([d.fixef().collect()["fe_estab"].to_numpy(allow_copy=True).astype(float)
                  for d in ds])
    se = fx["se_estab"].to_numpy().astype(float)
    ok = np.isfinite(se)
    np.testing.assert_allclose(A[:, ok].std(0), se[ok], rtol=0.08)
    zero = fx["fe_estab"].to_numpy() == 0
    assert np.all(A[:, zero] == 0)
    B = np.array([d.beta for d in ds])
    np.testing.assert_allclose(B.std(), np.sqrt(f.vcov[0, 0]), rtol=0.08)


def test_draws_need_one_dimension_and_the_blocks():
    df = _data(seed=7, F=10)
    g = clogit_stream("chosen ~ dist | estab", df, case="case", fe_solver="contraction",
                      verbose=False)
    with pytest.raises(ValueError, match="blocks"):
        g.draw_parameters(2)


def test_calibrated_predictions_match_the_totals():
    """New firms (none in the fit): each establishment's constant from its
    total, so that its predicted choices sum to it; the totals scaled to
    each firm's number of sets, with a warning when they differ; a total
    of 0 is probability 0; a missing total is an error; draws too."""
    fit = clogit_stream("chosen ~ dist | estab", _data(seed=8), case="case", verbose=False)
    new = _data(seed=9, F=20).with_columns(pl.col("estab") + 10_000)
    totals = new.group_by("estab").agg(pl.col("chosen").sum().cast(pl.Float64).alias("n"))
    p = fit.predict(new.drop("chosen"), calibrate={"estab": totals}).collect()
    got = p.group_by("estab").agg(pl.col("prob").sum()).join(totals, on="estab")
    np.testing.assert_allclose(got["prob"].to_numpy(), got["n"].to_numpy(), atol=1e-8)
    s = p.group_by("case").agg(pl.col("prob").sum())
    np.testing.assert_allclose(s["prob"].to_numpy(), 1.0, rtol=1e-10)
    assert p.filter(pl.col("estab").is_in(totals.filter(pl.col("n") == 0)["estab"].to_list())
                    )["prob"].max() in (0.0, None)
    assert p.height == new.height and p["case"].to_list() == new["case"].to_list()
    # totals off by a factor of two: scaled back, with a warning
    with pytest.warns(UserWarning, match="scaled"):
        q = fit.predict(new.drop("chosen"),
                        calibrate={"estab": totals.with_columns(pl.col("n") * 2)}).collect()
    np.testing.assert_allclose(q["prob"].to_numpy(), p["prob"].to_numpy(), atol=1e-9)
    with pytest.raises(ValueError, match="without a total"):
        fit.predict(new.drop("chosen"), calibrate={"estab": totals.head(3)}).collect()
    # a draw's predictions use its own coefficient
    d = fit.draw_parameters(1, seed=2)[0]
    pd_ = d.predict(new.drop("chosen"), calibrate={"estab": totals}).collect()
    gd = pd_.group_by("estab").agg(pl.col("prob").sum()).join(totals, on="estab")
    np.testing.assert_allclose(gd["prob"].to_numpy(), gd["n"].to_numpy(), atol=1e-8)
    assert not np.allclose(pd_["prob"].to_numpy(), p["prob"].to_numpy())


def _without_separation(df):
    """The data less what the fit would drop (never-chosen establishments,
    sets left with fewer than two, establishments chosen wherever they
    appear), so that the dummy fit's sets and clusters are the same."""
    for _ in range(10):
        ch = df.group_by("estab").agg(pl.col("chosen").sum().alias("c"), pl.len().alias("n"))
        never = ch.filter(pl.col("c") == 0)["estab"]
        always = ch.filter(pl.col("c") == pl.col("n"))["estab"]
        bad = df.filter(pl.col("estab").is_in(always.to_list()))["case"].unique()
        new = (df.filter(~pl.col("estab").is_in(never.to_list()))
               .filter(~pl.col("case").is_in(bad.to_list()))
               .filter(pl.len().over("case") >= 2))
        if new.height == df.height:
            return df
        df = new
    return df


@pytest.mark.parametrize("vcov", ["hetero", {"CRV1": "firm"}])
def test_robust_variances_are_the_dummy_fits(vcov):
    df = _without_separation(_data(seed=10, F=40))
    rv = _dummies(df, ["estab"], vcov=vcov)
    for solver in ("blocks", "contraction"):
        f = clogit_stream("chosen ~ dist | estab", df, case="case", vcov=vcov,
                          fe_solver=solver, verbose=False)
        k = list(rv.coefnames).index("dist")
        np.testing.assert_allclose(np.sqrt(f.vcov[0, 0]), np.sqrt(rv.vcov[k, k]), rtol=1e-5)
        assert not np.isclose(f.vcov[0, 0], f.all_vcovs["iid"][0][0, 0], rtol=1e-3)
