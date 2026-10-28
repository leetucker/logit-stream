"""clogit_stream: against the multinomial logit it nests, statsmodels, and
its handling of choice sets."""

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from logit_stream import ConditionalLogitResult, clogit_stream, mlogit_stream

FML = "chosen ~ cost + time + C(alt) + income:C(alt)"


@pytest.fixture(scope="module")
def fit(choice_data):
    return clogit_stream(FML, choice_data, case="case", verbose=False, tol=1e-14)


def test_multinomial_logit_is_a_conditional_logit():
    """A multinomial logit is the conditional logit with alternative-specific
    constants and characteristics interacted with the alternatives, on the
    long version of the data: the estimates, standard errors and
    log-likelihood are the same."""
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.normal(size=n)
    E = np.column_stack([np.zeros(n), 0.3 + 0.8 * x, -0.2 - 0.5 * x])
    P = np.exp(E) / np.exp(E).sum(axis=1, keepdims=True)
    y = (P.cumsum(axis=1) > rng.random(n)[:, None]).argmax(axis=1)
    wide = pl.DataFrame({"id": np.arange(n), "y": y, "x": x})
    long = (wide.join(pl.DataFrame({"alt": [0, 1, 2]}), how="cross")
                .with_columns((pl.col("alt") == pl.col("y")).cast(pl.Int8).alias("chosen"),
                              *[(pl.col("alt") == j).cast(pl.Float64).alias(f"a{j}")
                                for j in (1, 2)])
                .with_columns(*[(pl.col(f"a{j}") * pl.col("x")).alias(f"xa{j}") for j in (1, 2)])
                .sort("id", "alt"))
    c = clogit_stream("chosen ~ a1 + xa1 + a2 + xa2", long, case="id", verbose=False,
                      tol=1e-14)
    m = mlogit_stream("y ~ x", wide, verbose=False, tol=1e-14)
    np.testing.assert_allclose(c.beta, m.beta, rtol=1e-9)
    np.testing.assert_allclose(c.se, m.se, rtol=1e-8)
    assert c.loglik == pytest.approx(m.loglik, rel=1e-12)
    assert c.n_cases == c.n_obs == n and c.diagnostics["n_rows"] == 3 * n
    s = sm.MNLogit(y, sm.add_constant(x)).fit(disp=0, tol=1e-12)
    np.testing.assert_allclose(c.beta, s.params.T.ravel(), rtol=1e-7)


def test_matches_statsmodels(choice_data, fit):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    d = choice_data
    alt = d["alt"].to_numpy()
    X = np.column_stack([d["cost"], d["time"], alt == "metro", alt == "train",
                         d["income"] * (alt == "car"),
                         d["income"] * (alt == "metro")]).astype(float)
    m = ConditionalLogit(d["chosen"].to_numpy(), X, groups=d["case"].to_numpy()).fit(
        disp=0, method="newton", tol=1e-12, maxiter=100)
    assert fit.collin_vars == ["income:C(alt)[train]"]
    np.testing.assert_allclose(fit.beta, m.params, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(fit.se, m.bse, rtol=1e-6)
    assert fit.loglik == pytest.approx(m.llf, rel=1e-10)
    assert isinstance(fit, ConditionalLogitResult)


def test_rows_in_any_order(choice_data, fit):
    shuffled = choice_data.sample(fraction=1.0, shuffle=True, seed=1)
    r = clogit_stream(FML, shuffled, case="case", verbose=False, tol=1e-14, block_mb=0.02)
    np.testing.assert_allclose(r.beta, fit.beta, rtol=1e-10)
    np.testing.assert_allclose(r.se, fit.se, rtol=1e-10)
    small = clogit_stream(FML, choice_data, case="case", verbose=False, tol=1e-14,
                          block_mb=0.02, chunk_mb=0.01)
    np.testing.assert_allclose(small.beta, fit.beta, rtol=1e-10)
    assert small.diagnostics["store"]["blocks"] > 5


def test_null_loglik_is_equal_shares(choice_data, fit):
    sizes = choice_data.group_by("case").len()["len"].to_numpy()
    assert fit.loglik_null == pytest.approx(-np.log(sizes).sum(), rel=1e-12)


def test_choice_set_validation(choice_data):
    two = choice_data.with_columns(pl.lit(1, pl.Int8).alias("chosen"))
    with pytest.raises(ValueError, match="exactly one chosen"):
        clogit_stream("chosen ~ cost", two, case="case", verbose=False)
    bad = choice_data.with_columns((pl.col("chosen") * 2).alias("chosen"))
    with pytest.raises(ValueError, match="0 or 1"):
        clogit_stream("chosen ~ cost", bad, case="case", verbose=False)
    with pytest.raises(ValueError, match="case="):
        clogit_stream("chosen ~ cost", choice_data, case=None, verbose=False)
    w = choice_data.with_columns(pl.int_range(pl.len()).cast(pl.Float64).alias("w") + 1)
    with pytest.raises(ValueError, match="constant within"):
        clogit_stream("chosen ~ cost", w, case="case", weights="w", verbose=False)


def test_weights_are_per_choice_set(choice_data):
    d = choice_data.filter(pl.col("case") < 1500)
    w = d.with_columns((pl.col("case") % 3 + 1).cast(pl.Float64).alias("w"))
    a = clogit_stream("chosen ~ cost + time + C(alt)", w, case="case", weights="w",
                      weights_type="fweights", verbose=False, tol=1e-14)
    rep = (w.with_columns(pl.col("w").cast(pl.Int64).alias("k"))
            .select(pl.all().repeat_by("k")).explode(pl.all())
            .with_columns((pl.col("case") * 10 + pl.int_range(pl.len()).over("case", "alt"))
                          .alias("case2")))
    b = clogit_stream("chosen ~ cost + time + C(alt)", rep, case="case2", verbose=False,
                      tol=1e-14)
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-9)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-8)
    assert a.n_obs == b.n_cases


def test_predict_and_resid(choice_data, fit):
    p = fit.predict(choice_data).collect()
    sums = p.group_by("case").agg(pl.col("prob").sum())["prob"].to_numpy()
    np.testing.assert_allclose(sums, 1.0)
    r = fit.resid().collect()
    assert r.columns == ["row_id", "case", "chosen", "xb", "prob", "resid"]
    np.testing.assert_allclose(r["resid"].sum(), 0, atol=1e-8)
    lk = fit.predict(choice_data.head(10), type="link").collect()
    assert "xb" in lk.columns


def test_robust_unit_is_the_choice_set(choice_data):
    r = clogit_stream("chosen ~ cost + time + C(alt)", choice_data, case="case",
                      verbose=False, vcov="hetero", cluster="case")
    # clustering by the choice set itself is the robust variance up to the
    # small-sample factors: N / (N - 1) for both, as G = N
    np.testing.assert_allclose(r.se, r.with_vcov("CRV1:case").se, rtol=1e-12)


def test_poisson_with_a_choice_set_effect_is_the_conditional_logit(choice_data, fit):
    """The "Poisson trick" (Guimaraes, Figueiredo and Woodward 2003): Poisson
    with a fixed effect per choice set has the conditional logit's
    likelihood in the coefficients, so the same estimates and, from the same
    information, the same iid standard errors (hdfe-stream's carry
    pyfixest's (N - 1) / (N - K))."""
    import hdfe_stream as hs
    p = hs.fepois_stream(FML + " | case", choice_data,
                         verbose=False, iwls_tol=1e-14)
    common = [n for n in fit.coefnames if n in p.coefnames]
    assert len(common) == len(fit.coefnames)
    for n in common:
        assert p.coef()[n] == pytest.approx(fit.coef()[n], abs=1e-7)
    ssc = np.sqrt((p.n_obs - 1) / (p.n_obs - len(p.beta) - p.k_fe))
    np.testing.assert_allclose(np.array([p.se[p.coefnames.index(n)] for n in common]) / ssc,
                               np.array([fit.se[fit.coefnames.index(n)] for n in common]),
                               rtol=1e-6)


def test_incomplete_choice_sets_are_dropped_whole(choice_data):
    """A set with a missing alternative is dropped whole (Stata's cmclogit);
    altwise=True keeps its other alternatives, and sample() says why."""
    df = choice_data.with_row_index("r")
    sets = df["case"].unique().sort().head(40).to_list()
    hit = df.filter(pl.col("case").is_in(sets) & (pl.col("chosen") == 0)) \
            .group_by("case").agg(pl.col("r").first())["r"]
    bad = df.with_columns(pl.when(pl.col("r").is_in(hit.to_list())).then(None)
                          .otherwise(pl.col("time")).alias("time")).drop("r")
    with pytest.warns(UserWarning, match="choice sets dropped"):
        f = clogit_stream(FML, bad, case="case", verbose=False)
    n_bad = len(hit)
    assert f.n_cases == choice_data["case"].n_unique() - n_bad
    full = clogit_stream(FML, choice_data.filter(~pl.col("case").is_in(
        bad.filter(pl.col("time").is_null())["case"].to_list())), case="case", verbose=False)
    np.testing.assert_allclose(f.beta, full.beta, atol=1e-10)
    s = f.sample().collect()
    assert (s["dropped_because"] == "missing").sum() == n_bad
    inc = s.filter(pl.col("dropped_because") == "incomplete set")
    assert inc.height > 0 and set(inc["case"].unique()) <= set(sets)
    assert f.resid().collect().height == s["in_sample"].sum()
    g = clogit_stream(FML, bad, case="case", verbose=False, altwise=True)
    assert g.n_cases == choice_data["case"].n_unique()
