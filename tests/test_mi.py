"""Multiple imputation helpers (logit_stream.mi): `pool` and `pool_matrix`
against mice's pool.scalar and mitml's D1 test; and `predict(type="draw")`
for the binary, ordered, multinomial and conditional logit: the draws'
frequencies against the probabilities, a draw that depends only on the
seed and the row's identifier (not the rows' order), and refusals."""

import sys
import warnings

import numpy as np
import polars as pl
import pytest

from logit_stream import (
    clogit_stream,
    logit_stream,
    mlogit_stream,
    ologit_stream,
    pool,
    pool_matrix,
)

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_clogit_fe import _data as _choice_data  # noqa: E402

# five imputations of three estimates, with covariances that grow with m
Q = np.array([[1, 2, 0.3], [1.1, 2.2, 0.25], [0.9, 1.8, 0.41], [1.05, 2.1, 0.33],
              [0.95, 1.9, 0.29]])
S = np.array([[0.010, 0.002, 0.001], [0.002, 0.012, 0.0], [0.001, 0.0, 0.004]])
U = [S * (1 + 0.1 * m) for m in range(1, 6)]
# R 4.5.3, mice 3.19.0: pool.scalar(Q[, j], U[j, j], n = Inf) and, for the
# Barnard-Rubin degrees of freedom, n = 53, k = 3 (df_com = 50)
MICE = dict(t=[0.0205, 0.0456, 0.009496], df=[29.8844444444, 9.2416, 19.5439731733],
            r=[0.5769230769, 1.9230769231, 0.8261538462],
            fmi=[0.4044218236, 0.7137869835, 0.5009815327],
            df_br=[15.0971906974, 5.9185424257, 11.2205711328])


def test_pool_matches_mice():
    long = pl.DataFrame({"m": np.repeat(np.arange(5), 3), "stat": ["a", "b", "c"] * 5,
                         "estimate": Q.ravel(),
                         "se": np.sqrt(np.array([np.diag(u) for u in U]).ravel())})
    out = pool(long, by="stat")
    assert out["stat"].to_list() == ["a", "b", "c"]
    np.testing.assert_allclose(out["estimate"], Q.mean(0), rtol=1e-12)
    np.testing.assert_allclose(out["se"] ** 2, MICE["t"], rtol=1e-9)
    np.testing.assert_allclose(out["df"], MICE["df"], rtol=1e-9)
    np.testing.assert_allclose(out["riv"], MICE["r"], rtol=1e-9)
    np.testing.assert_allclose(out["fmi"], MICE["fmi"], rtol=1e-9)
    np.testing.assert_allclose(pool(long, by="stat", df_com=50)["df"], MICE["df_br"], rtol=1e-9)
    assert (out["m"] == 5).all()
    lo, hi = out["ci_low"].to_numpy(), out["ci_high"].to_numpy()
    from scipy import stats
    np.testing.assert_allclose(hi - lo, 2 * stats.t.ppf(0.975, MICE["df"]) * np.sqrt(MICE["t"]))
    # the same through pool_matrix
    pm = pool_matrix(Q, U, names=["a", "b", "c"])
    np.testing.assert_allclose(pm.tidy()["se"], out["se"], rtol=1e-12)
    np.testing.assert_allclose(np.diag(pm.vcov), MICE["t"], rtol=1e-9)


def test_wald_test_matches_mitml_d1():
    """mitml 0.4.5: testConstraints(qhat, uhat, constraints = c("a",
    "b - 2*a", "c - 0.3"), method = "D1"), and c("a - 1", "b - 2")."""
    pm = pool_matrix(Q, U, names=["a", "b", "c"])
    t = pm.wald_test(np.array([[1, 0, 0], [-2, 1, 0], [0, 0, 1]]), [0, 0, 0.3])
    assert t["statistic"] == pytest.approx(47.4731191895, rel=1e-7)
    assert t["df1"] == 3 and t["df2"] == pytest.approx(29.6810807757, rel=1e-7)
    assert t["p"] == pytest.approx(1.91787035576e-11, rel=1e-6)
    assert t["riv"] == pytest.approx(1.05260721846, rel=1e-7)
    t = pm.wald_test(["a", "b"], [1, 2])
    assert t["df2"] == pytest.approx(15.3630280766, rel=1e-7)
    assert t["riv"] == pytest.approx(1.0941644589, rel=1e-7)
    assert t["statistic"] == pytest.approx(0, abs=1e-12) and t["p"] == pytest.approx(1)


def test_pool_without_sampling_variance_and_refusals():
    """No standard errors: a statistic of a whole population, all of whose
    variance is the imputation's, on M - 1 degrees of freedom."""
    d = pl.DataFrame({"estimate": [1.0, 1.2, 0.8, 1.1]})
    out = pool(d, se=None)
    B = np.var([1.0, 1.2, 0.8, 1.1], ddof=1)
    assert out["se"][0] == pytest.approx(np.sqrt((1 + 1 / 4) * B))
    assert out["df"][0] == pytest.approx(3) and out["fmi"][0] == 1.0
    pm = pool_matrix(Q[:, :2])
    np.testing.assert_allclose(pm.within, 0)
    with pytest.raises(ValueError, match="singular"):
        pm.wald_test()
    with pytest.raises(ValueError, match="same number"):
        pool(pl.DataFrame({"s": ["a", "a", "b"], "estimate": [1.0, 2, 3], "se": 1.0}), by="s")
    with pytest.raises(ValueError, match="at least 2"):
        pool(pl.DataFrame({"estimate": [1.0], "se": [1.0]}))
    with pytest.raises(ValueError, match="not in the data"):
        pool(d, se="se")


def test_pool_matrix_from_results():
    rng = np.random.default_rng(0)
    fits = []
    for _ in range(4):
        x = rng.normal(size=2000)
        y = (rng.random(2000) < 1 / (1 + np.exp(-(0.3 + 0.8 * x)))).astype(int)
        fits.append(logit_stream("y ~ x", pl.DataFrame({"y": y, "x": x}), verbose=False))
    pm = pool_matrix(fits)
    assert pm.names == ["Intercept", "x"] and pm.m == 4
    np.testing.assert_allclose(pm.estimate, np.mean([f.beta for f in fits], axis=0))
    np.testing.assert_allclose(pm.within, np.mean([f.vcov for f in fits], axis=0))


# ---------------------------------------------------------------- draws
def _people(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    lat = 0.8 * x1 - 0.5 * x2 + rng.logistic(size=n)
    return pl.DataFrame({"pid": np.arange(n) * 7 + 3, "x1": x1, "x2": x2,
                         "y": np.searchsorted([-1.0, 0.3, 1.5], lat),
                         "b": (lat > 0).astype(int)})


def _check_freq(drawn, P):
    """The draws' frequency of each level against its mean probability,
    within 4.5 sds (the draws are independent Bernoullis)."""
    n = P.shape[0]
    for j in range(P.shape[1]):
        f = (drawn == j).mean()
        sd = np.sqrt(np.sum(P[:, j] * (1 - P[:, j]))) / n
        assert abs(f - P[:, j].mean()) < 4.5 * sd, (j, f, P[:, j].mean())


@pytest.mark.parametrize("model", ["binary", "ordered", "multinomial"])
def test_draws_follow_the_probabilities(model):
    d = _people()
    if model == "binary":
        f = logit_stream("b ~ x1 + x2", d, verbose=False)
        p = f.predict(d, type="response").collect()["prediction"].to_numpy()
        P = np.column_stack([1 - p, p])
        r = f.predict(d, type="draw", seed=4, id="pid").collect()
        drawn = r["prediction"].to_numpy()
    else:
        fit = ologit_stream if model == "ordered" else mlogit_stream
        f = fit("y ~ x1 + x2", d, verbose=False)
        P = f.predict(d).collect().select(pl.col("^prob_.*$")).to_numpy()
        r = f.predict(d, type="draw", seed=4, id="pid").collect()
        drawn = r["drawn"].to_numpy()
    _check_freq(drawn, P)
    # the draw depends on the seed and the identifier, not the rows' order
    sh = d.sample(fraction=1.0, shuffle=True, seed=1)
    col = "prediction" if model == "binary" else "drawn"
    r2 = f.predict(sh.lazy(), type="draw", seed=4, id="pid").collect()
    a = r.select("pid", col).sort("pid")
    b = r2.select("pid", col).sort("pid")
    assert a[col].to_list() == b[col].to_list()
    r3 = f.predict(d, type="draw", seed=5, id="pid").collect()
    assert (r3[col] != r[col]).mean() > 0.1
    # by default, the row's position
    r4 = f.predict(d, type="draw", seed=4).collect()
    assert r4.columns == r.columns
    assert r4[col].to_list() == f.predict(d, type="draw", seed=4).collect()[col].to_list()


def test_draws_with_string_ids_and_nulls():
    d = _people(2000).with_columns(pl.format("p{}", pl.col("pid")).alias("sid"))
    f = mlogit_stream("y ~ x1 + x2", d, verbose=False)
    a = f.predict(d, type="draw", seed=1, id="sid").collect()["drawn"]
    b = f.predict(d.reverse(), type="draw", seed=1, id="sid").collect()["drawn"].reverse()
    assert a.to_list() == b.to_list()
    nul = d.with_columns(pl.when(pl.col("pid") == 3).then(None).otherwise(pl.col("x1"))
                         .alias("x1"))
    r = f.predict(nul, type="draw", seed=1, id="pid").collect()
    assert r["drawn"].null_count() == 1
    with pytest.raises(ValueError, match="not in newdata"):
        f.predict(d, type="draw", id="nope")
    with pytest.raises(TypeError, match="integer"):
        f.predict(d, type="draw", seed=1.5)
    with pytest.raises(ValueError, match="type must be"):
        f.predict(d, type="sample")


def test_conditional_draws_one_alternative_per_set():
    df = _choice_data(seed=2, F=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = clogit_stream("chosen ~ dist | estab", df, case="case", verbose=False)
    p = f.predict(df).collect()
    r = f.predict(df, type="draw", seed=3, alt="estab").collect()
    per = r.group_by("case").agg(pl.col("drawn").sum(), pl.col("drawn").null_count().alias("n"))
    ok = per.filter(pl.col("n") == 0)
    assert (ok["drawn"] == 1).all()
    # the share of sets drawing their nearest establishment against its probability
    near = p.with_columns((pl.col("dist") == pl.col("dist").min().over("case")).alias("near"))
    q = near.filter("near")["prob"].fill_null(0).to_numpy()
    got = near.join(r.select("case", "estab", "drawn"), on=["case", "estab"]) \
        .filter("near")["drawn"].fill_null(False).to_numpy()
    assert abs(got.mean() - q.mean()) < 4.5 * np.sqrt(np.sum(q * (1 - q))) / len(q)
    # the same draw whatever the order of the rows, with the alternatives ordered by `alt`
    r2 = f.predict(df.sample(fraction=1.0, shuffle=True, seed=0), type="draw", seed=3,
                   alt="estab").collect()
    key = ["case", "estab"]
    assert r.select(*key, "drawn").sort(key).equals(r2.select(*key, "drawn").sort(key))
    # with calibration: drawn under the calibrated probabilities
    tot = df.group_by("estab").agg(pl.col("chosen").sum().cast(pl.Float64).alias("n"))
    new = df.drop("chosen").with_columns(pl.col("estab") + 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = f.predict(new, type="draw", seed=3, calibrate={"estab": tot}).collect()
    assert (rc.group_by("case").agg(pl.col("drawn").sum())["drawn"] == 1).all()
    # (its default order is the calibrated dimension's)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc2 = f.predict(new.reverse(), type="draw", seed=3, calibrate={"estab": tot}).collect()
    assert rc.select(*key, "drawn").sort(key).equals(rc2.select(*key, "drawn").sort(key))
