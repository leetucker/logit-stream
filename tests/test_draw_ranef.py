"""`draw_ranef` and `draw_parameters(ranef=True)`: each group's draw is from
its conditional distribution given its rows and the parameters, checked
against brute-force integration (its mean and sd exactly, the draws by
their probability integral transform), for the binary and the ordered
logit, small, large and one-outcome groups; invariance to the read's
chunks and to frequency weights; predict and ranef of the drawn result;
new groups; refusals."""

import numpy as np
import polars as pl
import pytest
from scipy.special import expit
from test_random_adaptive import _small

from logit_stream import draws as D
from logit_stream import logit_stream, ologit_stream

_Z = np.linspace(-1, 1, 8001)


def _brute(df, f, J):
    """Each group's conditional density on a dense grid, by brute force:
    (grid, cdf, mean, sd) per group, in the order of `f.ranef()`."""
    tau = f.random_effects["g"]["sd"][0]
    lev = dict(zip(*f.fixef().collect().select("cell", "fe_cell").to_numpy().T)) \
        if f.fe_names else None
    X = df.select("x1", "x2").to_numpy()
    names = list(f.coefnames)
    b = np.asarray(f.beta)
    if names[0] == "Intercept":
        eta = X @ b[1:3] + b[0]
        cuts = np.array([0.0])
    else:
        eta = X @ b[:2]
        cuts = f.cutpoints
    if lev is not None:
        eta = eta + np.array([lev[c] for c in df["cell"].to_numpy()])
    y = df["y"].to_numpy()
    g = df["g"].to_numpy()
    c = np.r_[-np.inf, cuts, np.inf]
    keys = f.ranef().collect()["g"].to_numpy()
    out = []
    for key in keys:
        rows = g == key
        grid = 10 * tau * _Z
        z = eta[rows][:, None] + grid[None, :]
        yy = y[rows][:, None]
        p = expit(c[yy + 1] - z) - expit(c[yy] - z)
        lp = np.log(np.maximum(p, 1e-300)).sum(axis=0) - 0.5 * grid**2 / tau**2
        w = np.exp(lp - lp.max())
        w /= w.sum()
        m = (w * grid).sum()
        out.append((grid, np.cumsum(w), m, np.sqrt((w * (grid - m) ** 2).sum())))
    return out


@pytest.mark.parametrize("J,fml", [(2, "y ~ x1 + x2 | cell"), (4, "y ~ x1 + x2"),
                                   (2, "y ~ x1 + x2")])
def test_is_the_conditional_distribution(J, fml):
    df = _small(J, seed=2, G=150, cells=3 if "cell" in fml else None)
    fit = (logit_stream if J == 2 else ologit_stream)(fml, df, random="g",
                                                      random_method="adaptive",
                                                      verbose=False)
    ref = _brute(df, fit, J)
    one = fit.draw_ranef(seed=1)
    t = one.ranef().collect()
    np.testing.assert_allclose(one.diagnostics["ranef_mean"], [r[2] for r in ref], atol=2e-5)
    np.testing.assert_allclose(t["sd_g"], [r[3] for r in ref], rtol=1e-4)
    # K draws at the estimates, in one read: their PIT under brute force's cdf is uniform
    K = 300
    many = D.draw_ranef_many([fit] * K, seed=3)
    A = np.stack([r.ranef().collect()["re_g"].to_numpy() for r in many])       # K x G
    pit = np.concatenate([np.interp(A[:, i], ref[i][0], ref[i][1]) for i in range(len(ref))])
    assert abs(pit.mean() - 0.5) < 0.01
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
    np.testing.assert_allclose(hist, 0.1, atol=0.012)
    assert len(set(A[:, 0])) == K                                  # independent draws


def test_one_outcome_and_large_groups():
    """Groups whose rows all have the same outcome (a skewed density), and
    groups of 400 rows (a narrow one), next to small ones."""
    rng = np.random.default_rng(4)
    G = 120
    size = np.where(np.arange(G) < 20, 400, 1 + rng.geometric(0.4, G))
    g = np.repeat(np.arange(G), size)
    x1, x2 = rng.normal(size=len(g)), rng.normal(size=len(g))
    lat = 0.5 * x1 + rng.normal(0, 1.2, G)[g] + rng.logistic(size=len(g))
    y = (lat > 0).astype(int)
    y[np.isin(g, [30, 31, 32])] = 1
    y[np.isin(g, [33, 34])] = 0
    df = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})
    fit = logit_stream("y ~ x1 + x2", df, random="g", random_method="adaptive", verbose=False)
    ref = _brute(df, fit, 2)
    one = fit.draw_ranef(seed=5)
    np.testing.assert_allclose(one.diagnostics["ranef_mean"], [r[2] for r in ref], atol=2e-5)
    np.testing.assert_allclose(one.ranef().collect()["sd_g"], [r[3] for r in ref], rtol=1e-4)


def test_chunks_and_frequency_weights():
    df = _small(2, seed=6, G=400, cells=4)
    fit = logit_stream("y ~ x1 + x2 | cell", df, random="g", random_method="adaptive",
                       verbose=False)
    a = D.draw_ranef_many([fit], seed=7)[0].ranef().collect()["re_g"].to_numpy()
    b = D.draw_ranef_many([fit], seed=7, chunk_cells=1)[0].ranef().collect()["re_g"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12)
    # rows repeated twice against frequency weights of 2: the same draws
    dup = pl.concat([df, df]).sort("g")
    fw = df.with_columns(pl.lit(2.0).alias("w"))
    f1 = logit_stream("y ~ x1 + x2 | cell", dup, random="g", random_method="adaptive",
                      verbose=False)
    f2 = logit_stream("y ~ x1 + x2 | cell", fw, random="g", random_method="adaptive",
                      verbose=False, weights="w", weights_type="fweights")
    np.testing.assert_allclose(f1.beta, f2.beta, rtol=1e-6)
    d1 = f1.draw_ranef(seed=8).ranef().collect().sort("g")["re_g"].to_numpy()
    d2 = f2.draw_ranef(seed=8).ranef().collect().sort("g")["re_g"].to_numpy()
    np.testing.assert_allclose(d1, d2, rtol=1e-5, atol=1e-6)


def test_predict_and_new_groups():
    df = _small(2, seed=9, G=300, cells=3)
    fit = logit_stream("y ~ x1 + x2 | cell", df, random="g", random_method="adaptive",
                       verbose=False)
    new = pl.DataFrame({"x1": [0.0, 0.5, 1.0, 0.0], "x2": [0.0, 1.0, 0.0, 0.0],
                        "cell": [0, 0, 1, 1], "g": [5, 10_000, 10_000, 10_001]})
    draws = fit.draw_parameters(4, seed=10, ranef=True, newdata=new)
    sds = []
    for d in draws:
        r = d.ranef().collect()
        re = dict(zip(r["g"].to_list(), r["re_g"].to_list()))
        assert r.filter(pl.col("g") >= 10_000)["n_obs"].to_list() == [0, 0]
        sds.append(r.filter(pl.col("g") == 10_000)["sd_g"][0])
        p = d.predict(new, type="link").collect()["prediction"].to_numpy()
        lev = dict(zip(*d.fixef().collect().select("cell", "fe_cell").to_numpy().T))
        fixed = new.select("x1", "x2").to_numpy() @ d.beta + np.array([lev[c] for c in
                                                                       new["cell"]])
        np.testing.assert_allclose(p, fixed + np.array([re[k] for k in new["g"]]), rtol=1e-12)
    # each draw's own sd for the new groups, and its own draws
    assert sds == [d.random_effects["g"]["sd"][0] for d in draws]
    assert len({d.ranef().collect()["re_g"][0] for d in draws}) == 4
    # the new groups' draws are the same on every call, and whatever newdata's order
    many = pl.concat([new] + [new.with_columns(pl.col("g") + 20_000 + 7 * i) for i in range(400)])
    t = [d.ranef().collect().sort("g")
         for d in (fit.draw_parameters(2, seed=10, ranef=True, newdata=many)[1],
                   fit.draw_parameters(2, seed=10, ranef=True, newdata=many)[1],
                   fit.draw_parameters(2, seed=10, ranef=True, newdata=many.reverse())[1])]
    for x in t[1:]:      # (the known groups' to rounding, from the threads' sums)
        assert x["g"].equals(t[0]["g"])
        np.testing.assert_allclose(x["re_g"].to_numpy(), t[0]["re_g"].to_numpy(), rtol=1e-12)
    # a draw of the group intercepts at the estimates: ranef() is the draw
    one = fit.draw_ranef(seed=1)
    assert not one.ranef().collect()["re_g"].equals(fit.ranef().collect()["re_g"])
    assert one.beta is fit.beta or np.array_equal(one.beta, fit.beta)


def test_refusals():
    df = _small(2, seed=1)
    with pytest.raises(ValueError, match="random intercepts"):
        logit_stream("y ~ x1 + x2", df, verbose=False).draw_ranef()
    f = logit_stream("y ~ x1 + x2", df, random="g", random_method="adaptive", verbose=False)
    with pytest.raises(ValueError, match="ranef=True"):
        f.draw_parameters(2, newdata=df)
