"""`predict(effects="average")`: probabilities averaged over the random
intercepts' distribution -- a seen group's conditional one (normal, its
mode and conditional covariance), an unseen group's prior N(0, Sigma) --
against brute-force integration, for the binary, ordered and multinomial
logit (Laplace and adaptive fits); the stored conditional covariances
against `ranef`'s sds; the averaged probabilities' calibration on new
groups, where the probabilities at the mean effect are biased; refusals."""

import numpy as np
import polars as pl
import pytest
from scipy.special import expit, softmax
from test_random import _binary_data, _ordered_data
from test_random_adaptive import _small
from test_random_adaptive_mnl import _data as _mnl_data

from logit_stream import logit_stream, mlogit_stream, ologit_stream

_Z = np.linspace(-12, 12, 4001)
_PHI = np.exp(-_Z * _Z / 2)
_PHI /= _PHI.sum()


def _avg1(f, mu, sd):
    """E f(mu + sd Z), Z standard normal, on a dense grid."""
    return np.tensordot(_PHI, f(mu + sd * _Z), axes=1)


@pytest.mark.parametrize("method", ["laplace", "adaptive"])
def test_binary_seen_and_unseen_groups(method):
    df = _binary_data()
    f = logit_stream("y ~ x1 + x2", df, random="g", random_method=method, verbose=False)
    new = pl.DataFrame({"x1": [0.5, -1.0, 0.5, 2.0], "x2": [1.0, 0.0, 1.0, 0.0],
                        "g": [3, 7, 10**6, 10**6 + 1]})
    got = f.predict(new, type="response", effects="average").collect()["prediction"].to_numpy()
    r = f.ranef().collect()
    re = dict(zip(r["g"], r["re_g"]))
    sd = dict(zip(r["g"], r["sd_g"]))
    tau = f.random_effects["g"]["sd"][0]
    eta = f.beta[0] + new["x1"].to_numpy() * f.beta[1] + new["x2"].to_numpy() * f.beta[2]
    want = [_avg1(expit, eta[i] + re.get(g, 0.0), sd.get(g, tau))
            for i, g in enumerate(new["g"])]
    np.testing.assert_allclose(got, want, rtol=1e-7)
    # the mode's is unchanged, and differs
    mode = f.predict(new, type="response").collect()["prediction"].to_numpy()
    np.testing.assert_allclose(mode, expit(eta + np.array([re.get(g, 0.0) for g in new["g"]])))
    assert np.all(np.abs(got - mode) > 1e-4)
    cls = f.predict(new, type="class", effects="average").collect()["prediction"].to_numpy()
    np.testing.assert_array_equal(cls, (got > 0.5).astype(int))


def test_stored_covariances_are_ranef_sds():
    for f in (logit_stream("y ~ x1 + x2", _binary_data(), random="g", verbose=False),
              logit_stream("y ~ x1 + x2", _binary_data(), random="g",
                           random_method="adaptive", verbose=False),
              mlogit_stream("y ~ x", _mnl_data(), random="g", verbose=False),
              mlogit_stream("y ~ x", _mnl_data(), random="g", random_method="adaptive",
                            verbose=False)):
        C = f._spec["ranef_cov"]["g"]
        r = f.ranef().collect()
        sds = r.select(pl.col("^sd_g.*$")).to_numpy()
        np.testing.assert_allclose(np.sqrt(np.diagonal(C, axis1=1, axis2=2)), sds, rtol=1e-10,
                                   atol=1e-12)
        if C.shape[1] > 1:
            # correlated, as the conditional covariance is
            assert np.any(np.abs(C[:, 1, 2]) > 1e-3)


def test_ordered_seen_and_unseen_groups():
    df = _ordered_data()
    f = ologit_stream("y ~ x1 + x2", df, random="g", verbose=False)
    new = pl.DataFrame({"x1": [0.3, 0.3], "x2": [1.0, 1.0], "g": [5, 10**6]})
    got = f.predict(new, effects="average").collect().select(pl.col("^prob_.*$")).to_numpy()
    r = f.ranef().collect().filter(pl.col("g") == 5)
    eta = 0.3 * f.beta[0] + f.beta[1]
    cuts = f.cutpoints

    def probs(e):
        c = expit(cuts[None, :] - e[:, None])
        return np.diff(np.column_stack([np.zeros(len(e)), c, np.ones(len(e))]), axis=1)

    tau = f.random_effects["g"]["sd"][0]
    want = [_avg1(probs, eta + r["re_g"][0], r["sd_g"][0]), _avg1(probs, eta, tau)]
    np.testing.assert_allclose(got, want, rtol=1e-7)


@pytest.mark.parametrize("method", ["laplace", "adaptive"])
def test_multinomial_against_a_dense_grid(method):
    """J = 3: the 5 x 5 product rule against a 2-D grid of 300 x 300 over
    the effect's distribution, for a seen group (its conditional mean and
    covariance) and an unseen one (Sigma)."""
    df = _mnl_data()
    f = mlogit_stream("y ~ x", df, random="g", random_method=method, verbose=False)
    new = pl.DataFrame({"x": [0.7, -0.4], "g": [4, 10**6]})
    got = f.predict(new, effects="average").collect().select(pl.col("^prob_.*$")).to_numpy()
    full = f.predict(new.with_columns(pl.lit(10**6 + 5).alias("g")), type="link").collect() \
        .select(pl.col("^xb_.*$")).to_numpy()                       # without the effects
    r = f.ranef().collect().filter(pl.col("g") == 4)
    mu = r.select(pl.col("^re_g.*$")).to_numpy()[0, 1:]
    row = f._spec["ranef"]["g"].with_row_index("i").filter(pl.col("g") == 4)["i"][0]
    C = f._spec["ranef_cov"]["g"][row][1:, 1:]
    S = np.asarray(f.random_effects["g"]["Sigma"])
    z = np.linspace(-7, 7, 301)
    w = np.exp(-z * z / 2)
    w /= w.sum()
    Z = np.array(np.meshgrid(z, z, indexing="ij")).reshape(2, -1).T
    W = np.outer(w, w).ravel()
    for i, (m, V) in enumerate([(mu, C), (np.zeros(2), S)]):
        L = np.linalg.cholesky(V)
        E = full[i][None, :] + m + Z @ L.T
        want = W @ softmax(np.column_stack([np.zeros(len(E)), E]), axis=1)
        np.testing.assert_allclose(got[i], want, atol=2e-5)


def test_averaging_is_calibrated_on_new_groups():
    """New groups drawn from the model: the mean averaged probability
    matches the share of 1s; the probability at the mean effect is too
    extreme for this rare outcome."""
    rng = np.random.default_rng(1)
    fit = logit_stream("y ~ x1 + x2", _small(2, seed=4, G=3000), random="g", verbose=False)
    b, tau = fit.beta, fit.random_effects["g"]["sd"][0]
    G, n = 20_000, 5
    g = np.repeat(np.arange(G), n) + 10**6
    x1 = rng.normal(size=G * n) - 1.5
    x2 = rng.binomial(1, 0.4, G * n).astype(float)
    y = rng.random(G * n) < expit(b[0] + b[1] * x1 + b[2] * x2 + rng.normal(0, tau, G)[g - 10**6])
    new = pl.DataFrame({"x1": x1, "x2": x2, "g": g})
    avg = fit.predict(new, type="response", effects="average").collect()["prediction"]
    mode = fit.predict(new, type="response").collect()["prediction"]
    se = np.sqrt(y.mean() * (1 - y.mean()) / G / n) * 2           # (within-group correlation)
    assert abs(avg.mean() - y.mean()) < 4 * se
    assert mode.mean() < y.mean() - 4 * se


def test_refusals():
    f = logit_stream("y ~ x1 + x2", _binary_data(), random="g", verbose=False)
    d = _binary_data().head(5)
    with pytest.raises(ValueError, match="averages probabilities"):
        f.predict(d, type="link", effects="average")
    with pytest.raises(ValueError, match="averages probabilities"):
        f.predict(d, type="draw", effects="average")
    with pytest.raises(ValueError, match="effects must be"):
        f.predict(d, type="response", effects="mean")
    ad = logit_stream("y ~ x1 + x2", _binary_data(), random="g", random_method="adaptive",
                      verbose=False)
    with pytest.raises(ValueError, match="parameter draw"):
        ad.draw_parameters(1, seed=1)[0].predict(d, type="response", effects="average")
    plain = logit_stream("y ~ x1 + x2", _binary_data(), verbose=False)
    with pytest.raises(ValueError, match="needs random intercepts"):
        plain.predict(d, type="response", effects="average").collect()
