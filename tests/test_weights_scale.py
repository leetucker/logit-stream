"""Analytic weights are rescaled to average 1 (Stata's convention for maximum
likelihood): multiplying them by a constant changes nothing -- not the iid
vcov, the log-likelihood, Firth's penalized fit or the random intercepts'
variance."""

import numpy as np
import polars as pl
import pytest

from logit_stream import clogit_stream, mlogit_stream, ologit_stream


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    g = rng.integers(0, 30, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(x + rng.normal(0, 0.7, 30)[g])))).astype(int)
    w = rng.uniform(0.5, 2, n)
    return pl.DataFrame({"y": y, "x": x, "g": g, "w": w, "w10": 10 * w})


@pytest.mark.parametrize("fn,kw", [(mlogit_stream, {}), (mlogit_stream, {"firth": True}),
                                   (ologit_stream, {}), (mlogit_stream, {"random": "g"}),
                                   (ologit_stream, {"fml": "y ~ x | g"})])
def test_scale_free(data, fn, kw):
    kw = dict(kw)
    fml = kw.pop("fml", "y ~ x")
    a = fn(fml, data, weights="w", verbose=False, **kw)
    b = fn(fml, data, weights="w10", verbose=False, **kw)
    np.testing.assert_allclose(a.beta, b.beta, atol=1e-8)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-7)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-9)


def test_conditional_scale_free(choice_data):
    c = choice_data.with_columns(w=(pl.col("case") % 3 + 1).cast(pl.Float64))
    c = c.with_columns(w10=10 * pl.col("w"))
    a = clogit_stream("chosen ~ cost + time", c, case="case", weights="w", verbose=False)
    b = clogit_stream("chosen ~ cost + time", c, case="case", weights="w10", verbose=False)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-9)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)


def test_frequency_weights_are_counts(data):
    """Frequency weights are not rescaled: w = 2 everywhere is the data
    twice (twice the log-likelihood, standard errors / sqrt 2)."""
    two = data.with_columns(f=pl.lit(2.0))
    a = mlogit_stream("y ~ x", data, verbose=False)
    b = mlogit_stream("y ~ x", two, weights="f", weights_type="fweights", verbose=False)
    assert b.loglik == pytest.approx(2 * a.loglik)
    np.testing.assert_allclose(b.se, a.se / np.sqrt(2), rtol=1e-8)
