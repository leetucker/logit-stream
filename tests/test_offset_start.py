"""Offsets and warm starts."""

import numpy as np
import polars as pl
import pytest

from logit_stream import clogit_stream, mlogit_stream, ologit_stream


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=n)
    g = rng.integers(0, 20, n)
    yo = np.searchsorted([-0.5, 0.8], 0.7 * x + rng.normal(0, 0.5, 20)[g] + rng.logistic(size=n))
    return pl.DataFrame({"yo": yo, "x": x, "g": g})


@pytest.mark.parametrize("fml,kw", [("yo ~ x", {}), ("yo ~ x | g", {}),
                                    ("yo ~ x", {"random": "g"})])
def test_ordered_offset_shifts_the_coefficient(data, fml, kw):
    """An offset of 2x is a coefficient of 2 on x held fixed: the estimate
    moves by exactly 2, with and without fixed effects or random
    intercepts."""
    a = ologit_stream(fml, data, verbose=False, tol=1e-14, **kw)
    b = ologit_stream(fml, data, offset=pl.col("x") * 2, verbose=False, tol=1e-14, **kw)
    assert b.coef()["x"] == pytest.approx(a.coef()["x"] - 2, abs=1e-6)
    assert b.loglik == pytest.approx(a.loglik, abs=1e-6)


def test_offset_in_predictions(data):
    f = ologit_stream("yo ~ x", data, offset=pl.col("x") * 2, verbose=False)
    xb = f.predict(data.head(5), type="link").collect()["xb"].to_numpy()
    np.testing.assert_allclose(xb, (f.coef()["x"] + 2) * data["x"][:5].to_numpy())
    # probabilities as the fit without the offset gives them
    g = ologit_stream("yo ~ x", data, verbose=False)
    p = f.predict(data.head(5)).collect().select(pl.col("^prob_.*$")).to_numpy()
    q = g.predict(data.head(5)).collect().select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(p, q, atol=1e-6)


def test_conditional_offset(choice_data):
    """McFadden's use: a known term of the utility (here 0.3 cost)."""
    c = choice_data.with_columns(lo=0.3 * pl.col("cost"))
    a = clogit_stream("chosen ~ cost + time", c, case="case", verbose=False, tol=1e-14)
    b = clogit_stream("chosen ~ cost + time", c, case="case", offset="lo", verbose=False,
                      tol=1e-14)
    assert b.coef()["cost"] == pytest.approx(a.coef()["cost"] - 0.3, abs=1e-8)
    np.testing.assert_allclose(b.resid().collect()["prob"].to_numpy(),
                               a.resid().collect()["prob"].to_numpy(), atol=1e-8)


def test_multinomial_offset_refused(data):
    with pytest.raises(ValueError, match="one per outcome level"):
        mlogit_stream("yo ~ x", data, offset="x", verbose=False)


def test_warm_starts(data):
    a = mlogit_stream("yo ~ x", data, verbose=False, tol=1e-12)
    for start in (a, dict(zip(a.coefnames, a.beta)), np.asarray(a.beta)):
        b = mlogit_stream("yo ~ x", data, verbose=False, tol=1e-12, start=start)
        np.testing.assert_allclose(b.beta, a.beta, atol=1e-9)
        assert b.diagnostics["newton"]["iterations"] < a.diagnostics["newton"]["iterations"]
    part = mlogit_stream("yo ~ x", data, verbose=False, tol=1e-12, start={"1:x": 0.5})
    np.testing.assert_allclose(part.beta, a.beta, atol=1e-9)
    o = ologit_stream("yo ~ x", data, verbose=False)
    assert ologit_stream("yo ~ x", data, verbose=False, start=o).converged
    with pytest.raises(ValueError, match="does not have"):
        mlogit_stream("yo ~ x", data, verbose=False, start={"x": 1.0})
    with pytest.raises(ValueError, match="values for"):
        mlogit_stream("yo ~ x", data, verbose=False, start=[0.0, 1.0])
    with pytest.raises(ValueError, match="feasible"):
        ologit_stream("yo ~ x", data, verbose=False, start={"cut1": 2.0, "cut2": 1.0})
