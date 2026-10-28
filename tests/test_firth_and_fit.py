"""Firth's penalized likelihood (against brglm2), the within pseudo
R-squared with a fixed effect, and the final Newton step."""

import numpy as np
import polars as pl
import pytest
from scipy.special import xlogy

from logit_stream import mlogit_stream, ologit_stream


def _firth_data():
    """The data the brglm2 numbers below were computed on (R 4.4, brglm2
    from conda-forge): a 3-level outcome, and a binary one that the
    indicator d separates (every d = 1 row has y = 1)."""
    rng = np.random.default_rng(3)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = (rng.random(n) < 0.08).astype(float)
    E = np.column_stack([np.zeros(n), 0.3 + 0.8 * x1 - 0.5 * d, -0.4 + 0.5 * x2 + 1.2 * d])
    P = np.exp(E)
    P /= P.sum(1, keepdims=True)
    y = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    yb = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 0.7 * x1)))).astype(int)
    yb[d == 1] = 1
    return pl.DataFrame({"y": y, "yb": yb, "x1": x1, "x2": x2, "d": d})


def test_firth_multinomial_matches_brmultinom():
    """brmultinom(y ~ x1 + x2 + d, type = "AS_mean"): for the multinomial
    logit (canonical link) the mean-bias-reducing adjusted score is Firth's."""
    f = mlogit_stream("y ~ x1 + x2 + d", _firth_data(), firth=True, verbose=False, tol=1e-14)
    b = [0.20631866080707, 0.93324001922491, -0.11023238589004, -0.47494326182749,
         -0.26605198219717, 0.00258443162593, 0.30338859034681, 0.87349964934407]
    se = [0.132494321514, 0.146636514922, 0.128828114476, 0.564157821777,
          0.148012529422, 0.142597952271, 0.141012329240, 0.481156298146]
    assert f.firth and f.converged
    np.testing.assert_allclose(f.beta, b, atol=1e-7)
    np.testing.assert_allclose(f.se, se, rtol=1e-6)
    assert "Firth" in f.summary_text()
    # the reported log-likelihood is the plain one, below the maximum likelihood fit's
    ml = mlogit_stream("y ~ x1 + x2 + d", _firth_data(), verbose=False, tol=1e-14)
    assert f.loglik < ml.loglik


def test_firth_binary_separation_matches_brglm2():
    """d separates the outcome: the MLE of its coefficient is infinite, the
    penalized one finite. brglm2's glm(..., method = "brglmFit",
    type = "AS_mean"), as logistf's estimates."""
    f = mlogit_stream("yb ~ x1 + d", _firth_data(), firth=True, verbose=False, tol=1e-14)
    assert f.converged
    np.testing.assert_allclose(f.beta, [-0.305298861500, 0.709370259507, 4.551778442246],
                               atol=1e-6)
    np.testing.assert_allclose(f.se, [0.111183351790, 0.123301429907, 1.470634232858],
                               rtol=1e-5)


def test_firth_restrictions():
    df = _firth_data().with_columns(g=pl.col("x1").abs().round(0))
    with pytest.raises(ValueError, match="canonical"):
        ologit_stream("y ~ x1", df, firth=True, verbose=False)
    with pytest.raises(ValueError, match="two penalties"):
        mlogit_stream("y ~ x1", df, firth=True, l2=1.0, verbose=False)
    with pytest.raises(ValueError, match="absorbed"):
        mlogit_stream("y ~ x1 | g", df, firth=True, verbose=False)


def _fe_data(seed=5):
    rng = np.random.default_rng(seed)
    n = 3000
    g = rng.integers(0, 20, n)
    x = rng.normal(size=n)
    a = rng.normal(0, 0.8, 20)[g]
    E = np.column_stack([np.zeros(n), 0.5 * x + a, -0.3 * x - a])
    P = np.exp(E)
    P /= P.sum(1, keepdims=True)
    y = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    lat = 0.7 * x + a + rng.logistic(size=n)
    yo = np.searchsorted([-1.0, 0.3, 1.5], lat)
    return pl.DataFrame({"y": y, "yo": yo, "x": x, "g": g})


def test_within_pseudo_r2_multinomial():
    """With an absorbed fixed effect, the null with only it is each level's
    outcome shares."""
    df = _fe_data()
    f = mlogit_stream("y ~ x | g", df, verbose=False)
    t = df.group_by("g", "y").len().pivot(on="y", index="g", values="len").fill_null(0)
    n = t.drop("g").to_numpy().astype(float)
    ll = float(np.sum(xlogy(n, n / n.sum(1, keepdims=True))))
    assert f.loglik_fe_null == pytest.approx(ll, abs=1e-8)
    assert f.pseudo_r2_within == pytest.approx(1 - f.loglik / ll)
    assert 0 < f.pseudo_r2_within < f.pseudo_r2
    assert "within pseudo R2" in f.summary_text()
    assert np.isnan(mlogit_stream("y ~ x", df, verbose=False).pseudo_r2_within)


def test_within_pseudo_r2_ordered():
    """The ordered logit's null with only the fixed effect (a location per
    level, common cutpoints), fitted on the count table, is the fit with the
    level indicators and nothing else."""
    df = _fe_data()
    f = ologit_stream("yo ~ x | g", df, verbose=False)
    r = ologit_stream("yo ~ C(g)", df, verbose=False, tol=1e-14)
    assert f.loglik_fe_null == pytest.approx(r.loglik, abs=1e-6)
    assert f.pseudo_r2_within == pytest.approx(1 - f.loglik / r.loglik, rel=1e-7)


def test_the_final_newton_step_is_taken():
    """At the default tolerance the estimates are those of a tight one to
    far better than the ~1e-4 standard errors of the point where the
    decrement was checked, since the step it computed is taken."""
    df = _fe_data()
    a = mlogit_stream("y ~ x", df, verbose=False)
    b = mlogit_stream("y ~ x", df, verbose=False, tol=1e-14)
    assert np.max(np.abs(a.beta - b.beta) / b.se) < 1e-6
