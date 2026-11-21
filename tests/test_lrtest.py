"""`lr_test`: the statistics against R (glm and lme4's glmer, nAGQ = 1:
anova's), the boundary mixture's choice and its p-values, its null
distribution in a simulation (half the statistics 0, the level held), and
refusals."""

import warnings

import numpy as np
import polars as pl
import pytest
from scipy import stats
from test_random import _binary_data
from test_random_adaptive_mnl import _data as _mnl_data

from logit_stream import clogit_stream, logit_stream, lr_test, mixlogit_stream, mlogit_stream

# R 4.5.3, lme4 2.0.6, on _binary_data(): glm(y ~ x1 + x2), glm(y ~ x1), glmer(y ~ x1 + x2 +
# (1 | g), nAGQ = 1); anova(m1, m0): Chisq 318.18; anova(m2, m0): Deviance 27.966, p 1.235e-07
R_LL = {"glm": -2974.45940155, "glm_x1": -2988.44231661, "glmer": -2815.36912313}


@pytest.fixture(scope="module")
def fits():
    d = _binary_data()
    return {"glm": logit_stream("y ~ x1 + x2", d, verbose=False),
            "glm_x1": logit_stream("y ~ x1", d, verbose=False),
            "glmer": logit_stream("y ~ x1 + x2", d, random="g", verbose=False, iwls_tol=1e-12,
                                  random_tol=1e-8)}


def test_matches_r(fits):
    for k, ll in R_LL.items():
        assert fits[k].loglik == pytest.approx(ll, abs=1e-4)
    t = lr_test(fits["glm_x1"], fits["glm"])
    assert t["statistic"] == pytest.approx(27.966, abs=1e-3) and t["df"] == 1
    assert t["p"] == pytest.approx(1.235e-07, rel=1e-3)
    assert t["reference"] == "chi2(1)" and not t["boundary"]
    t = lr_test(fits["glm"], fits["glmer"])
    assert t["statistic"] == pytest.approx(2 * (R_LL["glmer"] - R_LL["glm"]), abs=1e-3)
    # a random intercept against none: the 50:50 mixture, half the plain p-value
    assert t["boundary"] and t["reference"] == "0.5 chi2(0) + 0.5 chi2(1)"
    plain = lr_test(fits["glm"], fits["glmer"], boundary=False)
    assert plain["p"] == pytest.approx(stats.chi2.sf(t["statistic"], 1))
    assert t["p"] == pytest.approx(plain["p"] / 2)


def test_boundary_choices():
    # the multinomial's random intercept adds 3 parameters (2 variances, a covariance):
    # "auto" keeps the plain chi-squared; True is Stram and Lee's mixture
    d = _mnl_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = mlogit_stream("y ~ x", d, verbose=False)
        b = mlogit_stream("y ~ x", d, random="g", verbose=False)
    t = lr_test(a, b)
    assert t["df"] == 3 and not t["boundary"]
    s = lr_test(a, b, boundary=True)
    assert s["reference"] == "0.5 chi2(2) + 0.5 chi2(3)"
    assert s["p"] == pytest.approx(0.5 * stats.chi2.sf(s["statistic"], 2)
                                   + 0.5 * stats.chi2.sf(s["statistic"], 3))


def test_mixed_logit_random_coefficient_is_on_the_boundary():
    from logit_stream.simulate import simulate_choices
    ch = simulate_choices(3000, seed=2)
    c = clogit_stream("chosen ~ cost + time", ch, case="case", verbose=False)
    m = mixlogit_stream("chosen ~ cost + time", ch, case="case", random={"time": "normal"},
                        draws=100, verbose=False)
    t = lr_test(c, m)
    assert t["df"] == 1 and t["boundary"]


def test_null_distribution():
    """No group variance: about half the statistics are 0, and the mixture
    holds the level, where the plain chi-squared rejects less than half as
    often."""
    rng = np.random.default_rng(0)
    p, s = [], []
    for _ in range(200):
        G, n = 60, 10
        x = rng.normal(size=G * n)
        y = (rng.random(G * n) < 1 / (1 + np.exp(-(-0.3 + 0.5 * x)))).astype(int)
        d = pl.DataFrame({"y": y, "x": x, "g": np.repeat(np.arange(G), n)})
        t = lr_test(logit_stream("y ~ x", d, verbose=False),
                    logit_stream("y ~ x", d, random="g", verbose=False))
        p.append(t["p"])
        s.append(t["statistic"])
    p, s = np.array(p), np.array(s)
    assert 0.4 < (s < 1e-4).mean() < 0.7
    assert 0.04 < (p < 0.10).mean() < 0.17
    assert (np.minimum(1, 2 * p) < 0.10).mean() < (p < 0.10).mean()


def test_refusals(fits):
    with pytest.raises(ValueError, match="restricted model first"):
        lr_test(fits["glm"], fits["glm_x1"])
    d = _binary_data()
    with pytest.raises(ValueError, match="different rows"):
        lr_test(fits["glm_x1"], logit_stream("y ~ x1 + x2", d.head(4000), verbose=False))
    w = d.with_columns(pl.lit(2.0).alias("w"))
    with pytest.raises(ValueError, match="pseudo-likelihood"):
        lr_test(logit_stream("y ~ x1", w, weights="w", verbose=False), fits["glm"])
    with pytest.raises(ValueError, match="penalized"):
        lr_test(mlogit_stream("y ~ x1", d, l2=1.0, verbose=False), fits["glm"])
    with pytest.raises(ValueError, match="boundary must be"):
        lr_test(fits["glm_x1"], fits["glm"], boundary="yes")
    other = logit_stream("x2 ~ x1", d, verbose=False)
    with pytest.raises(ValueError, match="different outcomes"):
        lr_test(fits["glm_x1"], other)
