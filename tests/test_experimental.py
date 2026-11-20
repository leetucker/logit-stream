"""Which features warn with `ExperimentalWarning` (see
`logit_stream._experimental`): each experimental path does, at the caller's
line; the validated ones do not."""

import warnings

import hdfe_stream
import numpy as np
import polars as pl
import pytest

import logit_stream
from logit_stream import (
    ExperimentalWarning,
    clogit_stream,
    mixlogit_stream,
    mlogit_stream,
    nlogit_stream,
    ologit_stream,
)
from logit_stream import logit_stream as logit
from logit_stream.simulate import simulate_choices


def _groups(seed=0, G=80, size=6, J=2):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), size)
    x = rng.normal(size=len(g))
    lat = 0.5 * x + rng.normal(0, 0.7, G)[g] + rng.logistic(size=len(g))
    y = np.searchsorted([0.0] if J == 2 else [-0.8, 0.8], lat)
    return pl.DataFrame({"y": y, "x": x, "g": g})


def _caught(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn()
    return out, [x for x in w if issubclass(x.category, ExperimentalWarning)]


def test_the_class():
    assert issubclass(ExperimentalWarning, hdfe_stream.ExperimentalWarning)
    assert logit_stream.ExperimentalWarning is ExperimentalWarning


@pytest.mark.parametrize("case", ["slopes", "mnl_adaptive", "clogit_slopes", "mixed"])
def test_experimental_fits_warn(case):
    d = _groups()
    ch = simulate_choices(1500, seed=1)
    calls = {
        "slopes": lambda: logit("y ~ x", d, random={"g": ["1", "x"]}, verbose=False),
        "mnl_adaptive": lambda: mlogit_stream("y ~ x", _groups(J=3), random="g",
                                              random_method="adaptive", verbose=False),
        "clogit_slopes": lambda: clogit_stream(
            "chosen ~ cost + time", ch.with_columns((pl.col("case") % 10).alias("cz")),
            case="case", random={"cz": "time"}, verbose=False),
        "mixed": lambda: mixlogit_stream("chosen ~ cost + time", ch, case="case",
                                         random={"time": "normal"}, draws=50, verbose=False),
    }
    _, w = _caught(calls[case])
    assert w, case
    assert all("(experimental)" in str(x.message) for x in w)
    assert all(x.filename == __file__ for x in w)          # the caller's line


def test_experimental_postestimation_warns():
    f = logit("y ~ x", _groups(), random="g", random_method="adaptive", verbose=False)
    for fn in (lambda: f.draw_ranef(seed=1),
               lambda: f.draw_parameters(2, seed=1, ranef=True),
               lambda: f.predict(_groups().head(3), type="response", effects="average")):
        _, w = _caught(fn)
        assert w


def test_validated_features_do_not_warn():
    d, d3 = _groups(), _groups(J=3)
    ch = simulate_choices(1500, seed=1)
    calls = [
        lambda: logit("y ~ x", d, random="g", verbose=False),                     # Laplace
        lambda: logit("y ~ x", d, random="g", random_method="adaptive", verbose=False),
        lambda: ologit_stream("y ~ x", d3, random="g", random_method="adaptive",
                              verbose=False),
        lambda: mlogit_stream("y ~ x", d3, random="g", verbose=False),
        lambda: nlogit_stream("chosen ~ cost + time", ch, case="case", alt="alt",
                              nests={"public": ["train", "metro"], "car": ["car"]},
                              verbose=False),
        lambda: mlogit_stream("y ~ x", d3.with_columns(pl.col("x").alias("z")),
                              own_covariates={1: "z"}, verbose=False),
    ]
    for fn in calls:
        f, w = _caught(fn)
        assert not w, [str(x.message) for x in w]
    f = logit("y ~ x", d, random="g", random_method="adaptive", verbose=False)
    _, w = _caught(lambda: f.draw_parameters(3, seed=1))
    assert not w
    _, w = _caught(lambda: f.predict(d.head(3), type="draw", seed=1))
    assert not w
