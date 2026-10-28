"""The simulators' options for the nested, mixed and fixed-effects logit:
their probabilities, and the estimators recovering what they draw from."""

import numpy as np
import polars as pl
import pytest

from logit_stream import cfelogit_stream, mixlogit_stream, nlogit_stream
from logit_stream.simulate import _nested_probs, simulate_choices, simulate_panel

ALTS = ["train", "metro", "car"]


def test_nested_probabilities():
    """Two levels against the closed form, unavailable alternatives (and a
    nest with none available) included; a nest of nests by hand."""
    rng = np.random.default_rng(0)
    V = rng.normal(size=(6, 3))
    V[1, 1] = V[2, 2] = -np.inf
    lam = 0.6
    P = _nested_probs(V, ALTS, {"rail": ["train", "metro"], "road": ["car"]}, lam)
    with np.errstate(divide="ignore"):
        I = np.log(np.exp(V[:, 0] / lam) + np.exp(V[:, 1] / lam))
    D = np.exp(lam * I) + np.exp(V[:, 2])
    ref = np.column_stack([np.exp(lam * I) / D * np.exp(V[:, j] / lam - I) for j in (0, 1)]
                          + [np.exp(V[:, 2]) / D])
    np.testing.assert_allclose(P, ref, atol=1e-12)
    assert P[1, 1] == 0 and P[2, 2] == 0
    tree = {"public": {"rail": ["train", "metro"], "bus": ["bus"]}, "private": ["car"]}
    P = _nested_probs(np.zeros((1, 4)), ["train", "metro", "bus", "car"], tree,
                      {"public": 0.7, "rail": 0.4})
    rail = 0.4 * np.log(2)
    I_pub = np.log(np.exp(rail / 0.7) + 1)
    p_pub = np.exp(0.7 * I_pub) / (np.exp(0.7 * I_pub) + 1)
    np.testing.assert_allclose(P[0, 2], p_pub / np.exp(I_pub), atol=1e-12)
    np.testing.assert_allclose(P.sum(), 1.0, atol=1e-12)
    with pytest.raises(ValueError, match="every alternative"):
        _nested_probs(V, ALTS, {"rail": ["train"], "road": ["car"]}, lam)


def test_defaults_and_panel_columns():
    a = simulate_choices(500, seed=3)
    assert (a["person"] == a["case"]).all()
    b = simulate_choices(500, tasks=5, seed=3)
    assert (b["person"] == b["case"] // 5).all()
    assert b.group_by("person").agg(pl.col("income").n_unique())["income"].max() == 1
    with pytest.raises(ValueError, match="not both"):
        simulate_choices(10, nests={"a": ALTS}, sd_time=1.0)


def test_nested_recovered():
    nests = {"rail": ["train", "metro"], "road": ["car"]}
    d = simulate_choices(20_000, nests=nests, lambdas=0.5, seed=1).with_columns(
        income_car=pl.col("income") * (pl.col("alt") == "car"))
    f = nlogit_stream("chosen ~ cost + time + C(alt) + income_car", d, case="case",
                      alt="alt", nests=nests, verbose=False)
    truth = {"cost": -1.1, "time": -1.3, "lambda.rail": 0.5, "income_car": 0.3}
    for k, v in truth.items():
        assert abs(f.coef()[k] - v) < 4 * f.se[f.coefnames.index(k)], k


def test_mixed_recovered():
    d = simulate_choices(10_000, tasks=5, sd_time=1.0, seed=2).with_columns(
        income_car=pl.col("income") * (pl.col("alt") == "car"))
    f = mixlogit_stream("chosen ~ cost + time + C(alt) + income_car", d, case="case",
                        panel="person", random={"time": "normal"}, draws=100, verbose=False)
    for k, v in {"cost": -1.1, "time": -1.3, "sd.time": 1.0}.items():
        assert abs(f.coef()[k] - v) < 4 * f.se[f.coefnames.index(k)], k


def test_panel_recovered():
    p = simulate_panel(4000, 4, seed=1)
    assert p.height == 16_000 and set(p["period"].unique()) == {1, 2, 3, 4}
    f = cfelogit_stream("y ~ x1 + x2 | unit", p, verbose=False)
    for k, v in {"x1": 1.0, "x2": -0.5}.items():
        assert abs(f.coef()[k] - v) < 4 * f.se[f.coefnames.index(k)], k
