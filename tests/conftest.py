import warnings

import numpy as np
import polars as pl
import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _mnl_draw(rng, E):
    P = np.exp(E - E.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    return (P.cumsum(axis=1) > rng.random(len(E))[:, None]).argmax(axis=1)


@pytest.fixture(scope="session")
def mnl_data():
    """Four-level multinomial data with two continuous covariates, a
    categorical one, and a cluster variable."""
    rng = np.random.default_rng(1)
    n = 6000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    g = rng.integers(0, 4, n)
    cl = rng.integers(0, 30, n)
    B = np.array([[0.2, 0.5, -0.3], [-0.4, 0.1, 0.8], [0.3, -0.6, 0.2]])
    E = np.column_stack([np.zeros(n)] + [B[j, 0] + B[j, 1] * x1 + B[j, 2] * x2 + 0.15 * j * g
                                         for j in range(3)])
    y = _mnl_draw(rng, E)
    return pl.DataFrame({"y": np.array(["a", "b", "c", "d"])[y], "yi": y, "x1": x1, "x2": x2,
                         "g": g, "cl": cl, "w": rng.integers(1, 4, n).astype(float)})


@pytest.fixture(scope="session")
def ord_data():
    rng = np.random.default_rng(2)
    n = 6000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cl = rng.integers(0, 30, n)
    lat = 0.7 * x1 - 0.4 * x2 + rng.normal(size=30)[cl] * 0.3 + rng.logistic(size=n)
    y = np.digitize(lat, [-1.5, 0.0, 1.2])
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "cl": cl,
                         "w": rng.integers(1, 4, n).astype(float)})


@pytest.fixture(scope="session")
def fe_data():
    """Multinomial and ordered outcomes with group effects (40 groups)."""
    rng = np.random.default_rng(5)
    n, G = 8000, 40
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    g = rng.integers(0, G, n)
    A = rng.normal(size=(G, 3)) * 0.7
    B = np.array([[0.5, -0.3], [0.1, 0.8], [-0.6, 0.2]])
    E = np.column_stack([np.zeros(n)] + [B[j, 0] * x1 + B[j, 1] * x2 + A[g, j]
                                         for j in range(3)])
    y = _mnl_draw(rng, E)
    lat = 0.6 * x1 - 0.5 * x2 + A[g, 0] + rng.logistic(size=n)
    yo = np.digitize(lat, [-1.0, 0.3, 1.5])
    return pl.DataFrame({"y": y, "yo": yo, "x1": x1, "x2": x2, "g": g,
                         "cl": rng.integers(0, 25, n)})


@pytest.fixture(scope="session")
def choice_data():
    """Long-format choice data: 3 alternatives, some unavailable."""
    from logit_stream.simulate import simulate_choices
    return simulate_choices(3000, seed=4)
