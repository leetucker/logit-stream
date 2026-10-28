"""Multi-way CRV1, the CRV3 jackknife, and the conditional logit's check that
clusters are constant within choice sets."""

import numpy as np
import polars as pl
import pytest

from logit_stream import clogit_stream, mlogit_stream, ologit_stream


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n, G = 2400, 12
    c = rng.integers(0, G, n)
    b = rng.integers(0, 9, n)
    x = rng.normal(size=n) + rng.normal(0, 0.5, G)[c] + rng.normal(0, 0.3, 9)[b]
    x2 = rng.normal(size=n)
    E = np.column_stack([np.zeros(n), 0.4 + 0.6 * x + rng.normal(0, 0.4, G)[c],
                         -0.2 - 0.4 * x + 0.5 * x2])
    P = np.exp(E)
    P /= P.sum(1, keepdims=True)
    y = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    yo = np.searchsorted([-0.5, 0.8], 0.7 * x - 0.3 * x2 + rng.normal(0, 0.5, G)[c]
                         + rng.logistic(size=n))
    return pl.DataFrame({"y": y, "yo": yo, "x": x, "x2": x2, "c": c, "b": b})


def test_two_way_crv1_is_inclusion_exclusion(data):
    """V(c+b) = G/(G-1) [V0(c) + V0(b) - V0(c^b)], the unscaled one-way
    sandwiches, with G the smaller one-way count; df = G - 1."""
    f = mlogit_stream("y ~ x + x2", data, vcov={"CRV1": "c+b"}, cluster=["c", "b", "c^b"],
                      verbose=False)
    V = {k: v for k, (v, _) in f.all_vcovs.items()}
    G = {k: f.n_clusters[k] for k in ("c", "b", "c^b")}
    raw = {k: V[f"CRV1:{k}"] / (G[k] / (G[k] - 1)) for k in G}
    Gm = min(G["c"], G["b"])
    np.testing.assert_allclose(V["CRV1:c+b"], Gm / (Gm - 1) * (raw["c"] + raw["b"] - raw["c^b"]),
                               rtol=1e-12, atol=1e-15)
    assert f.df_t == Gm - 1 and f.n_clusters["c+b"] == [12, 9]


@pytest.mark.parametrize("fn,fml", [(mlogit_stream, "y ~ x + x2"),
                                    (ologit_stream, "yo ~ x + x2")])
def test_crv3_matches_the_jackknife(data, fn, fml):
    """The one-step CV3 is close to the jackknife of refits without each
    cluster (12 clusters of 200 rows: within 2% here)."""
    f = fn(fml, data, vcov={"CRV3": "c"}, verbose=False, tol=1e-14)
    G = 12
    D = np.array([fn(fml, data.filter(pl.col("c") != g), verbose=False, tol=1e-14).beta - f.beta
                  for g in range(G)])
    exact = np.sqrt(np.diag((G - 1) / G * D.T @ D))
    np.testing.assert_allclose(f.se, exact, rtol=0.02)
    assert f.vcov_type == "CRV3:c" and f.df_t == G - 1


def test_crv3_restrictions(data, choice_data):
    with pytest.raises(ValueError, match="CRV3 is available"):
        mlogit_stream("y ~ x | c", data, vcov={"CRV3": "b"}, verbose=False)
    with pytest.raises(ValueError, match="CRV3 is available"):
        clogit_stream("chosen ~ cost + time", choice_data, case="case",
                      vcov={"CRV3": "case"}, verbose=False)
    with pytest.raises(ValueError, match="one-way"):
        mlogit_stream("y ~ x", data, vcov={"CRV3": "c+b"}, verbose=False)


def test_conditional_logit_clusters_must_be_constant_within_sets(choice_data):
    bad = choice_data.with_row_index("r").with_columns(cl=pl.col("r") % 7)
    with pytest.raises(ValueError, match="constant within each choice set"):
        clogit_stream("chosen ~ cost + time", bad, case="case", vcov={"CRV1": "cl"},
                      verbose=False)
    ok = choice_data.with_columns(cl=pl.col("case") % 7)
    f = clogit_stream("chosen ~ cost + time", ok, case="case", vcov={"CRV1": "cl"},
                      verbose=False)
    assert f.n_clusters["cl"] == 7
