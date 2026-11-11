"""The robust and clustered variances of the multinomial and ordered logit
(meat_kernels.py) against the NumPy code (`_Driver.meat`)."""

import numpy as np
import polars as pl
import pytest

import logit_stream.meat_kernels as MK
import logit_stream.models as M
from logit_stream import mlogit_stream, oprobit_stream, ologit_stream


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(31)
    n, K = 40_000, 4
    X = rng.normal(size=(n, K))
    lat = X @ rng.normal(scale=0.4, size=K) + rng.logistic(size=n)
    E = np.column_stack([np.zeros(n)] + [X @ rng.normal(scale=0.3, size=K) for _ in range(3)])
    P = np.exp(E - E.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    S = rng.dirichlet(np.ones(4), n)
    return pl.DataFrame({
        "ym": ((rng.random(n)[:, None] > P.cumsum(1)).sum(1).clip(0, 3)).astype(str),
        "yo": np.digitize(lat, [-1.0, 0.0, 1.0]), "cl": rng.integers(0, 400, n),
        "cl2": rng.integers(0, 9, n), "clmany": rng.integers(0, 30_000, n),
        "w": rng.integers(1, 4, n).astype(float), "o": rng.normal(scale=0.2, size=n),
        **{f"x{j}": X[:, j] for j in range(K)}, **{f"s{k}": S[:, k] for k in range(1, 4)}})


RHS = "x0 + x1 + x2 + x3"
CASES = {
    "mlogit hetero": (mlogit_stream, f"ym ~ {RHS}", dict(levels=list("0123"), vcov="hetero")),
    "mlogit CRV1": (mlogit_stream, f"ym ~ {RHS}", dict(levels=list("0123"), vcov={"CRV1": "cl"})),
    "mlogit two-way, analytic weights": (
        mlogit_stream, f"ym ~ {RHS}",
        dict(levels=list("0123"), vcov={"CRV1": "cl+cl2"}, weights="w")),
    "mlogit hetero, frequency weights": (
        mlogit_stream, f"ym ~ {RHS}",
        dict(levels=list("0123"), vcov="hetero", weights="w", weights_type="fweights")),
    "mlogit own covariates, CRV1": (
        mlogit_stream, "ym ~ x0 + x1",
        dict(levels=list("0123"), vcov={"CRV1": "cl"},
             own_covariates={"1": "s1", "2": "s2", "3": "s3"})),
    "ologit hetero": (ologit_stream, f"yo ~ {RHS}", dict(vcov="hetero")),
    "ologit CRV1, offset": (ologit_stream, f"yo ~ {RHS}",
                            dict(vcov={"CRV1": "cl"}, offset="o")),
    "ologit two-way, frequency weights": (
        ologit_stream, f"yo ~ {RHS}",
        dict(vcov={"CRV1": "cl+cl2"}, weights="w", weights_type="fweights")),
    "oprobit analytic weights, hetero": (oprobit_stream, f"yo ~ {RHS}",
                                         dict(vcov="hetero", weights="w")),
}


def _count(monkeypatch):
    calls = {"n": 0}
    for cls in (MK.MultinomialMeat, MK.OrderedMeat):
        add = cls.add

        def spy(self, blk, add=add):
            calls["n"] += 1
            return add(self, blk)
        monkeypatch.setattr(cls, "add", spy)
    return calls


@pytest.mark.parametrize("owner", [False, True], ids=["copies per part", "owned clusters"])
@pytest.mark.parametrize("name", list(CASES))
def test_fused_variance_equals_the_numpy_code(data, monkeypatch, name, owner):
    fn, fml, kw = CASES[name]
    calls = _count(monkeypatch)
    if owner:
        monkeypatch.setattr(MK, "MEAT_CLUSTER_MB", 1e-6)      # no copy per part fits
    kw = dict(verbose=False, tol=1e-13, **kw)
    fast = fn(fml, data, **kw)
    assert calls["n"] >= 1
    calls["n"] = 0
    monkeypatch.setattr(M, "MNL_KERNEL_MIN_ROWS", 10**9)        # the NumPy code
    ref = fn(fml, data, **kw)
    assert calls["n"] == 0
    np.testing.assert_allclose(fast.beta, ref.beta, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fast.vcov, ref.vcov, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(fast.se, ref.se, rtol=1e-8)


def test_many_clusters_use_the_owned_ranges(data, monkeypatch):
    """More clusters than an eighth of a block's rows: no copy per part."""
    seen = {}
    orig = MK._mnl_cluster_pass
    monkeypatch.setattr(MK, "_mnl_cluster_pass",
                        lambda *a, **k: (seen.setdefault("owner", True), orig(*a, **k))[1])
    kw = dict(levels=list("0123"), vcov={"CRV1": "clmany"}, verbose=False, tol=1e-13)
    fast = mlogit_stream(f"ym ~ {RHS}", data, **kw)
    assert seen.get("owner")
    monkeypatch.setattr(M, "MNL_KERNEL_MIN_ROWS", 10**9)
    ref = mlogit_stream(f"ym ~ {RHS}", data, **kw)
    np.testing.assert_allclose(fast.vcov, ref.vcov, rtol=1e-8, atol=1e-14)


def test_a_fixed_effect_or_a_wide_design_keeps_the_numpy_code(data, monkeypatch):
    calls = _count(monkeypatch)
    d = data.with_columns(g=pl.col("cl") % 25)
    mlogit_stream(f"ym ~ {RHS} | g", d, levels=list("0123"), vcov="hetero", verbose=False)
    assert calls["n"] == 0                                 # (the fixed-effects passes' own)
    monkeypatch.setattr(M, "ORD_KERNEL_MAX_K", 2)
    ologit_stream(f"yo ~ {RHS}", data, vcov="hetero", verbose=False)
    assert calls["n"] == 0


# ------------------------------------------- the conditional and fixed-effects logit
@pytest.fixture(scope="module")
def choices():
    from logit_stream.simulate import simulate_choices
    rng = np.random.default_rng(5)
    d = simulate_choices(3000, seed=4)
    return d.with_columns(
        cl=pl.col("case") % 120, cl2=pl.col("case") % 7, clmany=pl.col("case") % 2000,
        w=(pl.col("case") % 4 + 1).cast(pl.Float64),
        o=pl.Series(rng.normal(scale=0.2, size=d.height)))


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(6)
    N, T = 2500, 4
    i = np.repeat(np.arange(N), T)
    a = rng.normal(size=N)[i]
    x1 = rng.normal(size=N * T) + 0.5 * a
    x2 = rng.normal(size=N * T)
    y = (rng.random(N * T) < 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2 + a)))).astype(int)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "i": i, "cl": i % 120, "cl2": i % 7,
                         "clmany": i % 2000, "w": (i % 4 + 1).astype(float),
                         "o": rng.normal(scale=0.2, size=N * T)}).sample(
        fraction=1.0, shuffle=True, seed=1)


CFML = "chosen ~ cost + time + C(alt) + income:C(alt)"
CASE_CASES = {
    "clogit hetero": ("choices", lambda: __import__("logit_stream").clogit_stream, CFML,
                      dict(case="case", vcov="hetero")),
    "clogit CRV1": ("choices", lambda: __import__("logit_stream").clogit_stream, CFML,
                    dict(case="case", vcov={"CRV1": "cl"})),
    "clogit two-way, analytic weights": (
        "choices", lambda: __import__("logit_stream").clogit_stream, CFML,
        dict(case="case", vcov={"CRV1": "cl+cl2"}, weights="w")),
    "clogit frequency weights, offset": (
        "choices", lambda: __import__("logit_stream").clogit_stream, CFML,
        dict(case="case", vcov="hetero", weights="w", weights_type="fweights", offset="o")),
    "clogit many clusters": ("choices", lambda: __import__("logit_stream").clogit_stream, CFML,
                             dict(case="case", vcov={"CRV1": "clmany"})),
    "cfelogit hetero": ("panel", lambda: __import__("logit_stream").cfelogit_stream,
                        "y ~ x1 + x2 | i", dict(vcov="hetero")),
    "cfelogit CRV1": ("panel", lambda: __import__("logit_stream").cfelogit_stream,
                      "y ~ x1 + x2 | i", dict(vcov={"CRV1": "cl"})),
    "cfelogit two-way, analytic weights": (
        "panel", lambda: __import__("logit_stream").cfelogit_stream, "y ~ x1 + x2 | i",
        dict(vcov={"CRV1": "cl+cl2"}, weights="w")),
    "cfelogit frequency weights, offset": (
        "panel", lambda: __import__("logit_stream").cfelogit_stream, "y ~ x1 + x2 | i",
        dict(vcov="hetero", weights="w", weights_type="fweights", offset="o")),
    "cfelogit many clusters": ("panel", lambda: __import__("logit_stream").cfelogit_stream,
                               "y ~ x1 + x2 | i", dict(vcov={"CRV1": "clmany"})),
}


@pytest.mark.parametrize("owner", [False, True], ids=["copies per part", "owned clusters"])
@pytest.mark.parametrize("name", list(CASE_CASES))
def test_sets_and_units_variance_equals_the_numpy_code(request, monkeypatch, name, owner):
    which, getfn, fml, kw = CASE_CASES[name]
    df, fn = request.getfixturevalue(which), getfn()
    calls = {"n": 0}
    add = MK.CasesMeat.add

    def spy(self, blk):
        calls["n"] += 1
        return add(self, blk)
    monkeypatch.setattr(MK.CasesMeat, "add", spy)
    if owner:
        monkeypatch.setattr(MK, "MEAT_CLUSTER_MB", 1e-6)
    kw = dict(verbose=False, tol=1e-13, **kw)
    fast = fn(fml, df, **kw)
    assert calls["n"] >= 1
    calls["n"] = 0
    monkeypatch.setattr(M, "MNL_KERNEL_MIN_ROWS", 10**9)
    ref = fn(fml, df, **kw)
    assert calls["n"] == 0
    np.testing.assert_allclose(fast.beta, ref.beta, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(fast.vcov, ref.vcov, rtol=1e-8, atol=1e-14)
