"""The multinomial and ordered logit with one absorbed fixed effect: the fused
passes over rows grouped by the fixed effect (fe_kernels.py) against the
passes over the rows in their own order."""

import numpy as np
import polars as pl
import pytest

import logit_stream.estimator as est
import logit_stream.fe_kernels as FK
from logit_stream import mlogit_stream, ologit_stream, oprobit_stream
from logit_stream.fe_kernels import part_starts


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(21)
    n, G, K = 60_000, 150, 4
    X = rng.normal(size=(n, K))
    g = rng.integers(0, G, n)
    a = rng.normal(size=G)
    lat = X @ rng.normal(scale=0.4, size=K) + a[g] + rng.logistic(size=n)
    E = np.column_stack([np.zeros(n)] + [X @ rng.normal(scale=0.3, size=K)
                                         + rng.normal(size=G)[g] for _ in range(3)])
    P = np.exp(E - E.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    ym = (rng.random(n)[:, None] > P.cumsum(1)).sum(1).clip(0, 3)
    # three groups whose outcomes never vary: dropped before the fit
    const = np.isin(g, [0, 1, 2])
    ym = np.where(const, g, ym)
    yo = np.where(const, 0, np.digitize(lat, [-1.5, -0.4, 0.6]))
    return pl.DataFrame({"ym": ym.astype(str), "yo": yo, "g": g,
                         "w": rng.integers(1, 4, n).astype(float),
                         "o": rng.normal(scale=0.2, size=n), "cl": rng.integers(0, 30, n),
                         **{f"x{j}": X[:, j] for j in range(K)}})


RHS = "x0 + x1 + x2 + x3"
CASES = {
    "mlogit": (mlogit_stream, f"ym ~ {RHS} | g", dict(levels=list("0123"))),
    "mlogit weighted, clustered": (mlogit_stream, f"ym ~ {RHS} | g",
                                   dict(levels=list("0123"), weights="w",
                                        weights_type="fweights", vcov={"CRV1": "cl"})),
    "ologit": (ologit_stream, f"yo ~ {RHS} | g", {}),
    "ologit weighted, clustered": (ologit_stream, f"yo ~ {RHS} | g",
                                   dict(weights="w", weights_type="fweights",
                                        vcov={"CRV1": "cl"})),
    "ologit offset": (ologit_stream, f"yo ~ {RHS} | g", dict(offset="o")),
    "oprobit": (oprobit_stream, f"yo ~ {RHS} | g", {}),
}


def _spy(monkeypatch):
    """Count the blocks the kernels sum, and check each is grouped: a
    group's rows consecutive, and a group in one block only."""
    seen = {"blocks": 0, "owner": {}}

    def wrap(cls):
        add = cls.add

        def spy(self, blk):
            f = blk["fe"]
            runs = f[np.r_[True, f[1:] != f[:-1]]]
            assert len(runs) == len(np.unique(f)), "a group's rows are not consecutive"
            block = int(blk["row"].min())               # the same block in every pass
            for gr in runs.tolist():
                assert seen["owner"].setdefault(gr, block) == block, "a group is in two blocks"
            seen["blocks"] += 1
            return add(self, blk)
        monkeypatch.setattr(cls, "add", spy)

    wrap(FK.MultinomialPass)
    wrap(FK.OrderedPass)
    return seen


@pytest.mark.parametrize("name", list(CASES))
def test_grouped_kernels_equal_the_passes_over_unsorted_rows(data, monkeypatch, name):
    fn, fml, kw = CASES[name]
    seen = _spy(monkeypatch)
    kw = dict(verbose=False, tol=1e-13, **kw)
    with pytest.warns(UserWarning, match="never varies"):
        fast = fn(fml, data, **kw)
    assert seen["blocks"] >= fast.diagnostics["newton"]["passes"]
    assert fast.diagnostics["information"]["form"] == "grouped-outer-product"
    seen["blocks"] = 0
    seen["owner"].clear()
    monkeypatch.setattr(est, "SORT_FE", False)
    with pytest.warns(UserWarning, match="never varies"):
        ref = fn(fml, data, **kw)
    assert seen["blocks"] == 0 and ref.diagnostics["information"]["form"] == "array"
    np.testing.assert_allclose(fast.beta, ref.beta, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(fast.se, ref.se, rtol=1e-7)
    assert fast.loglik == pytest.approx(ref.loglik, rel=1e-11)
    # (nan with an offset, in both)
    np.testing.assert_allclose(fast.loglik_fe_null, ref.loglik_fe_null, rtol=1e-10)


def test_a_level_too_big_for_a_block_leaves_the_rows_unsorted(data, monkeypatch):
    seen = _spy(monkeypatch)
    d = data.with_columns(g=pl.when(pl.col("x0") > -0.5).then(0).otherwise(pl.col("g") + 1))
    fit = ologit_stream(f"yo ~ {RHS} | g", d, block_mb=0.05, verbose=False)
    assert seen["blocks"] == 0 and fit.diagnostics["information"]["form"] == "array"
    assert fit.converged


def test_the_wide_designs_keep_the_array_passes(data, monkeypatch):
    seen = _spy(monkeypatch)
    monkeypatch.setattr(FK, "_parts", FK._parts)           # (the kernels are importable)
    import logit_stream.fe as fe_mod
    monkeypatch.setattr(fe_mod, "ORD_KERNEL_MAX_K", 2)
    fit = ologit_stream(f"yo ~ {RHS} | g", data, verbose=False)
    assert seen["blocks"] == 0 and fit.diagnostics["information"]["form"] == "array"


@pytest.mark.parametrize("n_parts", [1, 3, 8, 50])
def test_parts_start_at_group_boundaries(n_parts):
    rng = np.random.default_rng(0)
    f = np.repeat(np.arange(40), rng.integers(1, 30, 40))
    st = part_starts(f, n_parts)
    assert st[0] == 0 and st[-1] == len(f) and len(st) == n_parts + 1
    assert np.all(np.diff(st) >= 0)
    inner = st[1:-1]
    assert np.all((inner == len(f)) | (f[np.minimum(inner, len(f) - 1) - 1]
                                       != f[np.minimum(inner, len(f) - 1)]))
    # one group: every cut is at the end
    one = part_starts(np.zeros(100, np.int64), 4)
    assert one.tolist() == [0, 100, 100, 100, 100]


# ------------------------------------------------- the ordered fixed-effects-only null
def _table(rng, link, G, J, per, thin=False):
    from logit_stream.models import LINKS
    L = LINKS[link]
    loc, cuts = rng.normal(size=G), np.sort(rng.normal(size=J - 1))
    cdf = np.c_[np.zeros(G), L.cdf(cuts[None, :] - loc[:, None]), np.ones(G)]
    n = np.array([rng.multinomial(per, p) for p in np.diff(cdf, axis=1)]).astype(float)
    if thin:                                    # many absent levels within groups
        n = n * (rng.random(n.shape) < 0.6)
    ok = (n > 0).sum(axis=1) > 1
    return n[ok], L


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("G,J,per,thin", [(40, 2, 30, False), (60, 3, 25, False),
                                          (300, 5, 40, False), (200, 6, 12, True)])
def test_the_table_null_by_newton_equals_lbfgs(link, G, J, per, thin):
    from logit_stream.fe import _ordered_table_lbfgs, _ordered_table_newton
    n, L = _table(np.random.default_rng(G + J), link, G, J, per, thin)
    newton = _ordered_table_newton(n, L)
    assert newton is not None
    assert newton == pytest.approx(_ordered_table_lbfgs(n, L), rel=1e-9)


def test_the_table_null_falls_back_to_lbfgs_when_newton_fails(monkeypatch):
    import logit_stream.fe as fe_mod
    n, L = _table(np.random.default_rng(1), "logit", 80, 4, 30)
    expect = fe_mod._ordered_table_lbfgs(n, L)
    monkeypatch.setattr(fe_mod, "_ordered_table_newton", lambda *a, **k: None)
    assert fe_mod._ordered_table_loglik(n, L) == pytest.approx(expect, rel=1e-12)


def test_the_tall_product_equals_the_plain_one():
    from logit_stream.fe import _tall_product
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(90_000, 7)), rng.normal(size=(90_000, 5))
    np.testing.assert_allclose(_tall_product(A, B), A.T @ B, rtol=1e-10, atol=1e-9)
    np.testing.assert_allclose(_tall_product(A[:100], B[:100]), A[:100].T @ B[:100])


# ------------------------------------------------- the variances, with a fixed effect
VCASES = {
    "mlogit hetero": (mlogit_stream, f"ym ~ {RHS} | g", dict(levels=list("0123"), vcov="hetero")),
    "mlogit two-way, analytic weights": (
        mlogit_stream, f"ym ~ {RHS} | g",
        dict(levels=list("0123"), vcov={"CRV1": "cl+cl2"}, weights="w")),
    "mlogit hetero, frequency weights": (
        mlogit_stream, f"ym ~ {RHS} | g",
        dict(levels=list("0123"), vcov="hetero", weights="w", weights_type="fweights")),
    "mlogit many clusters": (mlogit_stream, f"ym ~ {RHS} | g",
                             dict(levels=list("0123"), vcov={"CRV1": "clmany"})),
    "ologit hetero": (ologit_stream, f"yo ~ {RHS} | g", dict(vcov="hetero")),
    "ologit CRV1, offset": (ologit_stream, f"yo ~ {RHS} | g",
                            dict(vcov={"CRV1": "cl"}, offset="o")),
    "ologit two-way, frequency weights": (
        ologit_stream, f"yo ~ {RHS} | g",
        dict(vcov={"CRV1": "cl+cl2"}, weights="w", weights_type="fweights")),
    "oprobit analytic weights, hetero": (oprobit_stream, f"yo ~ {RHS} | g",
                                         dict(vcov="hetero", weights="w")),
}


@pytest.fixture(scope="module")
def vdata(data):
    rng = np.random.default_rng(41)
    return data.with_columns(cl2=pl.Series(rng.integers(0, 9, data.height)),
                             clmany=pl.Series(rng.integers(0, 20_000, data.height)))


def _count_meat(monkeypatch):
    calls = {"n": 0}
    for cls in (FK.MultinomialFEMeat, FK.OrderedFEMeat):
        add = cls.add

        def spy(self, blk, add=add):
            calls["n"] += 1
            return add(self, blk)
        monkeypatch.setattr(cls, "add", spy)
    return calls


@pytest.mark.parametrize("owner", [False, True], ids=["copies per part", "owned clusters"])
@pytest.mark.parametrize("name", list(VCASES))
def test_grouped_variance_kernels_equal_the_numpy_pass(vdata, monkeypatch, name, owner):
    import logit_stream.meat_kernels as MK
    fn, fml, kw = VCASES[name]
    calls = _count_meat(monkeypatch)
    if owner:
        monkeypatch.setattr(MK, "MEAT_CLUSTER_MB", 1e-6)
    kw = dict(verbose=False, tol=1e-13, **kw)
    with pytest.warns(UserWarning, match="never varies"):
        fast = fn(fml, vdata, **kw)
    assert calls["n"] >= 1
    calls["n"] = 0
    monkeypatch.setattr(est, "SORT_FE", False)
    with pytest.warns(UserWarning, match="never varies"):
        ref = fn(fml, vdata, **kw)
    assert calls["n"] == 0
    np.testing.assert_allclose(fast.vcov, ref.vcov, rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(fast.se, ref.se, rtol=1e-8)


def test_the_variance_pass_keeps_the_numpy_code_for_many_slopes(vdata, monkeypatch):
    import logit_stream.fe as fe_mod
    calls = _count_meat(monkeypatch)
    monkeypatch.setattr(fe_mod, "FE_MEAT_MAX_P", 3)
    ologit_stream(f"yo ~ {RHS} | g", vdata, vcov="hetero", verbose=False)
    assert calls["n"] == 0


def test_rows_that_cannot_be_sorted_inside_the_memory_limit_stay_unsorted(data, monkeypatch):
    """Sorting ungrouped rows partitions them in memory; with too little room
    the rows stay in their order (the same estimates), but rows already
    grouped by the fixed effect are sorted whatever the room."""
    seen = _spy(monkeypatch)
    sorted_fit = ologit_stream(f"yo ~ {RHS} | g", data, verbose=False)
    assert seen["blocks"] > 0
    seen["blocks"] = 0
    monkeypatch.setattr("logit_stream.store._partition_gb", lambda path, rows: 1e6)
    tight = ologit_stream(f"yo ~ {RHS} | g", data, verbose=False)
    assert seen["blocks"] == 0
    assert np.allclose(tight.beta, sorted_fit.beta, rtol=0, atol=1e-8)
    seen["blocks"] = 0
    ologit_stream(f"yo ~ {RHS} | g", data.sort("g"), verbose=False)
    assert seen["blocks"] > 0


@pytest.mark.parametrize("name", ["mlogit", "ologit"])
def test_weights_and_clusters_count_the_rows_used(data, name):
    """The levels of g whose outcome never varies are dropped; the fit must
    equal the one on the data without them (analytic weights average 1, and
    clusters are counted, over the rows used)."""
    fn, fml, kw = CASES[name]
    kw = {k: v for k, v in kw.items() if k not in ("weights", "offset")}
    kept = data.filter(~pl.col("g").is_in([0, 1, 2]))
    for extra in ({"weights": "w"}, {"vcov": {"CRV1": "g"}},
                  {"weights": "w", "vcov": {"CRV1": "g"}}):
        a = fn(fml, data, verbose=False, **kw, **extra)
        b = fn(fml, kept, verbose=False, **kw, **extra)
        assert np.allclose(a.beta, b.beta, rtol=0, atol=1e-8)
        assert a.n_obs == b.n_obs
        for k in b.all_vcovs:
            assert np.allclose(a.all_vcovs[k][0], b.all_vcovs[k][0], rtol=1e-6, atol=1e-10), k
            assert a.all_vcovs[k][1] == b.all_vcovs[k][1], k
        for stat in ("loglik", "loglik_null", "aic", "bic"):
            assert np.isclose(getattr(a, stat), getattr(b, stat), rtol=1e-8), stat


@pytest.mark.parametrize("rows", [7, 100, 10_000])
def test_grouped_counts_runs_across_batches(tmp_path, rows):
    from logit_stream.store import _grouped
    rng = np.random.default_rng(3)
    g = np.sort(rng.integers(0, 40, 500))
    path = tmp_path / "d.parquet"
    pl.DataFrame({"a": g, "b": g % 3, "s": g.astype(str)}).write_parquet(path)
    n = len(np.unique(g))
    assert _grouped(path, ["a"], n, rows) and _grouped(path, ["s"], n, rows)
    shuffled = tmp_path / "s.parquet"
    pl.DataFrame({"a": rng.permutation(g)}).write_parquet(shuffled)
    assert not _grouped(shuffled, ["a"], n, rows)
