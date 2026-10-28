"""``device="cuda"`` without a GPU: the device code (logit_stream/device/)
run on a numpy stand-in for CuPy, as hdfe-stream's GPU tests do. Everything
but CuPy itself is exercised: the per-block sums of each model, the blocks
kept on the device or copied each pass, the column selection after a
collinear column is dropped, and how the fit reaches them. On a GPU the sums
are formed in a different order, so the agreement there is to the tolerance
(benchmarks/gpu_benchmark.py checks it on the hardware)."""

from types import SimpleNamespace

import hdfe_stream.core.gpu as gpu
import numpy as np
import polars as pl
import pytest

from logit_stream import clogit_stream, mlogit_stream, ologit_stream
from logit_stream import device as dev


@pytest.fixture
def numpy_device(monkeypatch):
    backend = SimpleNamespace(xp=np, asnumpy=np.asarray, name="numpy stand-in")
    monkeypatch.setattr(gpu, "cuda_backend", lambda: backend)
    return backend


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    n = 4000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    g = rng.integers(0, 6, n)
    E = np.column_stack([np.zeros(n), 0.3 + 0.8 * x1 + 0.2 * g, -0.4 + 0.5 * x2 - 0.1 * g])
    P = np.exp(E)
    P /= P.sum(1, keepdims=True)
    y = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    yo = np.searchsorted([-0.5, 0.4, 1.2], 0.7 * x1 - 0.3 * x2 + rng.logistic(size=n))
    return pl.DataFrame({"y": y, "yo": yo, "x1": x1, "x2": x2, "x3": 2 * x1, "g": g,
                         "w": rng.uniform(0.5, 2, n), "o": 0.2 * x2})


def _same(a, b):
    np.testing.assert_allclose(a.beta, b.beta, atol=1e-9)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-8)
    assert a.loglik == pytest.approx(b.loglik, abs=1e-8)


KW = dict(verbose=False, tol=1e-14, block_mb=0.05)        # several blocks


def test_multinomial(data, numpy_device):
    for extra in ({}, {"weights": "w"}, {"weights": "w", "weights_type": "fweights"}):
        cpu = mlogit_stream("y ~ x1 + x2 + C(g)", data, vcov="hetero", **KW, **extra)
        gpu_ = mlogit_stream("y ~ x1 + x2 + C(g)", data, vcov="hetero", device="cuda", **KW,
                             **extra)
        _same(gpu_, cpu)
    assert gpu_.diagnostics["device"]["name"] == "numpy stand-in"
    assert gpu_.diagnostics["device"]["resident"]


def test_binary_columns_stay_compact(data, numpy_device):
    """The 0/1 columns go over as bytes and stay so on the device; the
    estimates are as with them stored whole."""
    import logit_stream.store as st
    g = mlogit_stream("y ~ x1 + x2 + C(g)", data, device="cuda", **KW)
    n, K = data.height, 8                       # intercept, x1, x2, five indicators
    assert g.diagnostics["device"]["bytes"] < 0.6 * n * K * 8
    try:
        st.COMPACT_BINARY = False
        h = mlogit_stream("y ~ x1 + x2 + C(g)", data, device="cuda", **KW)
    finally:
        st.COMPACT_BINARY = True
    assert h.diagnostics["device"]["bytes"] > n * K * 8
    _same(g, h)


def test_collinear_column_dropped(data, numpy_device):
    with pytest.warns(UserWarning, match="collinear"):
        g = mlogit_stream("y ~ x1 + x2 + x3", data, device="cuda", **KW)
    _same(g, mlogit_stream("y ~ x1 + x2", data, **KW))


def test_ordered_with_offset(data, numpy_device):
    cpu = ologit_stream("yo ~ x1 + x2", data, offset="o", weights="w", vcov="hetero", **KW)
    _same(ologit_stream("yo ~ x1 + x2", data, offset="o", weights="w", vcov="hetero",
                        device="cuda", **KW), cpu)


def test_conditional(choice_data, numpy_device):
    fml = "chosen ~ cost + time + C(alt) + income:C(alt)"
    c = choice_data.with_columns(w=(pl.col("case") % 3 + 1).cast(pl.Float64))
    for extra in ({}, {"weights": "w"}):
        cpu = clogit_stream(fml, c, case="case", **KW, **extra)
        _same(clogit_stream(fml, c, case="case", device="cuda", **KW, **extra), cpu)


def test_blocks_copied_each_pass(data, numpy_device, monkeypatch):
    """A sample larger than the device's budget is copied over each pass."""
    orig = dev.DeviceStore.__init__
    monkeypatch.setattr(dev.DeviceStore, "__init__",
                        lambda self, store, be, budget=None: orig(self, store, be, budget=1))
    g = mlogit_stream("y ~ x1 + x2", data, device="cuda", **KW)
    assert not g.diagnostics["device"]["resident"]
    _same(g, mlogit_stream("y ~ x1 + x2", data, **KW))


def test_firth(data, numpy_device):
    for extra in ({}, {"weights": "w"}):
        cpu = mlogit_stream("y ~ x1 + x2 + C(g)", data, firth=True, **KW, **extra)
        _same(mlogit_stream("y ~ x1 + x2 + C(g)", data, firth=True, device="cuda", **KW,
                            **extra), cpu)


def _panel(N=600, seed=5):
    rng = np.random.default_rng(seed)
    T = rng.integers(2, 9, N)
    i = np.repeat(np.arange(N), T)
    a = rng.normal(0, 1.5, N)[i]
    x1, x2 = rng.normal(size=len(i)) + 0.5 * a, rng.normal(size=len(i))
    y = (rng.random(len(i)) < 1 / (1 + np.exp(-(0.8 * x1 - 0.5 * x2 + a)))).astype(int)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "i": i,
                         "w": rng.uniform(0.5, 2, N)[i], "o": 0.1 * x1})


def test_conditional_fixed_effects(numpy_device, monkeypatch):
    """Units of every length, some fitted as their complement, in batches of
    a few units (the batch size forced down) and in several blocks."""
    from logit_stream import cfelogit_stream
    panel = _panel()
    monkeypatch.setattr(dev.models, "CFE_CHUNK_BYTES", 20_000)
    for extra in ({}, {"weights": "w"}, {"offset": "o", "vcov": {"CRV1": "i"}}):
        cpu = cfelogit_stream("y ~ x1 + x2 | i", panel, **KW, **extra)
        g = cfelogit_stream("y ~ x1 + x2 | i", panel, device="cuda", **KW, **extra)
        _same(g, cpu)
        assert g.diagnostics["device"]["name"] == "numpy stand-in"
    assert g.diagnostics["store"]["blocks"] > 1


def test_restrictions(data, numpy_device):
    with pytest.raises(ValueError, match="device must be"):
        mlogit_stream("y ~ x1", data, device="tpu", verbose=False)


def test_without_cupy_says_what_to_install(data, monkeypatch):
    def missing():
        raise ImportError(gpu._CUPY_HINT)
    monkeypatch.setattr(gpu, "cuda_backend", missing)
    with pytest.raises(ImportError, match="CuPy"):
        mlogit_stream("y ~ x1", data, device="cuda", verbose=False)


@pytest.mark.parametrize("kind", ["mlogit", "ologit"])
def test_absorbed_fixed_effect(data, numpy_device, kind):
    """The fixed-effect passes' per-level sums on the device (scatter-adds),
    with weights, and for the ordered model an offset."""
    fn, fml = ((mlogit_stream, "y ~ x1 + x2 | g") if kind == "mlogit"
               else (ologit_stream, "yo ~ x1 + x2 | g"))
    extra = {"weights": "w"} if kind == "mlogit" else {"offset": "o"}
    cpu = fn(fml, data, vcov="hetero", **KW, **extra)
    gpu_ = fn(fml, data, vcov="hetero", device="cuda", **KW, **extra)
    _same(gpu_, cpu)
    np.testing.assert_allclose(
        gpu_.fixef().sort("g").collect().drop("g").to_numpy().astype(float),
        cpu.fixef().sort("g").collect().drop("g").to_numpy().astype(float), atol=1e-8)


def test_fixed_effect_with_dropped_levels(numpy_device):
    """Levels whose outcome never varies are dropped; their rows masked on
    the device."""
    rng = np.random.default_rng(4)
    n = 2000
    g = rng.integers(0, 40, n)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(x + rng.normal(0, 1, 40)[g])))).astype(int)
    y[g < 3] = 0                                        # three levels without variation
    df = pl.DataFrame({"y": y, "x": x, "g": g})
    with pytest.warns(UserWarning, match="never varies"):
        cpu = mlogit_stream("y ~ x | g", df, **KW)
    with pytest.warns(UserWarning, match="never varies"):
        gpu_ = mlogit_stream("y ~ x | g", df, device="cuda", **KW)
    _same(gpu_, cpu)


def _re_data(seed=6):
    rng = np.random.default_rng(seed)
    n = 3000
    s = rng.integers(0, 8, n)
    p = s * 5 + rng.integers(0, 5, n)                     # PUMAs nested in states
    o = rng.integers(0, 6, n)                              # crossed with both
    x = rng.normal(size=n)
    # effects of every grouping in both equations, so that no Sigma is at the
    # boundary (where the stopping point is sensitive to the order of sums)
    st, oc = rng.normal(0, 0.5, (8, 2))[s], rng.normal(0, 0.5, (6, 2))[o]
    E = np.column_stack([np.zeros(n), 0.4 * x + rng.normal(0, 0.5, 40)[p] + st[:, 0] + oc[:, 0],
                         -0.3 * x + rng.normal(0, 0.4, 40)[p] + st[:, 1] + oc[:, 1]])
    P = np.exp(E)
    P /= P.sum(1, keepdims=True)
    y = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    yo = np.searchsorted([-0.4, 0.6], 0.6 * x + rng.normal(0, 0.6, 40)[p] + rng.logistic(size=n))
    return pl.DataFrame({"y": y, "yo": yo, "x": x, "s": s, "p": p, "o": o,
                         "w": rng.uniform(0.5, 2, n), "off": 0.1 * x})


@pytest.mark.parametrize("fn,fml,kw", [
    (mlogit_stream, "y ~ x", {"random": "p"}),
    (mlogit_stream, "y ~ x", {"random": "p", "random_method": "joint-mode", "weights": "w"}),
    (mlogit_stream, "y ~ x", {"random": ["s", "p"]}),
    (mlogit_stream, "y ~ x", {"random": ["p", "o"]}),
    (ologit_stream, "yo ~ x", {"random": "p", "offset": "off"}),
    (ologit_stream, "yo ~ x | s", {"random": "p"}),
], ids=["mnl", "mnl-joint-weights", "mnl-nested", "mnl-crossed", "ord-offset",
        "ord-within-fe"])
def test_random_intercepts(numpy_device, fn, fml, kw):
    """The random intercepts' data passes on the device: the unit sums, the
    cross blocks (per unit, or per group when crossed) and the exact
    Laplace terms, by scatter-adds."""
    df = _re_data()
    cpu = fn(fml, df, verbose=False, tol=1e-12, random_tol=1e-8, block_mb=0.05, **kw)
    gpu_ = fn(fml, df, verbose=False, tol=1e-12, random_tol=1e-8, block_mb=0.05,
              device="cuda", **kw)
    np.testing.assert_allclose(gpu_.beta, cpu.beta, atol=1e-7)
    np.testing.assert_allclose(gpu_.se, cpu.se, rtol=1e-6)
    for g, info in cpu.random_effects.items():
        np.testing.assert_allclose(gpu_.random_effects[g]["Sigma"], info["Sigma"], atol=1e-7)
    assert gpu_.loglik == pytest.approx(cpu.loglik, abs=1e-6)


@pytest.mark.parametrize("extra", [{}, {"halton": False, "seed": 3},
                                   {"draws_cache_mb": 0, "block_mb": 0.02}])
def test_mixed(numpy_device, monkeypatch, extra):
    """The vectorized pass (rows x draws arrays) against the numba kernel:
    Halton and pseudo-random draws generated on the device, kept or
    regenerated each pass, choosers in small batches, several blocks."""
    import test_mixed as T

    from logit_stream import mixlogit_stream
    monkeypatch.setattr(dev.mixed, "MXL_CHUNK_ELEMS", 3000)
    df = T._panel_data()
    kw = dict(case="situation", panel="person", random={"x1": "normal", "x2": "lognormal"},
              draws=60, verbose=False, vcov={"CRV1": "person"}, **extra)
    a = mixlogit_stream("chosen ~ x1 + x2 + x3 - 1", df, **kw)
    b = mixlogit_stream("chosen ~ x1 + x2 + x3 - 1", df, device="cuda", **kw)
    np.testing.assert_allclose(b.beta, a.beta, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(b.se, a.se, rtol=1e-6)
    assert b.loglik == pytest.approx(a.loglik, abs=1e-8)
    d = b.diagnostics["device"]
    assert d["name"] == "numpy stand-in"
    assert d["draws_cached"] == (extra.get("draws_cache_mb", 1) > 0)


@pytest.mark.parametrize("extra", [{}, {"weights": "aw", "offset": "o"},
                                   {"three": True}])
def test_nested(numpy_device, monkeypatch, extra):
    """The vectorized pass (dense sets x nodes arrays) against the numba
    kernel: two and three levels, weights and an offset, sets in small
    batches and several blocks."""
    import test_nested as T

    from logit_stream import nlogit_stream
    monkeypatch.setattr(dev.nested, "NL_CHUNK_BYTES", 30_000)
    d = T._simulate(1500).with_columns(aw=(pl.col("case") % 4 + 1).cast(pl.Float64),
                                       o=0.1 * pl.col("time"))
    nests = T.THREE if extra.pop("three", False) else T.NESTS
    kw = dict(case="case", alt="alt", nests=nests, verbose=False, tol=1e-12, block_mb=0.05,
              vcov="hetero", **extra)
    a = nlogit_stream("chosen ~ cost + time + C(alt)", d, **kw)
    b = nlogit_stream("chosen ~ cost + time + C(alt)", d, device="cuda", **kw)
    assert b.diagnostics["device"]["name"] == "numpy stand-in"
    assert b.diagnostics["store"]["blocks"] > 1
    np.testing.assert_allclose(b.beta, a.beta, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(b.se, a.se, rtol=1e-5)
    assert b.loglik == pytest.approx(a.loglik, abs=1e-8)


def _refuse(*args, **kwargs):
    raise AssertionError("the scores pass ran on the CPU")


ROBUST = [
    ("mnl", lambda d: d["data"], mlogit_stream, "y ~ x1 + x2 + C(g)",
     {"vcov": "hetero", "cluster": ["g", "g+o"], "weights": "w"}),
    ("ord", lambda d: d["data"], ologit_stream, "yo ~ x1 + x2",
     {"vcov": "hetero", "cluster": "g", "offset": "o",
      "weights": "w", "weights_type": "fweights"}),
    ("mnl-fe", lambda d: d["fe"], mlogit_stream, "y ~ x1 + x2 | g",
     {"vcov": "hetero", "cluster": ["cl", "g"]}),
    ("ord-fe", lambda d: d["fe"], ologit_stream, "yo ~ x1 + x2 | g",
     {"vcov": "hetero", "cluster": "g+cl"}),
    ("mnl-re", lambda d: d["re"], mlogit_stream, "y ~ x",
     {"random": "p", "vcov": {"CRV1": "p"}}),
    ("ord-re-fe", lambda d: d["re"], ologit_stream, "yo ~ x | s",
     {"random": "p", "vcov": {"CRV1": "s"}}),
    ("clogit", lambda d: d["choice"], clogit_stream, "chosen ~ cost + time",
     {"case": "case", "vcov": "hetero", "cluster": "region"}),
]


@pytest.mark.parametrize("name,get,fn,fml,kw", ROBUST, ids=[r[0] for r in ROBUST])
def test_robust_variances_on_the_device(data, fe_data, choice_data, numpy_device, monkeypatch,
                                        name, get, fn, fml, kw):
    """The robust and clustered variances' scores pass on the device: the
    same vcovs as the CPU's (one-way, multi-way, by row), with the CPU's
    accumulation of the scores made to fail under device="cuda"."""
    from logit_stream.newton import _Driver
    d = {"data": data.with_columns(o=pl.col("g") % 3), "fe": fe_data, "re": _re_data(),
         "choice": choice_data.with_columns(region=pl.col("case") % 17)}
    df = get(d)
    common = dict(verbose=False, tol=1e-12, block_mb=0.05)
    if "random" in kw:
        common["random_tol"] = 1e-8
    cpu = fn(fml, df, **common, **kw)
    orig = _Driver._add_meat
    monkeypatch.setattr(_Driver, "_add_meat", staticmethod(_refuse))
    gpu_ = fn(fml, df, device="cuda", **common, **kw)
    monkeypatch.setattr(_Driver, "_add_meat", staticmethod(orig))
    assert set(gpu_.all_vcovs) == set(cpu.all_vcovs)
    for k, (V, df_) in cpu.all_vcovs.items():
        np.testing.assert_allclose(gpu_.all_vcovs[k][0], V, rtol=1e-6, atol=1e-12, err_msg=k)
        assert gpu_.all_vcovs[k][1] == df_


def test_robust_variances_on_the_device_nested_and_cfe(numpy_device, monkeypatch):
    import test_nested as T

    from logit_stream import cfelogit_stream, nlogit_stream
    from logit_stream.newton import _Driver
    orig = _Driver._add_meat
    monkeypatch.setattr(_Driver, "_add_meat", staticmethod(_refuse))
    nd = T._simulate(1500).with_columns(region=pl.col("case") % 23, aw=pl.col("case") % 3 + 1.0)
    kw = dict(case="case", alt="alt", nests=T.NESTS, verbose=False, tol=1e-12, block_mb=0.05,
              vcov="hetero", cluster="region", weights="aw")
    pd_ = _panel().with_columns(c=pl.col("i") % 31)
    ckw = dict(verbose=False, tol=1e-12, block_mb=0.05, vcov={"CRV1": "c"}, cluster="i")
    gpu_n = nlogit_stream("chosen ~ cost + time + C(alt)", nd, device="cuda", **kw)
    gpu_c = cfelogit_stream("y ~ x1 + x2 | i", pd_, device="cuda", **ckw)
    monkeypatch.setattr(_Driver, "_add_meat", staticmethod(orig))
    cpu_n = nlogit_stream("chosen ~ cost + time + C(alt)", nd, **kw)
    cpu_c = cfelogit_stream("y ~ x1 + x2 | i", pd_, **ckw)
    for g, c in ((gpu_n, cpu_n), (gpu_c, cpu_c)):
        for k, (V, _) in c.all_vcovs.items():
            np.testing.assert_allclose(g.all_vcovs[k][0], V, rtol=1e-5, atol=1e-12, err_msg=k)
