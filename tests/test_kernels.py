"""The likelihood kernels: alternative code paths agree, and the gradient
and information are the derivatives of the log-likelihood."""

import numpy as np
import pytest

from logit_stream import mlogit_stream
from logit_stream.models import _Conditional, _Multinomial, _Ordered


def _indicator_design(rng, n, K):
    return np.column_stack([np.ones(n), rng.normal(size=n)]
                           + [(rng.random(n) < 0.2).astype(float) for _ in range(K - 2)])


@pytest.mark.parametrize("weighted", [False, True])
def test_sparse_and_dense_information_agree(weighted):
    rng = np.random.default_rng(0)
    n, K, J = 4000, 12, 5
    X = _indicator_design(rng, n, K)
    y = rng.integers(0, J, n)
    w = rng.random(n) + 0.5 if weighted else None
    m = _Multinomial([f"x{i}" for i in range(K)], K, list(range(J)), 0)
    th = rng.normal(size=K * (J - 1)) * 0.3
    m.density = None
    dense = m.accumulate(th, X, y, w, None, True)
    m.density = 0.1
    assert m.sparse(X)
    sparse = m.accumulate(th, X, y, w, None, True)
    np.testing.assert_allclose(sparse[2], dense[2], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(sparse[1], dense[1])
    assert sparse[0] == dense[0]


def test_fit_uses_the_sparse_path_and_agrees(mnl_data):
    a = mlogit_stream("y ~ x1 + x2 + C(g)", mnl_data, verbose=False, tol=1e-14)
    import logit_stream.models as M
    old = M.SPARSE_DENSITY
    try:
        M.SPARSE_DENSITY = -1.0              # force the dense products
        b = mlogit_stream("y ~ x1 + x2 + C(g)", mnl_data, verbose=False, tol=1e-14)
    finally:
        M.SPARSE_DENSITY = old
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-10)


@pytest.mark.parametrize("weights", [None, "x2w"])
def test_fixed_effects_sparse_and_dense_agree(fe_data, weights):
    import polars as pl

    import logit_stream.models as M
    df = fe_data.with_columns(x2w=pl.col("x2").abs() + 0.5)
    fits = []
    old = M.SPARSE_DENSITY
    try:
        for d in (2.0, -1.0):               # always the sparse kernel, then never
            M.SPARSE_DENSITY = d
            fits.append(mlogit_stream("y ~ x1 + x2 | g", df, weights=weights,
                                      vcov={"CRV1": "cl"}, verbose=False, tol=1e-14))
    finally:
        M.SPARSE_DENSITY = old
    np.testing.assert_allclose(fits[0].beta, fits[1].beta, rtol=1e-10)
    np.testing.assert_allclose(fits[0].se, fits[1].se, rtol=1e-10)


def _numeric_check(model, th, X, y, w, starts, eps=1e-6):
    ll, g, H = model.accumulate(th, X, y, w, starts, True)
    rng = np.random.default_rng(1)
    for _ in range(3):
        v = rng.normal(size=len(th)) * 0.1
        lp = model.accumulate(th + eps * v, X, y, w, starts, True)
        lm = model.accumulate(th - eps * v, X, y, w, starts, True)
        assert (lp[0] - lm[0]) / (2 * eps) == pytest.approx(g @ v, rel=1e-5, abs=1e-6)
        np.testing.assert_allclose((lp[1] - lm[1]) / (2 * eps), -H @ v, rtol=1e-4, atol=1e-5)


def test_derivatives_multinomial():
    rng = np.random.default_rng(2)
    X = _indicator_design(rng, 500, 5)
    m = _Multinomial(list("abcde"), 5, [0, 1, 2], 0)
    _numeric_check(m, rng.normal(size=10) * 0.3, X, rng.integers(0, 3, 500),
                   rng.random(500) + 0.5, None)


def test_derivatives_ordered():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 3))
    m = _Ordered(list("abc"), 3, [0, 1, 2, 3])
    th = np.r_[rng.normal(size=3) * 0.3, -1.0, 0.2, 1.1]
    _numeric_check(m, th, X, rng.integers(0, 4, 500), rng.random(500) + 0.5, None)
    m.fix_first = True
    _numeric_check(m, np.r_[th[:3], 1.2, 2.1], X, rng.integers(0, 4, 500), None, None)


def test_derivatives_conditional():
    rng = np.random.default_rng(4)
    sizes = rng.integers(2, 5, 200)
    starts = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    n = starts[-1]
    X = rng.normal(size=(n, 3))
    y = np.zeros(n, np.int8)
    y[starts[:-1] + rng.integers(0, 2, 200)] = 1
    w = np.repeat(rng.random(200) + 0.5, sizes)
    _numeric_check(_Conditional(list("abc"), 3), rng.normal(size=3) * 0.3, X, y, w, starts)


# ---------------------------------------------------------------- the fused pass
def _mnl_model(rng, n, K, J, own=None, density=None):
    X = _indicator_design(rng, n, K)
    if own:                                  # the own columns come last, continuous
        X[:, K - len(own):] = rng.normal(size=(n, len(own))) * 3
    y = rng.integers(0, J, n)
    m = _Multinomial([f"x{i}" for i in range(K)], K, list(range(J)), 0,
                     own={K - len(own) + i: e for i, e in enumerate(own)} if own else None)
    m.density = density
    th = rng.normal(size=m.P) * 0.3
    return m, X, y, th


@pytest.mark.parametrize("density", [None, 0.1], ids=["outer-product", "nonzeros"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("own", [None, [0, 1, 2]], ids=["plain", "own"])
@pytest.mark.parametrize("hessian", [True, False])
def test_fused_pass_equals_the_array_code(monkeypatch, density, weighted, own, hessian):
    import logit_stream.models as M
    rng = np.random.default_rng(5)
    n, K, J = 6000, 12, 4
    m, X, y, th = _mnl_model(rng, n, K, J, own, density)
    w = rng.random(n) + 0.5 if weighted else None
    assert m.kernel_ok(n)
    fused = m.accumulate(th, X, y, w, None, hessian)
    monkeypatch.setattr(M, "MNL_KERNEL_MIN_ROWS", 10**9)         # the array code
    assert not m.kernel_ok(n)
    ref = m.accumulate(th, X, y, w, None, hessian)
    assert fused[0] == pytest.approx(ref[0], rel=1e-12)
    np.testing.assert_allclose(fused[1], ref[1], rtol=1e-10, atol=1e-8)
    if hessian:
        np.testing.assert_allclose(fused[2], ref[2], rtol=1e-10, atol=1e-7)
    else:
        assert fused[2] is None


def test_fused_pass_reads_a_blocks_columns_in_place():
    """`accumulate_full` with `cols` a subset (collinear columns dropped): the
    same as `accumulate` on the subset's own array."""
    rng = np.random.default_rng(6)
    m, X, y, th = _mnl_model(rng, 5000, 8, 4)
    cols = np.array([0, 1, 3, 4, 6, 7])
    m.cols = cols
    m._free_cache = None
    th = rng.normal(size=m.P) * 0.3
    a = m.accumulate_full(th, X, cols, y, None, None, True)
    b = m.accumulate(th, np.ascontiguousarray(X[:, cols]), y, None, None, True)
    assert a[0] == pytest.approx(b[0], rel=1e-12)
    np.testing.assert_allclose(a[2], b[2], rtol=1e-12, atol=1e-9)


def test_small_blocks_and_very_wide_dense_designs_use_the_array_code():
    rng = np.random.default_rng(7)
    m, X, y, th = _mnl_model(rng, 100, 6, 4)
    assert not m.kernel_ok(100)
    wide = _Multinomial([f"x{i}" for i in range(130)], 130, [0, 1, 2], 0)
    wide.density = None
    assert not wide.kernel_ok(5000)                  # 130 dense columns
    wide.density = 0.1
    assert wide.nonzero_form() and wide.kernel_ok(5000)
    mid = _Multinomial([f"x{i}" for i in range(80)], 80, [0, 1, 2], 0)
    mid.density = 0.5
    assert not mid.nonzero_form() and mid.kernel_ok(5000)       # the outer-product form


@pytest.mark.parametrize("weights", [None, "w"])
def test_fit_agrees_with_the_array_code(mnl_data, monkeypatch, weights):
    import logit_stream.models as M
    kw = dict(weights=weights, weights_type="fweights", vcov={"CRV1": "cl"}, verbose=False,
              tol=1e-14)
    a = mlogit_stream("y ~ x1 + x2 + C(g)", mnl_data, **kw)
    monkeypatch.setattr(M, "MNL_KERNEL_MIN_ROWS", 10**9)
    b = mlogit_stream("y ~ x1 + x2 + C(g)", mnl_data, **kw)
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-9)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-9)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)


def test_diagnostics_report_the_information_form(mnl_data, monkeypatch):
    import logit_stream.models as M
    fml = "y ~ x1 + x2 + C(g)"
    fit = mlogit_stream(fml, mnl_data, verbose=False)
    info = fit.diagnostics["information"]
    assert info["form"] == "outer-product" and 0.5 < info["density"] <= 1
    monkeypatch.setattr(M, "MNL_NNZ_DENSITY", 1.01)
    assert mlogit_stream(fml, mnl_data, verbose=False).diagnostics["information"]["form"] \
        == "nonzero"
    monkeypatch.setattr(M, "MNL_KERNEL_MIN_ROWS", 10**9)                 # the array code
    assert mlogit_stream(fml, mnl_data, verbose=False).diagnostics["information"]["form"] \
        == "array"
    monkeypatch.setattr(M, "SPARSE_DENSITY", 1.01)                       # with its numba kernel
    assert mlogit_stream(fml, mnl_data, verbose=False).diagnostics["information"]["form"] \
        == "array-nonzero"
    # an absorbed fixed effect has its own passes: no entry
    assert "information" not in mlogit_stream("y ~ x1 | g", mnl_data,
                                              verbose=False).diagnostics


# ------------------------------------------------- the ordered models' fused pass
def _ord_model(rng, n, K, link, density):
    X = rng.normal(size=(n, K)) * (rng.random((n, K)) < (density if density < 1 else 1.1))
    y = np.digitize(X[:, 0] * 0.8 + rng.normal(size=n), [-1.2, -0.3, 0.5, 1.3])
    m = _Ordered([f"x{i}" for i in range(K)], K, [0, 1, 2, 3, 4], link=link)
    m.density = float(np.mean(X != 0))
    th = np.r_[rng.normal(scale=0.2, size=K), -1.3, -0.4, 0.45, 1.2]
    return m, X, y, th


@pytest.mark.parametrize("link", ["logit", "probit"])
@pytest.mark.parametrize("density", [1.0, 0.15], ids=["outer-product", "nonzeros"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("hessian", [True, False])
def test_ordered_fused_pass_equals_the_array_code(monkeypatch, link, density, weighted, offset,
                                                  hessian):
    import logit_stream.models as M
    rng = np.random.default_rng(11)
    n, K = 6000, 9
    m, X, y, th = _ord_model(rng, n, K, link, density)
    w = rng.random(n) + 0.5 if weighted else None
    off = rng.normal(scale=0.3, size=n) if offset else None
    assert m.kernel_ok(n) and m.nonzero_form() == (density < 1)
    fused = m.accumulate(th, X, y, w, None, hessian, offset=off)
    monkeypatch.setattr(M, "MNL_KERNEL_MIN_ROWS", 10**9)
    ref = m.accumulate(th, X, y, w, None, hessian, offset=off)
    assert fused[0] == pytest.approx(ref[0], rel=1e-12)
    np.testing.assert_allclose(fused[1], ref[1], rtol=1e-10, atol=1e-8)
    if hessian:
        np.testing.assert_allclose(fused[2], ref[2], rtol=1e-10, atol=1e-7)
    else:
        assert fused[2] is None


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_ordered_fused_information_is_the_hessian(link):
    rng = np.random.default_rng(12)
    m, X, y, th = _ord_model(rng, 5000, 5, link, 1.0)
    ll, g, H = m.accumulate(th, X, y, None, None, True)
    v = rng.normal(size=len(th)) * 0.1
    eps = 1e-6
    gp = m.accumulate(th + eps * v, X, y, None, None, False)[1]
    gm = m.accumulate(th - eps * v, X, y, None, None, False)[1]
    np.testing.assert_allclose((gp - gm) / (2 * eps), -H @ v, rtol=1e-5, atol=1e-5)


def test_ordered_fused_pass_reads_a_blocks_columns_in_place():
    rng = np.random.default_rng(13)
    m, X, y, th = _ord_model(rng, 5000, 8, "logit", 1.0)
    cols = np.array([0, 2, 3, 6, 7])
    m.cols = cols
    th = np.r_[rng.normal(scale=0.2, size=len(cols)), -1.3, -0.4, 0.45, 1.2]
    a = m.accumulate_full(th, X, cols, y, None, None, True)
    b = m.accumulate(th, np.ascontiguousarray(X[:, cols]), y, None, None, True)
    assert a[0] == pytest.approx(b[0], rel=1e-12)
    np.testing.assert_allclose(a[2], b[2], rtol=1e-12, atol=1e-9)


def test_ordered_fallbacks():
    rng = np.random.default_rng(14)
    m, X, y, th = _ord_model(rng, 100, 4, "logit", 1.0)
    assert not m.kernel_ok(100)                                   # few rows
    wide = _Ordered([f"x{i}" for i in range(130)], 130, [0, 1, 2], link="logit")
    wide.density = 1.0
    assert not wide.kernel_ok(5000)                               # dense and wide
    wide.density = 0.1
    assert wide.kernel_ok(5000) and wide.information_form(5000) == "nonzero"
    fe = _Ordered(["a", "b"], 2, [0, 1, 2], link="logit")
    fe.density, fe.fix_first = 1.0, True
    assert not fe.kernel_ok(5000)                                 # an absorbed fixed effect


@pytest.mark.parametrize("fit_fn", ["ologit_stream", "oprobit_stream"])
@pytest.mark.parametrize("weights", [None, "w"])
def test_ordered_fit_agrees_with_the_array_code(ord_data, monkeypatch, fit_fn, weights):
    import logit_stream
    import logit_stream.models as M
    fit = getattr(logit_stream, fit_fn)
    kw = dict(weights=weights, weights_type="fweights", vcov={"CRV1": "cl"}, verbose=False,
              tol=1e-14)
    a = fit("y ~ x1 + x2", ord_data, **kw)
    assert a.diagnostics["information"]["form"] == "outer-product"
    monkeypatch.setattr(M, "MNL_KERNEL_MIN_ROWS", 10**9)
    b = fit("y ~ x1 + x2", ord_data, **kw)
    assert b.diagnostics["information"]["form"] == "array"
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-9)
    np.testing.assert_allclose(a.se, b.se, rtol=1e-9)
    assert a.loglik == pytest.approx(b.loglik, rel=1e-12)
