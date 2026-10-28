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
