"""The mixed logit (prototype): the kernel's derivatives and log-likelihood,
the draws, invariance to the blocks and the row order, recovery of the
simulated parameters, and agreement with the exact likelihood when one
coefficient is random (one-dimensional quadrature)."""

import numpy as np
import polars as pl
import pytest
import scipy.special as sp

from logit_stream import mixlogit_stream
from logit_stream.mixed import PRIMES, _mxl_kernel, _radical_inverse, draws

NO_CACHE = np.empty((0, 1, 1))


def _panel_data(seed=0, N=600, T=4, J=3, mu1=1.0, s1=0.8, mu2=-0.6931471805599453, s2=0.4,
                b3=-0.7):
    """Choosers with T choices among J alternatives: x1's coefficient
    normal, x2's lognormal, x3's fixed."""
    rng = np.random.default_rng(seed)
    b1 = mu1 + s1 * rng.standard_normal(N)
    b2 = np.exp(mu2 + s2 * rng.standard_normal(N))
    X = rng.standard_normal((N, T, J, 3))
    V = b1[:, None, None] * X[..., 0] + b2[:, None, None] * X[..., 1] + b3 * X[..., 2]
    ch = np.argmax(V + rng.gumbel(size=V.shape), axis=2)
    person = np.repeat(np.arange(N), T * J)
    situation = np.repeat(np.arange(N * T), J)
    chosen = (np.tile(np.arange(J), N * T) == np.repeat(ch.ravel(), J)).astype(np.int8)
    Xf = X.reshape(-1, 3)
    return pl.DataFrame({"person": person, "situation": situation, "chosen": chosen,
                         "x1": Xf[:, 0], "x2": Xf[:, 1], "x3": Xf[:, 2]})


def _block(df):
    X = df.select("x1", "x2", "x3").to_numpy()
    y = df["chosen"].to_numpy().astype(np.int8)
    cst = np.r_[np.flatnonzero(np.diff(df["situation"].to_numpy(), prepend=-1) != 0),
                df.height].astype(np.int64)
    pst = np.r_[np.flatnonzero(np.diff(df["person"].to_numpy(), prepend=-1) != 0),
                df.height].astype(np.int64)
    return X, y, cst, pst, np.arange(len(pst) - 1, dtype=np.int64)


KIND = np.array([1, 2, 0], np.int64)
RIDX = np.array([0, 1, -1], np.int64)
THETA = np.array([0.9, np.log(0.6), -0.5, 0.7, 0.3])


def _kernel(blk, theta, R=60):
    X, y, cst, pst, pidx = blk
    return _mxl_kernel(X, y, cst, pst, pidx, np.atleast_2d(theta), KIND, RIDX, R, True, 100,
                       PRIMES[:2], 0, 1, 2, NO_CACHE)


def test_draws_halton_and_quantile():
    Z = draws(40, 37, 3)
    ref = np.array([[[sp.ndtri(_radical_inverse(100 + g * 37 + r, int(PRIMES[k])))
                      for k in range(3)] for r in range(37)] for g in range(40)])
    np.testing.assert_allclose(Z, ref, atol=1e-8)
    Zp = draws(2000, 100, 2, halton=False, seed=3)
    assert abs(Zp.mean()) < 0.01 and abs(Zp.std() - 1) < 0.01
    assert not np.allclose(Zp, draws(2000, 100, 2, halton=False, seed=4))


def test_kernel_gradient_and_loglik():
    df = _panel_data(N=40)
    blk = _block(df)
    ll, G, B = _kernel(blk, THETA)
    eps = 1e-6
    num = [(_kernel(blk, THETA + eps * e)[0][0] - _kernel(blk, THETA - eps * e)[0][0]) / (2 * eps)
           for e in np.eye(5)]
    np.testing.assert_allclose(G[0], num, atol=1e-6)
    # the simulated log-likelihood, by brute force with the same draws
    X, y, cst, pst, pidx = blk
    Z = draws(len(pidx), 60, 2)
    total = 0.0
    for p in range(len(pidx)):
        L = np.ones(60)
        for c in range(np.searchsorted(cst, pst[p]), np.searchsorted(cst, pst[p + 1])):
            a, b = cst[c], cst[c + 1]
            for r in range(60):
                beta = np.array([THETA[0] + THETA[3] * Z[p, r, 0],
                                 np.exp(THETA[1] + THETA[4] * Z[p, r, 1]), THETA[2]])
                v = X[a:b] @ beta
                L[r] *= np.exp(v[y[a:b] == 1][0] - v.max()) / np.exp(v - v.max()).sum()
        total += np.log(L.mean())
    assert ll[0] == pytest.approx(total, rel=1e-12)
    # the outer products are those of the panels' scores
    assert np.all(np.linalg.eigvalsh(B[0]) > -1e-9)


def test_cached_and_regenerated_draws_agree():
    df = _panel_data(N=30)
    X, y, cst, pst, pidx = _block(df)
    from logit_stream.mixed import _block_draws
    Zc = _block_draws(pidx, 50, 2, True, 100, PRIMES[:2], 0)
    a = _mxl_kernel(X, y, cst, pst, pidx, THETA[None], KIND, RIDX, 50, True, 100, PRIMES[:2], 0,
                    1, 2, NO_CACHE)
    b = _mxl_kernel(X, y, cst, pst, pidx, THETA[None], KIND, RIDX, 50, True, 100, PRIMES[:2], 0,
                    1, 2, Zc)
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v, rtol=1e-13)


@pytest.fixture(scope="module")
def panel_df():
    return _panel_data()


@pytest.fixture(scope="module")
def panel_fit(panel_df):
    return mixlogit_stream("chosen ~ x1 + x2 + x3 - 1", panel_df, case="situation",
                           panel="person", random={"x1": "normal", "x2": "lognormal"},
                           draws=200, vcov={"CRV1": "person"}, verbose=False)


def test_recovers_the_parameters(panel_fit):
    f = panel_fit
    assert f.converged
    assert f.coefnames == ["x1", "x2", "x3", "sd.x1", "sd.x2"]
    truth = np.array([1.0, np.log(0.5), -0.7, 0.8, 0.4])
    z = (f.beta - truth) / f.se
    assert np.all(np.abs(z) < 3.5), z
    assert f.loglik > f.loglik_clogit
    assert set(f.all_vcovs) == {"iid", "CRV1:person"} and f.vcov_type == "CRV1:person"
    assert f.n_clusters == {"person": 600} and f.df_t == 599
    # |sd| is reported, the signed estimates kept
    assert np.all(f.beta[3:] >= 0)
    np.testing.assert_allclose(np.abs(list(f.diagnostics["sd_signed"].values())), f.beta[3:])
    assert f.n_obs == f.n_panels == 600 and f.n_cases == 2400
    s = f.summary_dict()
    assert s["random"] == {"x1": "normal", "x2": "lognormal"} and s["draws"] == 200


def test_invariant_to_blocks_and_row_order(panel_df, panel_fit):
    # many small blocks, rows shuffled (pass 0 sorts them by chooser)
    shuffled = panel_df.sample(fraction=1.0, shuffle=True, seed=1)
    f = mixlogit_stream("chosen ~ x1 + x2 + x3 - 1", shuffled, case="situation",
                        panel="person", random={"x1": "normal", "x2": "lognormal"},
                        draws=200, vcov={"CRV1": "person"}, verbose=False, block_mb=0.02,
                        draws_cache_mb=0)
    assert f.diagnostics["store"]["blocks"] > 5
    np.testing.assert_allclose(f.beta, panel_fit.beta, rtol=1e-7, atol=1e-9)
    assert f.loglik == pytest.approx(panel_fit.loglik, rel=1e-12)


def test_bhhh_phase_reaches_the_same_optimum(panel_df, panel_fit):
    f = mixlogit_stream("chosen ~ x1 + x2 + x3 - 1", panel_df, case="situation",
                        panel="person", random={"x1": "normal", "x2": "lognormal"},
                        draws=200, verbose=False, switch=1.0)
    assert f.diagnostics["optimizer"]["bhhh_steps"] > 0
    # each stops within about 1e-4 standard errors of the optimum (tol=1e-8)
    assert np.all(np.abs(f.beta - panel_fit.beta) < 1e-3 * panel_fit.se)


def test_matches_the_exact_likelihood_with_one_random_coefficient():
    """One normal coefficient, no panel: the exact likelihood by Gauss-
    Legendre quadrature, maximized directly, against the simulated one with
    many Halton draws."""
    from scipy.optimize import minimize
    df = _panel_data(seed=2, N=1500, T=1, s2=0.0)
    f = mixlogit_stream("chosen ~ x1 + x2 + x3 - 1", df, case="situation",
                        random={"x1": "normal"}, draws=2000, verbose=False)
    X = df.select("x1", "x2", "x3").to_numpy()
    y = df["chosen"].to_numpy().astype(bool)
    xl, wl = np.polynomial.legendre.leggauss(400)
    z, w = 9 * xl, 9 * wl * np.exp(-(9 * xl) ** 2 / 2) / np.sqrt(2 * np.pi)

    def negll(t):
        V = (X @ t[:3])[:, None] + t[3] * X[:, [0]] * z[None, :]
        V = V.reshape(-1, 3, len(z))
        V = V - V.max(axis=1, keepdims=True)
        P = np.exp(V[y.reshape(-1, 3)]) / np.exp(V).sum(axis=1)
        return -np.log(P @ w).sum()

    res = minimize(negll, f.beta.copy(), method="BFGS", options={"gtol": 1e-8})
    exact = np.r_[res.x[:3], abs(res.x[3])]
    est = np.r_[f.beta[:3], abs(f.beta[3])]
    # simulation error with 2,000 Halton draws is far below the standard errors
    assert np.all(np.abs(est - exact) < 0.05 * f.se), (est - exact) / f.se
    assert f.loglik == pytest.approx(-res.fun, abs=0.05)


def test_errors(panel_df):
    with pytest.raises(ValueError, match="not covariates"):
        mixlogit_stream("chosen ~ x1 + x2 - 1", panel_df, case="situation",
                        random={"x9": "normal"}, verbose=False)
    with pytest.raises(ValueError, match="distribution"):
        mixlogit_stream("chosen ~ x1 + x2 - 1", panel_df, case="situation",
                        random={"x1": "uniform"}, verbose=False)
    with pytest.raises(ValueError, match="random= is empty"):
        mixlogit_stream("chosen ~ x1 + x2 - 1", panel_df, case="situation", random={},
                        verbose=False)
    with pytest.raises(NotImplementedError):
        mixlogit_stream("chosen ~ x1 + x2 - 1", panel_df.filter(pl.col("person") < 50),
                        case="situation", random={"x1": "normal"}, draws=20,
                        verbose=False).predict()


def test_robust_variance_names(panel_df):
    with pytest.raises(ValueError, match="CRV1"):
        mixlogit_stream("chosen ~ x1 + x2 + x3 - 1", panel_df, case="situation",
                        panel="person", random={"x1": "normal"}, draws=20, vcov="hetero",
                        verbose=False)
    one = panel_df.filter(pl.col("situation") % 4 == 0)
    f = mixlogit_stream("chosen ~ x1 + x2 + x3 - 1", one, case="situation",
                        random={"x1": "normal"}, draws=50, vcov="hetero", verbose=False)
    assert set(f.all_vcovs) == {"iid", "hetero"} and f.n_clusters == {}
    with pytest.raises(ValueError, match="hetero"):
        mixlogit_stream("chosen ~ x1 + x2 + x3 - 1", one, case="situation",
                        random={"x1": "normal"}, draws=20, vcov={"CRV1": "situation"},
                        verbose=False)
