"""Adaptive quadrature for the multinomial logit's random intercepts
(`random_method="adaptive"` with `mlogit_stream`): the estimates, the
log-likelihood and the standard errors against brute-force integration and
its numerical Hessian, with and without an absorbed fixed effect; the
diagonal Sigma; refusals."""

import numpy as np
import polars as pl
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import logsumexp

from logit_stream import mlogit_stream

_X, _W = hermgauss(40)
_Z = np.sqrt(2) * np.array(np.meshgrid(_X, _X)).reshape(2, -1).T
_LW = np.log(np.outer(_W, _W).ravel() / np.pi)


def _data(seed=1, G=250, cells=None):
    """J = 3, tiny groups (1 + Geometric(0.4) rows), correlated intercepts."""
    rng = np.random.default_rng(seed)
    n_g = 1 + rng.geometric(0.4, G)
    g = np.repeat(np.arange(G), n_g)
    n = len(g)
    x = rng.normal(size=n)
    a = rng.multivariate_normal([0, 0], [[0.64, 0.2], [0.2, 0.49]], G)
    E = np.column_stack([np.zeros(n), -0.5 + 0.6 * x + a[g, 0], -1.0 - 0.4 * x + a[g, 1]])
    cols = {}
    if cells:
        cell = rng.integers(0, cells, G)
        E[:, 1:] += rng.normal(0, 0.5, (cells, 2))[cell][g]
        cols["cell"] = cell[g]
    u = rng.random(n)[:, None]
    y = (np.cumsum(np.exp(E) / np.exp(E).sum(1, keepdims=True), 1) < u).sum(1)
    return pl.DataFrame({"y": y, "x": x, "g": g, **cols})


def _brute(df, cells):
    """Minus the marginal log-likelihood by a 40 x 40 Gauss-Hermite grid on
    the prior, in (slopes (and intercepts without cells), the cells' effects
    less the first's, Sigma11, Sigma12, Sigma22)."""
    y, x = df["y"].to_numpy(), df["x"].to_numpy()
    g = df["g"].to_numpy()
    st = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    n = len(y)
    if cells:
        c = df["cell"].to_numpy()
        D = np.zeros((n, cells))
        D[np.arange(n), c] = 1.0
        X = np.column_stack([D, x])                    # per level: cells, then x
    else:
        X = np.column_stack([np.ones(n), x])
    K = X.shape[1]

    def nll(p):
        B = p[:2 * K].reshape(2, K)
        S = np.array([[p[-3], p[-2]], [p[-2], p[-1]]])
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return np.inf
        A = _Z @ L.T
        eta = np.column_stack([np.zeros(n), X @ B.T])
        tot = eta[:, None, :] + np.concatenate([np.zeros((len(A), 1)), A], 1)[None]
        lp = (np.take_along_axis(tot, y[:, None, None].repeat(len(A), 1), 2)[..., 0]
              - logsumexp(tot, 2))
        return -logsumexp(np.add.reduceat(lp, st, axis=0) + _LW[None], axis=1).sum()
    return nll, K


def _hessian(f, p, h=1e-4):
    n = len(p)
    H = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i, e_j = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (f(p + e_i + e_j) - f(p + e_i - e_j) - f(p - e_i + e_j)
                                 + f(p - e_i - e_j)) / (4 * h * h)
    return H


@pytest.mark.parametrize("cells", [None, 3])
def test_matches_brute_force(cells):
    df = _data(seed=1 if cells is None else 2, cells=cells)
    fml = "y ~ x | cell" if cells else "y ~ x"
    fit = mlogit_stream(fml, df, random="g", random_method="adaptive", verbose=False)
    assert fit.converged
    nll, K = _brute(df, cells)
    S = fit.random_effects["g"]["Sigma"]
    b = dict(zip(fit.coefnames, fit.beta))
    if cells:
        fe = fit.fixef().collect().sort("cell")
        lev = fe.select("fe_cell:1", "fe_cell:2").to_numpy().T        # 2 x cells
        p0 = np.r_[lev[0], b["1:x"], lev[1], b["2:x"]]
    else:
        p0 = np.r_[b["1:Intercept"], b["1:x"], b["2:Intercept"], b["2:x"]]
    p0 = np.r_[p0, S[0, 0], S[0, 1], S[1, 1]]
    # the fit's log-likelihood is brute force's at its estimates, which are its maximum
    assert abs(fit.loglik + nll(p0)) < 0.01
    r = minimize(nll, p0, method="BFGS", options={"gtol": 1e-6})
    np.testing.assert_allclose(p0, r.x, atol=3e-3)
    # the standard errors: the inverse of brute force's numerical Hessian
    V = np.linalg.inv(_hessian(nll, r.x))
    se_b = np.sqrt(np.diag(V))
    ix = [K - 1, 2 * K - 1] if cells else [0, 1, 2, 3]
    se_fit = np.sqrt(np.diag(fit.vcov))
    np.testing.assert_allclose(se_fit[[fit.coefnames.index(k) for k in
                                       (["1:x", "2:x"] if cells else
                                        ["1:Intercept", "1:x", "2:Intercept", "2:x"])]],
                               se_b[ix], rtol=0.02)
    Sse = fit.random_effects["g"]["Sigma_se"]
    np.testing.assert_allclose([Sse[0, 0], Sse[0, 1], Sse[1, 1]], se_b[-3:], rtol=0.03)
    # the conditional sds are reported for the non-base levels
    t = fit.ranef().collect()
    assert t["sd_g:1"].min() > 0 and t["sd_g:0"].max() == 0


def test_diagonal_and_laplace_differ():
    df = _data(seed=3)
    d = mlogit_stream("y ~ x", df, random="g", random_method="adaptive",
                      random_cov="diagonal", verbose=False)
    S = d.random_effects["g"]["Sigma"]
    assert S[0, 1] == 0 and d.converged
    lap = mlogit_stream("y ~ x", df, random="g", random_cov="diagonal", verbose=False)
    assert d.loglik > lap.loglik            # Laplace's value is below the integral's


def test_refusals():
    df = _data(seed=4, G=60)
    with pytest.raises(ValueError, match="at least 5"):
        mlogit_stream("y ~ x", df, random="g", random_method="adaptive", random_nodes=3,
                      verbose=False)
    many = df.with_columns((pl.col("y") + 3 * (pl.col("x") > 0).cast(pl.Int64)).alias("y"))
    with pytest.raises(ValueError, match="points per group"):
        mlogit_stream("y ~ x", many, random="g", random_method="adaptive", random_nodes=8,
                      verbose=False)
    with pytest.raises(ValueError, match="iid"):
        mlogit_stream("y ~ x", df, random="g", random_method="adaptive",
                      vcov={"CRV1": "g"}, verbose=False)


def test_large_groups_take_the_smaller_rule(monkeypatch):
    """Groups of LARGE_ROWS rows or more use 3 points per dimension: the
    estimates are those of the full rule everywhere, to a small fraction of
    their standard errors; the log-likelihood is the full rule's."""
    from logit_stream.random_effects import adaptive_mnl as M
    rng = np.random.default_rng(5)
    G = 300
    n_g = np.where(rng.random(G) < 0.3, rng.integers(30, 90, G), 1 + rng.geometric(0.4, G))
    g = np.repeat(np.arange(G), n_g)
    n = len(g)
    x = rng.normal(size=n)
    a = rng.multivariate_normal([0, 0], [[0.64, 0.2], [0.2, 0.49]], G)
    E = np.column_stack([np.zeros(n), -0.5 + 0.6 * x + a[g, 0], -1.0 - 0.4 * x + a[g, 1]])
    y = (np.cumsum(np.exp(E) / np.exp(E).sum(1, keepdims=True), 1)
         < rng.random(n)[:, None]).sum(1)
    df = pl.DataFrame({"y": y, "x": x, "g": g})
    tiered = mlogit_stream("y ~ x", df, random="g", random_method="adaptive", verbose=False)
    monkeypatch.setattr(M, "LARGE_ROWS", 10**9)
    full = mlogit_stream("y ~ x", df, random="g", random_method="adaptive", verbose=False)
    se = np.sqrt(np.diag(full.vcov))
    assert np.max(np.abs(tiered.beta - full.beta) / se) < 0.03
    St, Sf = tiered.random_effects["g"], full.random_effects["g"]
    assert np.max(np.abs(St["Sigma"] - Sf["Sigma"]) / Sf["Sigma_se"]) < 0.05
    assert abs(tiered.loglik - full.loglik) < 0.01
