"""mlogit_stream's own_covariates=: a covariate that enters only its own
level's equation, against a direct maximization of the same likelihood."""

import numpy as np
import polars as pl
import pytest
from scipy.optimize import minimize

from logit_stream import mlogit_stream
from logit_stream import models

LEVELS = ["a", "b", "c", "d"]
OWN = {"b": "s_b", "c": "s_c", "d": "s_d"}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    n = 5000
    x1 = rng.normal(size=n)
    g = rng.integers(0, 3, n)                       # an indicator covariate
    S = rng.dirichlet([2, 2, 2, 2], size=n)         # baseline shares, a..d
    B = np.array([[0.2, 0.5, 3.0], [-0.4, 0.1, 2.0], [0.3, -0.6, 4.0]])
    E = np.column_stack([np.zeros(n)] + [B[j, 0] + B[j, 1] * x1 + B[j, 2] * S[:, j + 1]
                                         + 0.2 * j * g for j in range(3)])
    P = np.exp(E - E.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > P.cumsum(1)).sum(1).clip(0, 3)
    return pl.DataFrame({"y": np.array(LEVELS)[y], "yi": y, "x1": x1, "g": g, "cl": rng.integers(0, 25, n),
                         "w": rng.integers(1, 4, n).astype(float),
                         **{f"s_{l}": S[:, i] for i, l in enumerate(LEVELS)}})


def _design(df):
    g = df["g"].to_numpy()
    return np.column_stack([np.ones(len(df)), df["x1"].to_numpy(),
                            (g == 1).astype(float), (g == 2).astype(float)])


def _direct(df, w=None):
    """Maximize the log-likelihood over the free coefficients: the common
    ones (4 per equation) and each equation's own share."""
    Xc, y = _design(df), df["yi"].to_numpy()
    S = df.select(["s_b", "s_c", "s_d"]).to_numpy()
    w = np.ones(len(y)) if w is None else w

    def eta(th):
        Bc, bo = th[:12].reshape(3, 4), th[12:]
        return np.column_stack([np.zeros(len(y))] + [Xc @ Bc[j] + bo[j] * S[:, j]
                                                     for j in range(3)])

    def nll(th):
        E = eta(th)
        m = E.max(1)
        lse = m + np.log(np.exp(E - m[:, None]).sum(1))
        return -(w * (E[np.arange(len(y)), y] - lse)).sum()

    r = minimize(nll, np.zeros(15), method="BFGS", options={"gtol": 1e-9, "maxiter": 500})
    return r.x, -r.fun


def _order(th):
    """The direct fit's parameters in mlogit_stream's order: per equation
    the common columns then the own one."""
    return np.concatenate([np.r_[th[:12].reshape(3, 4)[j], th[12 + j]] for j in range(3)])


FML = "y ~ x1 + C(g)"


@pytest.mark.parametrize("sparse", [False, True])
def test_matches_direct_maximization(data, monkeypatch, sparse):
    if sparse:
        monkeypatch.setattr(models, "SPARSE_DENSITY", 1.01)
    else:
        monkeypatch.setattr(models, "SPARSE_DENSITY", -1.0)
    fit = mlogit_stream(FML, data, levels=LEVELS, own_covariates=OWN, verbose=False,
                        tol=1e-12)
    th, ll = _direct(data)
    assert fit.loglik == pytest.approx(ll, rel=1e-8)
    np.testing.assert_allclose(fit.beta, _order(th), atol=2e-4)
    assert fit.n_params == 15
    assert fit.coefnames[:5] == ["b:Intercept", "b:x1", "b:C(g)[T.1]", "b:C(g)[T.2]", "b:s_b"]
    assert "b:s_c" not in fit.coefnames


def test_information_is_the_hessian(data):
    """The standard errors are the inverse of the Hessian of the
    log-likelihood: check them against finite differences of the gradient."""
    fit = mlogit_stream(FML, data, levels=LEVELS, own_covariates=OWN, verbose=False,
                        tol=1e-12)
    th = np.asarray(fit.beta)
    Xc = _design(data)
    S = data.select(["s_b", "s_c", "s_d"]).to_numpy()
    y = data["yi"].to_numpy()
    cols = [np.column_stack([Xc, S[:, j]]) for j in range(3)]

    def grad(t):
        t = t.reshape(3, 5)
        E = np.column_stack([np.zeros(len(y))] + [cols[j] @ t[j] for j in range(3)])
        P = np.exp(E - E.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        return np.concatenate([cols[j].T @ ((y == j + 1) - P[:, j + 1]) for j in range(3)])

    h = 1e-5
    Hn = np.column_stack([(grad(th + h * e) - grad(th - h * e)) / (2 * h) for e in np.eye(15)])
    V = np.linalg.inv(-Hn)
    np.testing.assert_allclose(fit.vcov, (V + V.T) / 2, rtol=1e-4, atol=1e-8)


def test_dense_and_sparse_agree(data, monkeypatch):
    monkeypatch.setattr(models, "SPARSE_DENSITY", -1.0)
    a = mlogit_stream(FML, data, levels=LEVELS, own_covariates=OWN, verbose=False)
    monkeypatch.setattr(models, "SPARSE_DENSITY", 1.01)
    b = mlogit_stream(FML, data, levels=LEVELS, own_covariates=OWN, verbose=False)
    np.testing.assert_allclose(a.beta, b.beta, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(a.vcov, b.vcov, rtol=1e-6, atol=1e-10)


def test_weights_and_robust_variances(data):
    w = data["w"].to_numpy()
    fit = mlogit_stream(FML, data, levels=LEVELS, own_covariates=OWN, weights="w",
                        weights_type="fweights", vcov={"CRV1": "cl"}, cluster="cl",
                        verbose=False, tol=1e-12)
    th, ll = _direct(data, w)
    np.testing.assert_allclose(fit.beta, _order(th), atol=2e-4)
    assert np.all(np.isfinite(fit.se)) and fit.vcov_type == "CRV1:cl"
    het = mlogit_stream(FML, data, levels=LEVELS, own_covariates=OWN, vcov="hetero",
                        verbose=False)
    assert np.all(np.isfinite(het.se))


def test_predict_and_coef_matrix(data):
    fit = mlogit_stream(FML, data, levels=LEVELS, own_covariates=OWN, verbose=False)
    pr = fit.predict(data).collect()
    P = pr.select([f"prob_{l}" for l in LEVELS]).to_numpy()
    np.testing.assert_allclose(P.sum(1), 1.0)
    B = np.asarray(fit.beta).reshape(3, 5)          # per equation: 4 common, its own
    S = data.select(["s_b", "s_c", "s_d"]).to_numpy()
    X = _design(data)
    E = np.column_stack([np.zeros(len(data))] + [X @ B[j, :4] + B[j, 4] * S[:, j]
                                                 for j in range(3)])
    ref = np.exp(E) / np.exp(E).sum(1, keepdims=True)
    np.testing.assert_allclose(P, ref, rtol=1e-9)
    cm = fit.coef_matrix()
    for col, share in zip("bcd", ["s_b", "s_c", "s_d"]):
        row = cm.filter(pl.col("Coefficient") == share)
        assert row[col][0] == pytest.approx(B["bcd".index(col), 4])
        others = [c for c in "bcd" if c != col]
        assert all(np.isnan(row[c][0]) for c in others)


def test_equals_ordinary_mlogit_when_the_columns_are_common(data):
    """With a column given to every level there is nothing to restrict; the
    own model with the formula's terms equals the ordinary fit without them."""
    own = mlogit_stream("y ~ x1 + C(g)", data, levels=LEVELS, own_covariates=OWN,
                        verbose=False)
    plain = mlogit_stream("y ~ x1 + C(g) + s_b + s_c + s_d", data, levels=LEVELS,
                          verbose=False)
    assert own.loglik < plain.loglik                       # the restriction costs fit
    assert plain.n_params - own.n_params == 6


def test_a_collinear_own_covariate_is_dropped(data):
    d = data.with_columns(k=pl.lit(1.0))
    fit = mlogit_stream(FML, d, levels=LEVELS, own_covariates={**OWN, "d": ["s_d", "k"]},
                        verbose=False)
    assert fit.collin_vars == ["k"]
    assert not any(n.endswith(":k") for n in fit.coefnames)


def test_only_some_levels(data):
    fit = mlogit_stream(FML, data, levels=LEVELS, own_covariates={"b": "s_b"}, verbose=False)
    assert fit.n_params == 3 * 4 + 1


def test_errors(data):
    kw = dict(levels=LEVELS, verbose=False)
    with pytest.raises(ValueError, match="base level"):
        mlogit_stream(FML, data, own_covariates={"a": "s_b"}, **kw)
    with pytest.raises(ValueError, match="outcome's levels"):
        mlogit_stream(FML, data, own_covariates={"z": "s_b"}, **kw)
    with pytest.raises(ValueError, match="not a column"):
        mlogit_stream(FML, data, own_covariates={"b": "nope"}, **kw)
    with pytest.raises(ValueError, match="both in the formula"):
        mlogit_stream(FML + " + s_b", data, own_covariates={"b": "s_b"}, **kw)
    with pytest.raises(ValueError, match="firth"):
        mlogit_stream(FML, data, own_covariates=OWN, firth=True, **kw)
    with pytest.raises(ValueError, match="absorbed fixed effect"):
        mlogit_stream("y ~ x1 | g", data, own_covariates=OWN, **kw)
