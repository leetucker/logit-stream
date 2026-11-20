"""Probabilities averaged over the random effects' distribution, for
``predict(effects="average")``.

Each row's random effect (the sum of its groupings') is normal: for a group
the fit saw, its conditional distribution given the group's rows (mean the
mode that `ranef` reports, covariance the conditional one, whose diagonal
is ``sd_<grouping>``); for a group the fit did not see, the prior, N(0,
Sigma). The averaged probability is E[F(eta + delta)], by Gauss-Hermite
quadrature: 32 nodes for a scalar effect (the binary and ordered logit),
the product of 5 nodes per dimension for the multinomial logit's J - 1.
"""

from __future__ import annotations

import numba as nb
import numpy as np
import polars as pl
from numpy.polynomial.hermite_e import hermegauss

SCALAR_NODES = 32
VECTOR_NODES = 5


def scalar_rule():
    z, w = hermegauss(SCALAR_NODES)
    return z, w / w.sum()


def vector_rule(d):
    z, w = hermegauss(VECTOR_NODES)
    Z = np.array(np.meshgrid(*([z] * d), indexing="ij")).reshape(d, -1).T
    W = np.prod(np.array(np.meshgrid(*([w] * d), indexing="ij")).reshape(d, -1), axis=0)
    return np.ascontiguousarray(Z), W / W.sum()


def spread_inputs(result, frame):
    """`frame` with each random grouping's row index into its conditional
    covariances (``__ls_gi<k>``, null for a group the fit did not see), the
    columns' names, and each grouping's covariances with the prior's
    appended as the last row: (G + 1, m, m)."""
    spec = result._spec
    if spec.get("draw_index") is not None:
        raise ValueError('effects="average" is for the fitted result: a parameter draw\'s '
                         "group effects are draws (for multiple imputation, draw them with "
                         "draw_parameters(ranef=True, newdata=...))")
    if not any(kind in ("random", "random_slopes") for _, _, kind in spec.get("effects", [])):
        raise ValueError('effects="average" needs random intercepts')
    if "ranef_cov" not in spec:
        raise ValueError("this result has no conditional covariances of its random effects "
                         "(refit it with this version)")
    names, covs = [], []
    for d, _, kind in spec["effects"]:
        if kind not in ("random", "random_slopes"):
            continue
        C = spec["ranef_cov"][d]
        Sig = np.atleast_2d(np.asarray(result.random_effects[d]["Sigma"], dtype=float))
        if kind == "random_slopes":
            # the effects' own covariance (q x q), with its layout: each row's
            # covariance is then L C L', L the row's design by level
            Cx = np.ascontiguousarray(np.concatenate([np.nan_to_num(C), Sig[None]]))
            covs.append((Cx, spec["slope_layout"][d], list(spec["xcols"])))
        else:
            m = C.shape[1]
            S = np.zeros((m, m))
            if m == 1:
                S[:] = Sig.reshape(1, 1)
            else:
                S[1:, 1:] = Sig
            covs.append(np.ascontiguousarray(np.concatenate([np.nan_to_num(C), S[None]])))
        cols = spec["fe_cols"][d]
        name = f"__ls_gi{len(names)}"
        names.append(name)
        table = spec["ranef"][d]
        tschema = table.collect_schema()
        right = table.lazy().select(*cols).with_row_index(name) \
            .with_columns(pl.col(name).cast(pl.Int64))
        frame = frame.with_columns(pl.col(c).cast(tschema[c], strict=False) for c in cols)
        frame = frame.join(right, on=cols, how="left", maintain_order="left")
    return frame, names, covs


def row_covariance(df, names, covs, X=None, m=None):
    """Each row's total effect's covariance, (n, m, m): the sum over the
    groupings of its group's conditional covariance, or the prior's; for
    random slopes L C L', L (n x m x q) the row's covariates by level (`X`
    the rows' design, `m` the levels with effects' width)."""
    V = None
    for name, C in zip(names, covs):
        if isinstance(C, tuple):
            Cq, layout, xcols = C
            gi = df[name].fill_null(Cq.shape[0] - 1).to_numpy()
            n = len(gi)
            L = np.zeros((n, m, len(layout)))
            for k, (j, t) in enumerate(layout):
                L[:, j, k] = 1.0 if t == "1" else X[:, xcols.index(t)]
            Vg = np.einsum("njk,nkl,nml->njm", L, Cq[gi], L)
        else:
            gi = df[name].fill_null(C.shape[0] - 1).to_numpy()
            Vg = C[gi]
        V = Vg if V is None else V + Vg
    return V


def average_scalar(eta, sd, prob):
    """Mean over delta ~ N(0, sd^2) of prob(eta + delta), `prob` mapping an
    (n, Q) array of linear predictors to (n, Q, ...) probabilities."""
    z, w = scalar_rule()
    P = prob(eta[:, None] + sd[:, None] * z[None, :])
    return np.tensordot(P, w, axes=([1], [0])) if P.ndim == 2 else np.einsum("nq...,q->n...",
                                                                               P, w)


@nb.njit(parallel=True, cache=True)
def _softmax_average(full, V, Z, w, out):
    n, J = full.shape
    d = J - 1
    Q = Z.shape[0]
    for i in nb.prange(n):
        bad = False
        for j in range(J):
            if np.isnan(full[i, j]):
                bad = True
        if bad:
            for j in range(J):
                out[i, j] = np.nan
            continue
        # the free block's Cholesky factor (zero columns where it is singular)
        L = np.zeros((d, d))
        for a in range(d):
            s = V[i, a + 1, a + 1]
            for k in range(a):
                s -= L[a, k] * L[a, k]
            if s <= 1e-14:
                continue
            L[a, a] = np.sqrt(s)
            for b in range(a + 1, d):
                t = V[i, b + 1, a + 1]
                for k in range(a):
                    t -= L[b, k] * L[a, k]
                L[b, a] = t / L[a, a]
        acc = np.zeros(J)
        e = np.empty(J)
        for q in range(Q):
            mx = full[i, 0]
            e[0] = full[i, 0]
            for j in range(1, J):
                v = full[i, j]
                for k in range(j):
                    v += L[j - 1, k] * Z[q, k]
                e[j] = v
                if v > mx:
                    mx = v
            tot = 0.0
            for j in range(J):
                e[j] = np.exp(e[j] - mx)
                tot += e[j]
            for j in range(J):
                acc[j] += w[q] * e[j] / tot
        for j in range(J):
            out[i, j] = acc[j]


def average_softmax(full, V):
    """Mean over delta ~ N(0, V_i) (the base level's 0) of softmax(full_i +
    delta), row by row."""
    Z, w = vector_rule(full.shape[1] - 1)
    out = np.empty_like(full)
    _softmax_average(np.ascontiguousarray(full, dtype=np.float64),
                     np.ascontiguousarray(V, dtype=np.float64), Z, w, out)
    return out
