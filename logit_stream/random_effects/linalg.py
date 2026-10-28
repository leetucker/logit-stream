"""The effects' information matrix, factorized: per-group blocks, for
groupings that nest (a tree, eliminated from the finest grouping up), or
two crossed groupings (the larger one's blocks eliminated, the smaller
one's effects solved densely); solves with it, and its inverse's blocks
(the effects' covariances)."""

from __future__ import annotations

from types import SimpleNamespace

import numba as nb
import numpy as np

from ..fe import _sym
from ..newton import _Driver


@nb.njit(parallel=True, cache=True)
def _group_w(order, seg, segg, P, w, Haa):
    """Haa[g] += sum over the group's rows of w (diag p - p p') (all levels):
    `_mnl_group_info` without the information with the coefficients."""
    J = P.shape[1]
    for s in nb.prange(len(segg)):
        g = segg[s]
        for r in range(seg[s], seg[s + 1]):
            i = order[r]
            wi = w[i]
            for l in range(J):
                pl_ = P[i, l]
                if pl_ == 0.0:
                    continue
                for m in range(J):
                    Haa[g, l, m] += wi * ((pl_ if l == m else 0.0) - pl_ * P[i, m])


def _inv(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A, hermitian=True)


def _sum_by(idx, V, G):
    """Sums of V's rows (any trailing shape) within each index."""
    out = np.zeros((G,) + V.shape[1:])
    np.add.at(out, idx, V)
    return out


# --------------------------------------------------------------------------
# the algebra: nested groupings (a tree)
# --------------------------------------------------------------------------

def _tree_factor(levels, Htt=None):
    """Eliminate the levels of a tree from the finest up (and b's block
    after, if `Htt`). `levels`: bottom-up, each with D (G x m x m), B (G x m
    x P, or None), X {j: G x m x m} (with its ancestor at level j) and anc
    {j: (G,)}. Returns what `_tree_apply` needs, the reduced information for
    b (`R`), and each level's log-determinants after elimination."""
    L = len(levels)
    D = [lv.D.copy() for lv in levels]
    X = [{j: x.copy() for j, x in lv.X.items()} for lv in levels]
    B = [None if lv.B is None else lv.B.copy() for lv in levels]
    R = None if Htt is None else Htt.copy()
    Dinv, logdet = [None] * L, [None] * L
    pd = True
    for k in range(L):
        Dinv[k] = _inv(D[k])
        sign, logdet[k] = np.linalg.slogdet(D[k])
        pd = pd and bool(np.all(sign > 0))
        anc = levels[k].anc
        for j in sorted(X[k]):
            XtD = X[k][j].transpose(0, 2, 1) @ Dinv[k]
            np.add.at(D[j], anc[j], -(XtD @ X[k][j]))
            if R is not None:
                np.add.at(B[j], anc[j], -(XtD @ B[k]))
            for l in sorted(X[k]):
                if l > j:
                    np.add.at(X[j][l], anc[j], -(XtD @ X[k][l]))
        if R is not None:
            G, m, P = B[k].shape
            R -= B[k].reshape(G * m, P).T @ (Dinv[k] @ B[k]).reshape(G * m, P)
    if R is not None:
        R = _sym(R)
    return SimpleNamespace(kind="tree", Dinv=Dinv, X=X, B=B, R=R, logdet=logdet,
                           anc=[lv.anc for lv in levels], D=D, pd=pd)


def _tree_apply(F, r_levels, r_beta=None):
    """Solve with a `_tree_factor`: the effects' steps per level (and b's)."""
    L = len(F.Dinv)
    r = [x.copy() for x in r_levels]
    rb = None if r_beta is None else r_beta.copy()
    for k in range(L):
        t = (F.Dinv[k] @ r[k][..., None])[..., 0]
        for j in F.X[k]:
            np.add.at(r[j], F.anc[k][j], -(F.X[k][j].transpose(0, 2, 1) @ t[..., None])[..., 0])
        if rb is not None:
            rb -= np.einsum("gmp,gm->p", F.B[k], t)
    db = None
    pd = True
    if rb is not None:
        db, pd = _Driver.solve(F.R, rb)
    d = [None] * L
    for k in reversed(range(L)):
        rhs = r[k] - (F.B[k] @ db if db is not None else 0.0)
        for j in F.X[k]:
            rhs = rhs - np.einsum("gij,gj->gi", F.X[k][j], d[j][F.anc[k][j]])
        d[k] = (F.Dinv[k] @ rhs[..., None])[..., 0]
    return d, db, pd


def _tree_cov(F):
    """The effects' covariance blocks (the blocks of M^-1) of a random-only
    `_tree_factor`: each group's own, and each finest group's (unit's)
    covariance of its total effect (the sum of all the blocks along its
    path to the top)."""
    L = len(F.Dinv)
    P = [None] * L
    P[L - 1] = F.Dinv[L - 1]
    for k in reversed(range(L - 1)):
        js = sorted(F.X[k])                       # the ancestors, k+1 .. top
        Xc = np.concatenate([F.X[k][j] for j in js], axis=2)
        Ppar = P[k + 1][F.anc[k][k + 1]]
        W = F.Dinv[k] @ Xc
        Cka = -W @ Ppar
        Ck = F.Dinv[k] + W @ Ppar @ W.transpose(0, 2, 1)
        top = np.concatenate([Ck, Cka], axis=2)
        bot = np.concatenate([Cka.transpose(0, 2, 1), Ppar], axis=2)
        P[k] = np.concatenate([top, bot], axis=1)
    m = F.Dinv[0].shape[1]
    own = [Pk[:, :m, :m] for Pk in P]
    G0, n = P[0].shape[0], P[0].shape[1] // m
    Vu = P[0].reshape(G0, n, m, n, m).sum(axis=(1, 3))
    return own, Vu


# --------------------------------------------------------------------------
# the algebra: two crossed groupings
# --------------------------------------------------------------------------

def _crossed_factor(DA, DB, Xu, ua, ub, BA=None, BB=None, Htt=None):
    """Eliminate the larger grouping's blocks (DA), then the smaller's dense
    system (DB, coupled through the units' blocks Xu, unit u between A's
    group ua[u] and B's group ub[u]), then b's block (if `Htt`)."""
    GA, m, _ = DA.shape
    GB = DB.shape[0]
    N = GB * m
    DAinv = _inv(DA)
    Yu = DAinv[ua] @ Xu                                    # D_a^-1 X_u
    Mb = np.zeros((N, N))
    for g in range(GB):
        Mb[g * m:(g + 1) * m, g * m:(g + 1) * m] = DB[g]
    order = np.argsort(ua, kind="stable")
    st = np.flatnonzero(np.r_[True, ua[order][1:] != ua[order][:-1]])
    segs = np.r_[st, len(order)]
    beta = Htt is not None
    if beta:
        P = Htt.shape[0]
        Bb = BB.reshape(N, P).copy()
        R = Htt - BA.reshape(GA * m, P).T @ (DAinv @ BA).reshape(GA * m, P)
        DBA = DAinv @ BA
    groups = []
    for s in range(len(segs) - 1):
        us = order[segs[s]:segs[s + 1]]
        a = ua[us[0]]
        idx = (ub[us][:, None] * m + np.arange(m)[None, :]).ravel()
        Xa = Xu[us].transpose(1, 0, 2).reshape(m, -1)        # m x (n_a m)
        Ya = Yu[us].transpose(1, 0, 2).reshape(m, -1)
        Mb[np.ix_(idx, idx)] -= Xa.T @ Ya
        if beta:
            Bb[idx] -= Xa.T @ DBA[a]
        groups.append((a, us, idx, Ya))
    Mb = _sym(Mb)
    Mbinv = _inv(Mb)
    sa, la = np.linalg.slogdet(DA)
    sb, lb = np.linalg.slogdet(Mb)
    out = SimpleNamespace(kind="crossed", DAinv=DAinv, Mbinv=Mbinv, Yu=Yu, Xu=Xu, ua=ua, ub=ub,
                          groups=groups, m=m, GA=GA, GB=GB, logdet=[la, np.array([lb])],
                          pd=bool(np.all(sa > 0) and sb > 0))
    if beta:
        out.BA, out.Bb = BA, Bb
        out.R = _sym(R - Bb.T @ Mbinv @ Bb)
    return out


def _crossed_apply(F, rA, rB, rb=None):
    m, GB = F.m, F.GB
    tA = (F.DAinv @ rA[..., None])[..., 0]
    rBf = rB.reshape(-1).copy()
    np.add.at(rBf.reshape(GB, m), F.ub, -(F.Xu.transpose(0, 2, 1) @ tA[F.ua][..., None])[..., 0])
    db, pd = None, True
    if rb is not None:
        rb = rb - np.einsum("gmp,gm->p", F.BA, tA) - F.Bb.T @ (F.Mbinv @ rBf)
        db, pd = _Driver.solve(F.R, rb)
        rBf = rBf - F.Bb @ db
    dB = (F.Mbinv @ rBf).reshape(GB, m)
    rhs = rA - (F.BA @ db if db is not None else 0.0)
    rhs = rhs.copy()
    np.add.at(rhs, F.ua, -(F.Xu @ dB[F.ub][..., None])[..., 0])
    dA = (F.DAinv @ rhs[..., None])[..., 0]
    return [dA, dB], db, pd


def _crossed_cov(F):
    """The covariance blocks of a random-only `_crossed_factor`: each
    group's, and each unit's covariance of its total effect."""
    m, GB = F.m, F.GB
    covB = np.stack([F.Mbinv[g * m:(g + 1) * m, g * m:(g + 1) * m] for g in range(GB)])
    covA = F.DAinv.copy()
    Vu = np.empty_like(F.Xu)
    for a, us, idx, Ya in F.groups:
        Z = Ya @ F.Mbinv[np.ix_(idx, idx)]                   # m x (n_a m)
        covA[a] = F.DAinv[a] + Z @ Ya.T
        for t, u in enumerate(us):
            Cab = -Z[:, t * m:(t + 1) * m]
            Vu[u] = Cab + Cab.T
    Vu += covA[F.ua] + covB[F.ub]
    return [covA, covB], Vu


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
