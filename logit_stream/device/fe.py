"""The absorbed fixed effect's pass (`fe.py`): each level's gradient,
information and cross block with the slopes, by scatter-adds over the rows'
level codes."""

from __future__ import annotations

import numpy as np

from ._common import _gram, _scatter_add
from .models import _ord_assemble, _ord_rows


def fe_pass(xp, fe, theta, dstore, hessian, K0):
    """`_AbsorbedFE.pass_`'s sums on the device: the log-likelihood; the
    gradient of the slopes (`gt`) and of each level's effects (`ga`, G x m);
    and with `hessian` the slopes' information (`Htt`), each level's (`Haa`,
    G x m x m) and between the two (`B`, G x m x P), each level's sums by
    scatter-adds over the rows' level codes. Returned as numpy arrays (the
    Schur steps that follow are per-level and small)."""
    m = fe.model
    P, G, J = m.P, fe.G, fe.m
    th = xp.asarray(theta[:P], dtype=xp.float64)
    A = xp.asarray(fe._effects(theta), dtype=xp.float64)
    keep_g = xp.asarray(fe.keep_g)
    ll = 0.0
    gt, ga = xp.zeros(P), xp.zeros((G, J))
    Htt = xp.zeros((P, P)) if hessian else None
    Haa = xp.zeros((G, J, J)) if hessian else None
    B = xp.zeros((G, J, P)) if hessian else None
    cols = None if len(m.cols) == K0 else xp.asarray(m.cols)
    for blk in dstore:
        f = blk["fe"].astype(xp.int64)
        keep = keep_g[f]
        sel = None if bool(keep.all()) else xp.nonzero(keep)[0]
        X = blk["X"] if cols is None else blk["X"][:, cols]
        X, y, w, off, f = (None if a is None else a if sel is None else a[sel]
                           for a in (X, blk["y"], blk.get("w"), blk.get("off"), f))
        if not len(y):
            continue
        if m.kind == "mlogit":
            ll = ll + _fe_mnl(xp, m, th, A, X, y, w, f, gt, ga, Htt, Haa, B)
        else:
            o = A[f, 0] if off is None else A[f, 0] + off
            r = _ord_rows(xp, m, th, X, y, w, hessian, o)
            l, g, H = _ord_assemble(xp, m, r, X, w, hessian)
            ll = ll + l
            gt += g
            ga[:, 0] += xp.bincount(f, weights=r["s"], minlength=G)[:G]
            if hessian:
                Htt += H
                Haa[:, 0, 0] += xp.bincount(f, weights=r["h"], minlength=G)[:G]
                Q = r["Q"][:, 1:] if m.fix_first else r["Q"]
                Bt = xp.zeros((G, P))
                _scatter_add(xp, Bt, f, xp.concatenate([X * r["h"][:, None], Q], axis=1))
                B[:, 0, :] += Bt
    a = xp
    out = [float(ll), a.asnumpy(gt) if a is not np else gt, a.asnumpy(ga) if a is not np else ga]
    for M in (Htt, Haa, B):
        out.append(None if M is None else (a.asnumpy(M) if a is not np else M))
    return out



def _fe_mnl(xp, m, th, A, X, y, w, f, gt, ga, Htt, Haa, B):
    n, K, J = len(y), X.shape[1], m.J
    E = xp.empty((n, J))
    E[:, 0] = 0.0
    E[:, 1:] = X @ th.reshape(J - 1, K).T
    E += A[f]
    mx = E.max(axis=1)
    ex = xp.exp(E - mx[:, None])
    den = ex.sum(axis=1)
    Pr = ex / den[:, None]
    yi = y.astype(xp.int64)
    r = xp.arange(n)
    lp = E[r, yi] - mx - xp.log(den)
    ll = lp.sum() if w is None else lp @ w
    R = -Pr
    R[r, yi] += 1.0
    if w is not None:
        R = R * w[:, None]
    gt += (X.T @ R[:, 1:]).T.ravel()
    _scatter_add(xp, ga, f, R)
    if Htt is None:
        return ll
    Pw = Pr if w is None else Pr * w[:, None]
    for j in range(1, J):
        sj = slice((j - 1) * K, j * K)
        Htt[sj, sj] += _gram(X, Pw[:, j] * ((den - ex[:, j]) / den))
        for k in range(j + 1, J):
            sk = slice((k - 1) * K, k * K)
            Hjk = _gram(X, Pw[:, j] * Pr[:, k])
            Htt[sj, sk] -= Hjk
            Htt[sk, sj] -= Hjk
    # each row's w (diag p - p p') over all J levels: the levels' block, and
    # with the row's x its cross block with the slopes, in sub-chunks of rows
    # so that the n x J x P temporary stays near 64 MB
    W = -(Pw[:, :, None] * Pr[:, None, :])
    idx = xp.arange(J)
    W[:, idx, idx] += Pw
    _scatter_add(xp, Haa, f, W)
    P = (J - 1) * K
    step = max(1, int(64e6 // (8 * J * max(P, 1))))
    for s0 in range(0, n, step):
        sl = slice(s0, min(n, s0 + step))
        vals = (W[sl, :, 1:, None] * X[sl, None, None, :]).reshape(-1, J, P)
        _scatter_add(xp, B, f[sl], vals)
    return ll
