"""The mixed logit's pass (`mixed.py`), vectorized over rows and draws, and
its draws generated on the device."""

from __future__ import annotations

import numpy as np

from ._common import _host, _segmax, _segsum

MXL_CHUNK_ELEMS = 4_000_000     # rows x draws per batch of choosers (each such array 32 MB)

_ACKLAM = dict(
    a=(-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
       1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00),
    b=(-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
       6.680131188771972e+01, -1.328068155288572e+01),
    c=(-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
       -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00),
    d=(7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
       3.754408661907416e+00))



def _ndtri(xp, p):
    """`mixed._ndtri` (Acklam's approximation) elementwise, the same
    arithmetic."""
    A = _ACKLAM

    def tail(q):
        c, d = A["c"], A["d"]
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        return num / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    lo, hi = p < 0.02425, p > 1.0 - 0.02425
    q = p - 0.5
    r = q * q
    a, b = A["a"], A["b"]
    mid = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q \
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ql = xp.sqrt(-2.0 * xp.log(xp.where(lo, p, 0.5)))
        qh = xp.sqrt(-2.0 * xp.log(xp.where(hi, 1.0 - p, 0.5)))
    return xp.where(lo, tail(ql), xp.where(hi, -tail(qh), mid))



def mxl_draws(xp, pidx, R, Kr, halton, drop, primes, seed):
    """`mixed._block_draws` on the device: the choosers' standard normal
    draws, n_choosers x R x Kr, from their positions `pidx` among all
    choosers (Halton points by the radical inverse, digit by digit for all
    at once; or the counter-based hash of the pseudo-random draws)."""
    g = xp.asarray(pidx, dtype=xp.int64)[:, None]
    r = xp.arange(R, dtype=xp.int64)[None, :]
    out = xp.empty((g.shape[0], R, max(Kr, 1)))
    for k in range(Kr):
        if halton:
            i = drop + g * R + r
            b = int(primes[k])
            u = xp.zeros(i.shape)
            f = 1.0
            while bool((i > 0).any()):
                f /= b
                u += f * (i % b)
                i = i // b
        else:
            M64 = xp.uint64
            with np.errstate(over="ignore"):
                z = ((g * R + r) * 64 + k).astype(M64) \
                    + M64(np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15))
                z = z + M64(0x9E3779B97F4A7C15)
                z = (z ^ (z >> M64(30))) * M64(0xBF58476D1CE4E5B9)
                z = (z ^ (z >> M64(27))) * M64(0x94D049BB133111EB)
                z = z ^ (z >> M64(31))
            u = ((z >> M64(11)).astype(xp.float64) + 0.5) * (1.0 / 9007199254740992.0)
        out[:, :, k] = _ndtri(xp, u)
    return out



def mxl_pass(xp, X, y, cst, pst, Z, TH, kind, ridx, n_bhhh, chunk=None):
    """`mixed._mxl_kernel` vectorized: the simulated log-likelihood and
    gradient at each row of TH, and the choosers' score outer products at
    the first `n_bhhh`, over one block (X rows x K; cst, pst the choice
    sets' and choosers' row starts, with the end; Z the choosers' draws).
    For a batch of choosers, the utilities are a rows x draws array; each
    choice set's probabilities a softmax down its rows, for every draw; a
    chooser's log-likelihood per draw the sum over its sets, and its
    weights over the draws their softmax. Every gradient term is
    x_ik times a per-row sum over the draws (of the weight, the residual,
    and for a random coefficient its draw or its value), so the gradient is
    the rows' P columns summed, and a chooser's score the sum over its
    rows."""
    chunk = MXL_CHUNK_ELEMS if chunk is None else chunk
    n, K = X.shape
    M, P = TH.shape
    R = Z.shape[1]
    kind = np.asarray(kind)
    ridx = np.asarray(ridx)
    fixed = np.flatnonzero(kind == 0)
    rand = np.flatnonzero(kind > 0)
    nb_ = max(n_bhhh, 1)
    ll = np.zeros(M)
    G = np.zeros((M, P))
    B = np.zeros((nb_, P, P))
    pst_h = np.asarray(_host(xp, pst))
    cst_h = np.asarray(_host(xp, cst))
    yf = y.astype(xp.float64)
    THd = xp.asarray(TH)
    logR = np.log(R)
    # batches of whole choosers, about `chunk` rows x draws
    per = max(1, chunk // R)
    bounds, p0 = [], 0
    npan = len(pst_h) - 1
    while p0 < npan:
        p1 = int(np.searchsorted(pst_h, pst_h[p0] + per, side="right")) - 1
        p1 = min(max(p1, p0 + 1), npan)
        bounds.append((p0, p1))
        p0 = p1
    for p0, p1 in bounds:
        a, b = int(pst_h[p0]), int(pst_h[p1])
        c0, c1 = int(np.searchsorted(cst_h, a)), int(np.searchsorted(cst_h, b))
        Xc, yc = X[a:b], yf[a:b]
        cs = xp.asarray(cst_h[c0:c1 + 1] - a)
        ps = xp.asarray(pst_h[p0:p1 + 1] - a)
        nset, npc = c1 - c0, p1 - p0
        set_of = xp.repeat(xp.arange(nset), xp.diff(cs))
        pan_of = xp.repeat(xp.arange(npc), xp.diff(ps))
        # the choosers' starts among the sets
        pset = xp.asarray(np.searchsorted(cst_h[c0:c1 + 1] - a, pst_h[p0:p1 + 1] - a))
        Zc = Z[p0:p1]
        for m in range(M):
            th = THd[m]
            V = (Xc[:, fixed] @ th[fixed])[:, None] if len(fixed) else xp.zeros((b - a, 1))
            V = xp.broadcast_to(V, (b - a, R)).copy()
            Bk = {}
            for k in rand:
                e = th[k] + th[K + ridx[k]] * Zc[:, :, ridx[k]]
                Bk[k] = e if kind[k] == 1 else xp.exp(e)
                V += Xc[:, k][:, None] * Bk[k][pan_of]
            vmax = _segmax(xp, V, cs)
            E = xp.exp(V - vmax[set_of])
            S = _segsum(xp, E, cs)
            lset = _segsum(xp, yc[:, None] * V, cs) - vmax - xp.log(S)
            logL = _segsum(xp, lset, pset)
            mx = logL.max(axis=1)
            W = xp.exp(logL - mx[:, None])
            tot = W.sum(axis=1)
            ll[m] += float((mx + xp.log(tot)).sum()) - npc * logR
            W = W / tot[:, None]
            res = (yc[:, None] - E / S[set_of]) * W[pan_of]
            F = xp.empty((b - a, P))
            rw = res.sum(axis=1)
            for k in fixed:
                F[:, k] = Xc[:, k] * rw
            for k in rand:
                zk = Zc[:, :, ridx[k]][pan_of]
                if kind[k] == 1:
                    F[:, k] = Xc[:, k] * rw
                    F[:, K + ridx[k]] = Xc[:, k] * (res * zk).sum(axis=1)
                else:
                    rb = res * Bk[k][pan_of]
                    F[:, k] = Xc[:, k] * rb.sum(axis=1)
                    F[:, K + ridx[k]] = Xc[:, k] * (rb * zk).sum(axis=1)
            G[m] += _host(xp, F.sum(axis=0))
            if m < n_bhhh:
                Sp = _segsum(xp, F, ps)
                B[m] += _host(xp, Sp.T @ Sp)
    return ll, G, B
