"""The nested logit's pass (`nested.py`), vectorized over choice sets: dense
sets x nodes arrays carried up the tree a nest at a time."""

from __future__ import annotations

import numpy as np

from ._common import _host, _scatter_add, _scatter_max

NL_CHUNK_BYTES = 256e6          # the (sets x nodes x P) arrays, per batch of choice sets



def nl_pass(xp, X, y, w, off, starts, nest, TH, K, tree, n_bhhh, scores=False):
    """`nested._nl_kernel` vectorized over the choice sets: each set's node
    values in dense (sets x nodes) arrays (nodes the nests, then the root).
    The rows' exponents go to their (set, bottom nest) by scatter-max and
    scatter-add; then each nest, children first, forms its inclusive value
    and value, with their gradients, for all sets at once, and adds itself
    to its parent's running log-sum-exp; finally each set's chosen
    alternative's path is walked up to the root, all sets a level at a
    time. Returns (ll, G, B) as the kernel does; with `scores`, each set's
    unweighted score at TH[0] instead (sets x P, a device array)."""
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return _nl_pass(xp, X, y, w, off, starts, nest, TH, K, tree, n_bhhh, scores)



def _nl_pass(xp, X, y, w, off, starts, nest, TH, K, tree, n_bhhh, scores=False):
    M, P = len(tree.parent), TH.shape[1]
    N1 = M + 1
    parent = np.r_[tree.parent, M]
    lam_of = np.r_[tree.lam_of, -1]
    starts_h = np.asarray(_host(xp, starts))
    C = len(starts_h) - 1
    T = TH.shape[0]
    nb_ = max(n_bhhh, 1)
    ll, G, B = np.zeros(T), np.zeros((T, P)), np.zeros((nb_, P, P))
    yb = y > 0
    per = max(1, int(NL_CHUNK_BYTES // (8 * N1 * (3 * P + 6))))
    THd = xp.asarray(TH)
    par_d = xp.asarray(parent)
    lamof_d = xp.asarray(lam_of)
    S_all = []
    for c0 in range(0, C, per):
        c1 = min(C, c0 + per)
        a, b = int(starts_h[c0]), int(starts_h[c1])
        st = xp.asarray(starts_h[c0:c1 + 1] - a)
        nc = c1 - c0
        set_of = xp.repeat(xp.arange(nc), xp.diff(st))
        Xc, nc_rows = X[a:b], b - a
        key = set_of * N1 + nest[a:b]
        chosen = xp.flatnonzero(yb[a:b])                 # one per set, in set order
        wc = xp.ones(nc) if w is None else w[a:b][chosen]
        oc = None if off is None else off[a:b]
        ar = xp.arange(nc)
        for t in range(T):
            th = THd[t]
            lam_h = np.ones(N1)
            free = np.flatnonzero(lam_of >= 0)
            lam_h[free] = TH[t, K + lam_of[free]]
            if (lam_h <= 0).any():
                ll[t] = -np.inf
                continue
            lam = xp.asarray(lam_h)
            V = Xc @ th[:K]
            if oc is not None:
                V = V + oc
            x = V / lam[nest[a:b]]
            mx = xp.full(nc * N1, -xp.inf)
            _scatter_max(xp, mx, key, x)
            e = xp.exp(x - mx[key])
            s = xp.zeros(nc * N1)
            su = xp.zeros(nc * N1)
            sd = xp.zeros((nc * N1, P))
            _scatter_add(xp, s, key, e)
            _scatter_add(xp, su, key, e * V)
            dUr = xp.zeros((nc_rows, P))
            dUr[:, :K] = Xc * e[:, None]
            _scatter_add(xp, sd, key, dUr)
            mx, s, su = mx.reshape(nc, N1), s.reshape(nc, N1), su.reshape(nc, N1)
            sd = sd.reshape(nc, N1, P)
            IV = xp.zeros((nc, N1))
            U = xp.zeros((nc, N1))
            dIV = xp.zeros((nc, N1, P))
            dU = xp.zeros((nc, N1, P))
            for n in [*tree.order.tolist(), M]:
                has = s[:, n] > 0
                sn = xp.where(has, s[:, n], 1.0)
                ln = float(lam_h[n])
                IV[:, n] = xp.where(has, mx[:, n] + xp.log(sn), 0.0)
                ub = su[:, n] / sn
                dIV[:, n] = sd[:, n] / sn[:, None] / ln
                j = int(lam_of[n])
                if j >= 0:
                    dIV[:, n, K + j] -= ub / (ln * ln)
                if n == M:
                    break
                U[:, n] = ln * IV[:, n]
                dU[:, n] = ln * dIV[:, n]
                if j >= 0:
                    dU[:, n, K + j] += IV[:, n]
                # into the parent's running log-sum-exp, where the nest is present
                q = int(parent[n])
                xn = U[:, n] / lam[q]
                old = s[:, q] > 0
                new_mx = xp.where(has, xp.where(old, xp.maximum(mx[:, q], xn), xn), mx[:, q])
                f_old = xp.where(old, xp.exp(mx[:, q] - new_mx), 0.0)
                f_new = xp.where(has, xp.exp(xn - new_mx), 0.0)
                s[:, q] = s[:, q] * f_old + f_new
                su[:, q] = su[:, q] * f_old + f_new * U[:, n]
                sd[:, q] = sd[:, q] * f_old[:, None] + f_new[:, None] * dU[:, n]
                mx[:, q] = new_mx
            # the chosen alternatives' paths
            cur = nest[a:b][chosen]
            uc = V[chosen]
            g = xp.zeros((nc, P))
            g[:, :K] = Xc[chosen]
            lp = xp.zeros(nc)
            du = xp.zeros((nc, P))
            live = xp.ones(nc, dtype=bool)
            while bool(live.any()):
                lc = lam[cur]
                lp += xp.where(live, uc / lc - IV[ar, cur], 0.0)
                du += xp.where(live[:, None], g / lc[:, None] - dIV[ar, cur], 0.0)
                jc = lamof_d[cur]
                hit = live & (jc >= 0)
                if bool(hit.any()):
                    rows = xp.flatnonzero(hit)
                    du[rows, K + jc[rows]] -= (uc / (lc * lc))[rows]
                live = live & (cur != M)
                uc = xp.where(live, U[ar, xp.minimum(cur, M)], uc)
                g = xp.where(live[:, None], dU[ar, xp.minimum(cur, M)], g)
                cur = xp.where(live, par_d[cur], cur)
            if scores:
                S_all.append(du)
                break
            ll[t] += float(wc @ lp)
            G[t] += _host(xp, wc @ du)
            if t < n_bhhh:
                B[t] += _host(xp, (du * wc[:, None]).T @ du)
    if scores:
        return xp.concatenate(S_all, axis=0) if S_all else xp.zeros((0, P))
    return ll, G, B
