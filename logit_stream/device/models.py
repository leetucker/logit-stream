"""The models' Newton sums a block at a time (log-likelihood, gradient,
information): multinomial, ordered (logit and probit), conditional, and
conditional fixed-effects logit (its recursion vectorized over the units of
each length), and Firth's adjustment."""

from __future__ import annotations

from ._common import _cdf_parts, _gram, _segmax, _segsum


def accumulate(xp, model, theta, X, y, w, starts, hessian, offset=None):
    """`model.accumulate` on the device: the log-likelihood (a device
    scalar), the gradient and (with `hessian`) the information, all device
    arrays."""
    kind = model.kind
    if kind == "mlogit":
        return _mnl(xp, model, theta, X, y, w, hessian)
    if kind == "ologit":
        return _ord(xp, model, theta, X, y, w, hessian, offset)
    if kind == "cfelogit":
        return _cfe(xp, theta, X, y, w, starts, hessian, offset)
    return _cond(xp, theta, X, y, w, starts, hessian, offset)



def _mnl(xp, model, theta, X, y, w, hessian):
    J1, K = model.J - 1, model.K
    E = X @ theta.reshape(J1, K).T
    m = xp.maximum(E.max(axis=1), 0.0)
    ex = xp.exp(E - m[:, None])
    den = xp.exp(-m) + ex.sum(axis=1)
    P = ex / den[:, None]
    yi = y.astype(xp.int64)
    pos = yi > 0
    rows = xp.arange(len(yi))
    lp = -(m + xp.log(den)) + xp.where(pos, E[rows, xp.maximum(yi - 1, 0)], 0.0)
    ll = lp.sum() if w is None else lp @ w
    R = -P
    R[rows[pos], yi[pos] - 1] += 1.0
    if w is not None:
        R = R * w[:, None]
    g = (X.T @ R).T.ravel()
    if not hessian:
        return ll, g, None
    Pw = P if w is None else P * w[:, None]
    H = xp.empty((J1 * K, J1 * K))
    for j in range(J1):
        sj = slice(j * K, (j + 1) * K)
        H[sj, sj] = _gram(X, Pw[:, j] * ((den - ex[:, j]) / den))
        for k in range(j + 1, J1):
            sk = slice(k * K, (k + 1) * K)
            Hjk = -_gram(X, Pw[:, j] * P[:, k])
            H[sj, sk] = Hjk
            H[sk, sj] = Hjk
    return ll, g, H



def firth_adjustment(xp, model, theta, X, w, A):
    """`_Multinomial.firth_adjustment` on the device: a block's part of the
    gradient of log|I| / 2, given A = I^-1 (a device array)."""
    J1, K = model.J - 1, model.K
    E = X @ theta.reshape(J1, K).T
    m = xp.maximum(E.max(axis=1), 0.0)
    ex = xp.exp(E - m[:, None])
    P = ex / (xp.exp(-m) + ex.sum(axis=1))[:, None]
    Q = xp.empty((len(X), J1, J1))
    for j in range(J1):
        for l in range(j, J1):
            q = ((X @ A[j * K:(j + 1) * K, l * K:(l + 1) * K]) * X).sum(axis=1)
            Q[:, j, l] = q
            Q[:, l, j] = q
    v = xp.einsum("ijj->ij", Q) - 2 * xp.einsum("ijl,il->ij", Q, P)
    C = P * v - P * (P * v).sum(axis=1, keepdims=True)
    if w is not None:
        C = C * w[:, None]
    return 0.5 * (X.T @ C).T.ravel()



def _ord_rows(xp, model, theta, X, y, w, hessian, offset):
    """The ordered model's per-row pieces, as `_Ordered.rows`: the
    log-likelihood terms, d l / d eta (`s`), the cutpoints' scores, and with
    `hessian` the information in eta (`h`) and with the cutpoints (`Q`), all
    weighted."""
    K, C = model.K, model.C
    eta = X @ theta[:K]
    if offset is not None:
        eta = eta + offset
    c = theta[K:K + model.n_cuts]
    if model.fix_first:
        c = xp.concatenate([xp.zeros(1), c])
    big = xp.asarray([1e300])
    cp = xp.concatenate([-big, c, big])
    yi = y.astype(xp.int64)
    a, b = cp[yi + 1] - eta, cp[yi] - eta
    Fa, Sa, fa, fpa = _cdf_parts(xp, model.link.name, a)
    Fb, Sb, fb, fpb = _cdf_parts(xp, model.link.name, b)
    P = xp.where(b > 0, Sb - Sa, Fa - Fb)
    ga, gb = fa / P, -fb / P
    wt = xp.ones(len(yi)) if w is None else w
    r = {"yi": yi, "lp": xp.log(P), "s": -(ga + gb) * wt, "ga": ga * wt, "gb": gb * wt}
    if hessian:
        haa = fpa / P - ga ** 2
        hbb = -fpb / P - gb ** 2
        hab = -ga * gb
        up, lo = yi < C, yi > 0
        Q = xp.zeros((len(yi), C))
        iu, il = xp.nonzero(up)[0], xp.nonzero(lo)[0]
        Q[iu, yi[iu]] += (wt * (haa + hab))[iu]
        Q[il, yi[il] - 1] += (wt * (hab + hbb))[il]
        r.update(h=wt * xp.maximum(-(haa + 2 * hab + hbb), 0.0), Q=Q, haa=-(wt * haa),
                 hbb=-(wt * hbb), hab=-(wt * hab))
    return r



def _ord_assemble(xp, model, r, X, w, hessian):
    K, C = model.K, model.C
    yi = r["yi"]
    ll = r["lp"].sum() if w is None else r["lp"] @ w
    up, lo = yi < C, yi > 0
    gcut = (xp.bincount(yi[up], weights=r["ga"][up], minlength=C)[:C]
            + xp.bincount(yi[lo] - 1, weights=r["gb"][lo], minlength=C)[:C])
    g = xp.concatenate([X.T @ r["s"], gcut])
    H = None
    if hessian:
        H = xp.empty((K + C, K + C))
        H[:K, :K] = _gram(X, r["h"])
        Hbc = X.T @ r["Q"]
        H[:K, K:] = Hbc
        H[K:, :K] = Hbc.T
        dcc = (xp.bincount(yi[up], weights=r["haa"][up], minlength=C)[:C]
               + xp.bincount(yi[lo] - 1, weights=r["hbb"][lo], minlength=C)[:C])
        both = up & lo
        offd = xp.bincount(yi[both] - 1, weights=r["hab"][both], minlength=C)[:C]
        Hcc = xp.diag(dcc)
        idx = xp.arange(C - 1)
        Hcc[idx, idx + 1] = offd[:C - 1]
        Hcc[idx + 1, idx] = offd[:C - 1]
        H[K:, K:] = Hcc
    if model.fix_first:
        keep = xp.concatenate([xp.arange(K), xp.arange(K + 1, K + C)])
        g = g[keep]
        H = H[keep][:, keep] if H is not None else None
    return ll, g, H



def _ord(xp, model, theta, X, y, w, hessian, offset):
    r = _ord_rows(xp, model, theta, X, y, w, hessian, offset)
    return _ord_assemble(xp, model, r, X, w, hessian)



def _cond(xp, theta, X, y, w, starts, hessian, offset):
    starts = starts.astype(xp.int64)
    sizes = xp.diff(starts)
    eta = X @ theta
    if offset is not None:
        eta = eta + offset
    m = _segmax(xp, eta, starts)
    e = xp.exp(eta - xp.repeat(m, sizes))
    S = _segsum(xp, e, starts)
    p = e / xp.repeat(S, sizes)
    yf = y.astype(xp.float64)
    lc = _segsum(xp, yf * eta, starts) - m - xp.log(S)
    wc = None if w is None else w[starts[:-1]]
    ll = lc.sum() if wc is None else lc @ wc
    r = yf - p
    if w is not None:
        r = r * w
    g = X.T @ r
    if not hessian:
        return ll, g, None
    H = _gram(X, p if w is None else p * w)
    Xbar = _segsum(xp, X * p[:, None], starts)
    H = H - _gram(Xbar, xp.ones(len(S)) if wc is None else wc)
    return ll, g, H



CFE_CHUNK_BYTES = 256e6     # the conditional FE logit's working arrays, per batch of units



def _cfe(xp, theta, X, y, w, starts, hessian, offset, unit_scores=False):
    """Chamberlain's conditional likelihood (`models._cfe_units`) vectorized
    over units: the units of each length T together, in batches, their rows
    gathered into (units, T, P) and the recursion run over the T rows and,
    at once, over j = 1..k (each j from the previous row's j - 1). Each unit
    with k > T / 2 is fitted as its complement and exp(eta) is scaled by its
    largest, as there; the units of one length share the largest k among
    them (the e_j beyond a unit's own k are not used). With `unit_scores`,
    each unit's unweighted score instead (units x P, 0 for those without
    variation)."""
    starts = starts.astype(xp.int64)
    P = X.shape[1]
    eta = X @ theta
    if offset is not None:
        eta = eta + offset
    sizes = xp.diff(starts)
    yb = (y > 0).astype(xp.float64)
    k = _segsum(xp, yb, starts)
    live = (k > 0) & (k < sizes)
    sgn = xp.where(2 * k > sizes, -1.0, 1.0)
    kk = xp.where(sgn < 0, sizes - k, k)
    ll = xp.zeros(())
    g = xp.zeros(P)
    H = xp.zeros((P, P)) if hessian else None
    Su = xp.zeros((len(sizes), P)) if unit_scores else None
    for T in [int(t) for t in xp.unique(sizes[live]).tolist()]:
        units = xp.flatnonzero(live & (sizes == T))
        kmax = int(kk[units].max())
        per_unit = 8 * ((kmax + 1) * (4 * P * P * hessian + 2 * P + 1) + T * (2 * P + 4))
        step = max(1, int(CFE_CHUNK_BYTES // per_unit))
        for a in range(0, len(units), step):
            uu = units[a:a + step]
            n = len(uu)
            rows = starts[uu][:, None] + xp.arange(T)[None, :]
            s = sgn[uu]
            Xs = X[rows] * s[:, None, None]                 # (n, T, P)
            es = eta[rows] * s[:, None]
            ys = xp.where(s[:, None] > 0, yb[rows], 1.0 - yb[rows])
            c = es.max(axis=1)
            z = xp.exp(es - c[:, None])
            e = xp.zeros((n, kmax + 1))
            e[:, 0] = 1.0
            G = xp.zeros((n, kmax + 1, P))
            Hm = xp.zeros((n, kmax + 1, P, P)) if hessian else None
            for t in range(T):
                zt, xt = z[:, t][:, None], Xs[:, t, :]
                e0, G0 = e[:, :-1], G[:, :-1]
                # (each right-hand side is formed before its in-place add: the
                # previous row's values, as the recursion's descending j has them)
                if hessian:
                    xG = xt[:, None, :, None] * G0[:, :, None, :]
                    xx = xt[:, :, None] * xt[:, None, :]
                    Hm[:, 1:] += zt[:, :, None, None] * (
                        Hm[:, :-1] + xG + xG.transpose(0, 1, 3, 2)
                        + xx[:, None] * e0[:, :, None, None])
                G[:, 1:] += zt[:, :, None] * (G0 + xt[:, None, :] * e0[:, :, None])
                e[:, 1:] += zt * e0
            kc = kk[uu].astype(xp.int64)
            ar = xp.arange(n)
            ek = e[ar, kc]
            m = G[ar, kc] / ek[:, None]
            if unit_scores:
                Su[uu] = (ys[:, :, None] * Xs).sum(axis=1) - m
                continue
            wu = xp.ones(n) if w is None else w[starts[uu]]
            lu = (ys * es).sum(axis=1) - xp.log(ek) - kk[uu] * c
            ll = ll + lu @ wu
            g += wu @ ((ys[:, :, None] * Xs).sum(axis=1) - m)
            if hessian:
                Hk = Hm[ar, kc] / ek[:, None, None] - m[:, :, None] * m[:, None, :]
                H += xp.einsum("n,npq->pq", wu, Hk)
    return Su if unit_scores else (ll, g, H)
