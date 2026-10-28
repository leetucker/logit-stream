"""The robust and clustered variances' middle matrices: each model's scores
on the device (concentrated for an absorbed fixed effect, profiled for
random intercepts), then S'S and the clusters' sums."""

from __future__ import annotations

from ._common import _host, _scatter_add, _segmax, _segsum
from .models import _cfe, _ord_rows
from .store import DeviceStore

MEAT_CHUNK_BYTES = 64e6         # a sub-chunk's rows x levels x P gathers, for the FE meats



def meat_sums(xp, S, w, cl, weights_type, Mt, U, clusters):
    """`_Driver._add_meat` on the device: S' S into Mt (weighted as
    `weights_type` says) and each clustering's sums of S by cluster code
    into U[name] (G x P), all device arrays."""
    if Mt is not None:
        if w is None:
            Ss = S
        elif weights_type == "fweights":
            Ss = S * xp.sqrt(w)[:, None]
        else:
            Ss = S * w[:, None]
        Mt += Ss.T @ Ss
    if U:
        Sw = S if w is None else S * w[:, None]
        for name, (col, _) in clusters.items():
            _scatter_add(xp, U[name], cl[:, col].astype(xp.int64), Sw)



def to_host(xp, Mt, U):
    return (None if Mt is None else _host(xp, Mt)), {k: _host(xp, v) for k, v in U.items()}



def _mnl_resid(xp, th, X, y, J, A=None):
    """The multinomial rows' residuals 1[y = j] - p_j over all J levels (the
    base first), with the effects A (rows x J) added to the predictors."""
    n = len(y)
    E = xp.zeros((n, J))
    E[:, 1:] = X @ th.reshape(J - 1, -1).T
    if A is not None:
        E = E + A
    ex = xp.exp(E - E.max(axis=1)[:, None])
    R = -ex / ex.sum(axis=1)[:, None]
    R[xp.arange(n), y.astype(xp.int64)] += 1.0
    return R



def _ord_scores(xp, m, th, X, y, offset):
    """`_Ordered.scores` on the device, and the rows' d l / d eta."""
    K, C = m.K, m.C
    r = _ord_rows(xp, m, th, X, y, None, False, offset)
    yi = r["yi"]
    S = xp.zeros((len(yi), K + C))
    S[:, :K] = X * r["s"][:, None]
    iu, il = xp.nonzero(yi < C)[0], xp.nonzero(yi > 0)[0]
    S[iu, K + yi[iu]] += r["ga"][iu]
    S[il, K + yi[il] - 1] += r["gb"][il]
    if m.fix_first:
        S = xp.concatenate([S[:, :K], S[:, K + 1:]], axis=1)
    return S, r["s"]



def scores(xp, model, theta, X, y, starts, offset=None):
    """`model.scores` on the device: each unit's score (rows; choice sets
    for the conditional logit; units for the conditional fixed-effects
    logit), unweighted."""
    kind = model.kind
    if kind == "mlogit":
        R = _mnl_resid(xp, theta, X, y, model.J)
        return (R[:, 1:, None] * X[:, None, :]).reshape(len(y), -1)
    if kind == "ologit":
        return _ord_scores(xp, model, theta, X, y, offset)[0]
    if kind == "cfelogit":
        return _cfe(xp, theta, X, y, None, starts, False, offset, unit_scores=True)
    starts = starts.astype(xp.int64)
    sizes = xp.diff(starts)
    eta = X @ theta
    if offset is not None:
        eta = eta + offset
    mx = _segmax(xp, eta, starts)
    e = xp.exp(eta - xp.repeat(mx, sizes))
    p = e / xp.repeat(_segsum(xp, e, starts), sizes)
    return _segsum(xp, X * (y.astype(xp.float64) - p)[:, None], starts)



def plain_meat(xp, driver, theta, weights_type, clusters, hetero):
    """`_Driver.meat` on the device, from the blocks there (one pass)."""
    m = driver.model
    P = len(theta)
    th = xp.asarray(theta, dtype=xp.float64)
    Mt = xp.zeros((P, P)) if hetero else None
    U = {name: xp.zeros((G, P)) for name, (_, G) in clusters.items()}
    cols = None if len(m.cols) == driver.K0 else xp.asarray(m.cols)
    if driver._dstore is None:
        driver._dstore = DeviceStore(driver.store, driver.device)
    for blk in driver._dstore:
        X = blk["X"] if cols is None else blk["X"][:, cols]
        st = blk.get("starts")
        S = scores(xp, m, th, X, blk["y"], st, blk.get("off"))
        w, cl = blk.get("w"), blk.get("cl")
        if m.unit == "cases":
            first = st[:-1].astype(xp.int64)
            w = None if w is None else w[first]
            cl = None if cl is None else cl[first]
        meat_sums(xp, S, w, cl, weights_type, Mt, U, clusters)
    return to_host(xp, Mt, U)



def _row_chunks(n, width):
    """Sub-chunks of n rows for gathers of `width` floats a row."""
    step = max(256, int(MEAT_CHUNK_BYTES // (8 * max(width, 1))))
    return [(a, min(n, a + step)) for a in range(0, n, step)]



def fe_meat(xp, fe, driver, theta, H, weights_type, clusters, hetero):
    """`_AbsorbedFE.meat` on the device: the rows' scores for b less the
    part the fixed effect absorbs, s_b - H_bg H_gg^-1 s_g."""
    m = fe.model
    H = fe._schur(H)
    P, J = m.P, fe.m
    th = xp.asarray(theta[:P], dtype=xp.float64)
    A = xp.asarray(fe._effects(theta), dtype=xp.float64)
    BD = xp.asarray(H.BD, dtype=xp.float64)
    keep_g = xp.asarray(fe.keep_g)
    free = xp.asarray(fe.free, dtype=xp.float64)
    Mt = xp.zeros((P, P)) if hetero else None
    U = {name: xp.zeros((G, P)) for name, (_, G) in clusters.items()}
    cols = None if len(m.cols) == driver.K0 else xp.asarray(m.cols)
    if driver._dstore is None:
        driver._dstore = DeviceStore(fe.store, driver.device)
    for blk in driver._dstore:
        f = blk["fe"].astype(xp.int64)
        sel = xp.nonzero(keep_g[f])[0]
        X = blk["X"] if cols is None else blk["X"][:, cols]
        X, y, w, off, cl, f = (None if a is None else a[sel]
                               for a in (X, blk["y"], blk.get("w"), blk.get("off"),
                                         blk.get("cl"), f))
        for a, b in _row_chunks(len(y), J * P):
            Xc, yc, fc = X[a:b], y[a:b], f[a:b]
            if m.kind == "mlogit":
                R = _mnl_resid(xp, th, Xc, yc, J, A[fc])
                St = (R[:, 1:, None] * Xc[:, None, :]).reshape(b - a, -1)
                S = St - xp.einsum("nj,njp->np", R * free[fc], BD[fc])
            else:
                o = A[fc, 0] if off is None else A[fc, 0] + off[a:b]
                St, s = _ord_scores(xp, m, th, Xc, yc, o)
                S = St - s[:, None] * BD[fc, 0, :]
            meat_sums(xp, S, None if w is None else w[a:b], None if cl is None else cl[a:b],
                      weights_type, Mt, U, clusters)
    return to_host(xp, Mt, U)



def re_meat(xp, re, driver, theta, H, weights_type, clusters):
    """`_RandomEffects.meat` on the device: the clusters' sums of the rows'
    profile scores for b, concentrated with respect to the fixed effect if
    there is one."""
    m = re.model
    P = m.P
    th = xp.asarray(theta[:P], dtype=xp.float64)
    Au = xp.asarray(re._unit_effects(theta), dtype=xp.float64)
    keep_u = xp.asarray(re.keep_u)
    fe = re.fixed_factor
    EF = fe_of = fe_free = None
    if fe is not None:
        F = re._full(H)
        k = len(re.factors) - 1
        EF = xp.asarray(F.Dinv[k] @ F.B[k], dtype=xp.float64)
        fe_of = xp.asarray(fe.of_unit)
        fe_free = xp.asarray(fe.free, dtype=xp.float64)
    U = {name: xp.zeros((G, P)) for name, (_, G) in clusters.items()}
    cols = None if len(m.cols) == driver.K0 else xp.asarray(m.cols)
    if driver._dstore is None:
        driver._dstore = DeviceStore(re.store, driver.device,
                                     extra=lambda blk: {"u": re._units(blk)})
    for blk in driver._dstore:
        u = blk["u"].astype(xp.int64)
        sel = xp.nonzero(keep_u[u])[0]
        X = blk["X"] if cols is None else blk["X"][:, cols]
        X, y, w, off, cl, u = (None if a is None else a[sel]
                               for a in (X, blk["y"], blk.get("w"), blk.get("off"),
                                         blk.get("cl"), u))
        for a, b in _row_chunks(len(y), re.m * P):
            Xc, yc, uc = X[a:b], y[a:b], u[a:b]
            Ar = Au[uc]
            if off is not None:
                Ar = Ar + off[a:b][:, None]
            if m.kind == "mlogit":
                R = _mnl_resid(xp, th, Xc, yc, re.J, Ar)
                S = (R[:, 1:, None] * Xc[:, None, :]).reshape(b - a, -1)
                if EF is not None:
                    sg = fe_of[uc]
                    S = S - xp.einsum("nj,njp->np", R * fe_free[sg], EF[sg])
            else:
                S, s = _ord_scores(xp, m, th, Xc, yc, Ar[:, 0])
                if EF is not None:
                    S = S - s[:, None] * EF[fe_of[uc], 0, :]
            meat_sums(xp, S, None if w is None else w[a:b], None if cl is None else cl[a:b],
                      weights_type, None, U, clusters)
    return to_host(xp, None, U)
