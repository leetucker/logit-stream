"""The random intercepts' data pass (`random_effects/model.py`): each unit's
sums and cross blocks, and the exact Laplace terms, by scatter-adds over
the rows' unit codes (computed on the host as each block is copied)."""

from __future__ import annotations

import numpy as np

from ._common import _expit, _gram, _host, _scatter_add
from .models import _ord_assemble, _ord_rows
from .store import DeviceStore


def _logistic_derivs(xp, z):
    """The logistic's F and its first three derivatives at z (0 where z is
    +-1e300, the open ends)."""
    F = _expit(xp, z)
    f = F * (1 - F)
    fin = xp.abs(z) < 1e290
    return (F, xp.where(fin, f, 0.0), xp.where(fin, f * (1 - 2 * F), 0.0),
            xp.where(fin, f * (1 - 6 * F + 6 * F * F), 0.0))



def _re_mnl(xp, th, Ar, X, y, wt, u, P, M, d, tree, of_unit, acc, Vall):
    n, K, J = len(y), X.shape[1], M
    E = xp.empty((n, J))
    E[:, 0] = 0.0
    E[:, 1:] = X @ th.reshape(J - 1, K).T
    E += Ar
    mx = E.max(axis=1)
    ex = xp.exp(E - mx[:, None])
    den = ex.sum(axis=1)
    Pr = ex / den[:, None]
    yi = y.astype(xp.int64)
    rr = xp.arange(n)
    lp = E[rr, yi] - mx - xp.log(den)
    R = -Pr
    R[rr, yi] += 1.0
    R = R * wt[:, None]
    acc.gt += (X.T @ R[:, 1:]).T.ravel()
    _scatter_add(xp, acc.ga, u, R)
    Pw = Pr * wt[:, None]
    for j in range(1, J):
        sj = slice((j - 1) * K, j * K)
        acc.Htt[sj, sj] += _gram(X, Pw[:, j] * ((den - ex[:, j]) / den))
        for k in range(j + 1, J):
            sk = slice((k - 1) * K, k * K)
            Hjk = _gram(X, Pw[:, j] * Pr[:, k])
            acc.Htt[sj, sk] -= Hjk
            acc.Htt[sk, sj] -= Hjk
    W = -(Pw[:, :, None] * Pr[:, None, :])
    idx = xp.arange(J)
    W[:, idx, idx] += Pw
    _scatter_add(xp, acc.Haa, u, W)
    step = max(1, int(64e6 // (8 * J * max(P, 1))))
    for s0 in range(0, n, step):
        sl = slice(s0, min(n, s0 + step))
        vals = (W[sl, :, 1:, None] * X[sl, None, None, :]).reshape(-1, J, P)
        if tree:
            _scatter_add(xp, acc.B, u[sl], vals)
        else:
            for ou, Bg in zip(of_unit, acc.Bf):
                _scatter_add(xp, Bg, ou[u[sl]], vals)
    if Vall is not None:
        # the exact Laplace terms: c_i = (diag p - p p')(diag V_u - 2 V_u p),
        # into sum c_i (x) x_i, and each unit's information derivatives
        p = Pr[:, 1:]
        eye, di = xp.eye(d), xp.arange(d)
        for s0 in range(0, n, 20_000):
            sl = slice(s0, min(n, s0 + 20_000))
            ps, us, ws = p[sl], u[sl], wt[sl]
            Dm = -ps[:, :, None] * ps[:, None, :]
            Dm[:, di, di] += ps
            V = Vall[us]
            v = xp.diagonal(V, axis1=1, axis2=2) - 2 * xp.einsum("nij,nj->ni", V, ps)
            c = xp.einsum("nij,nj->ni", Dm, v) * ws[:, None]
            acc.Dt += (X[sl].T @ c).T.ravel()
            Tr = (eye[None, :, :, None] * Dm[:, :, None, :]
                  - Dm[:, :, None, :] * ps[:, None, :, None]
                  - ps[:, :, None, None] * Dm[:, None, :, :]) * ws[:, None, None, None]
            _scatter_add(xp, acc.T, us, Tr)
    return lp @ wt



def _re_ord(xp, m, th, Ar, X, y, w, wt, u, U, P, tree, of_unit, acc, Vall):
    r = _ord_rows(xp, m, th, X, y, w, True, Ar[:, 0])
    ll, g, H = _ord_assemble(xp, m, r, X, w, True)
    acc.gt += g
    acc.Htt += H
    acc.ga[:, 0] += xp.bincount(u, weights=r["s"], minlength=U)[:U]
    acc.Haa[:, 0, 0] += xp.bincount(u, weights=r["h"], minlength=U)[:U]
    Q = r["Q"][:, 1:] if m.fix_first else r["Q"]
    Bv = xp.concatenate([X * r["h"][:, None], Q], axis=1)
    if tree:
        Bt = xp.zeros((U, P))
        _scatter_add(xp, Bt, u, Bv)
        acc.B[:, 0, :] += Bt
    else:
        for ou, Bg in zip(of_unit, acc.Bf):
            Bt = xp.zeros((Bg.shape[0], P))
            _scatter_add(xp, Bt, ou[u], Bv)
            Bg[:, 0, :] += Bt
    if Vall is not None:
        # the exact Laplace terms: the information's derivatives in eta and in
        # the row's two cutpoints (fe._ordered_info_derivs)
        C = m.C
        c_ = th[m.K:m.K + m.n_cuts]
        if m.fix_first:
            c_ = xp.concatenate([xp.zeros(1), c_])
        big = xp.asarray([1e300])
        full = xp.concatenate([-big, c_, big])
        yi = y.astype(xp.int64)
        eta = X @ th[:m.K] + Ar[:, 0]
        upz, loz = full[yi + 1] - eta, full[yi] - eta
        Fu, fu, f1u, f2u = _logistic_derivs(xp, upz)
        Fl, fl, f1l, f2l = _logistic_derivs(xp, loz)
        Pp = xp.maximum(xp.where(loz > 0, _expit(xp, -loz) - _expit(xp, -upz), Fu - Fl),
                        1e-300)
        D1, D2 = fu - fl, f1u - f1l
        dwu = (2 * D1 * f1u / Pp ** 2 - 2 * D1 ** 2 * fu / Pp ** 3 - f2u / Pp
               + D2 * fu / Pp ** 2)
        dwl = (-2 * D1 * f1l / Pp ** 2 + 2 * D1 ** 2 * fl / Pp ** 3 + f2l / Pp
               - D2 * fl / Pp ** 2)
        dwe = -(dwu + dwl)
        vi = Vall[u, 0, 0]
        dcut = xp.zeros(C)
        upm, lom = yi < C, yi >= 1
        dcut += xp.bincount(yi[upm], weights=(vi * dwu * wt)[upm], minlength=C)[:C]
        dcut += xp.bincount(yi[lom] - 1, weights=(vi * dwl * wt)[lom], minlength=C)[:C]
        if m.fix_first:
            dcut = dcut[1:]
        acc.Dt += xp.concatenate([X.T @ (vi * dwe * wt), dcut])
        acc.T[:, 0, 0, 0] += xp.bincount(u, weights=dwe * wt, minlength=U)[:U]
    return ll



def re_pass(xp, re, driver, theta, extra):
    """`_RandomEffects._data_pass` on the device: the log-likelihood, the
    coefficients' gradient and information, each unit's gradient and
    information for its linear predictor, their information with the
    coefficients (per unit for nested groupings, per group of each grouping
    when they cross), and with `extra` the exact Laplace terms (the
    coefficients' Dt and each unit's d x d x d derivatives T), the unit sums
    by scatter-adds over the rows' unit codes. Returned on the host, in the
    namespace `_data_pass` returns. The unit codes are computed on the host
    as each block is copied over."""
    from types import SimpleNamespace
    m = re.model
    P, U, M, d = m.P, re.U, re.m, re.d
    th = xp.asarray(theta[:P], dtype=xp.float64)
    Au = xp.asarray(re._unit_effects(theta), dtype=xp.float64)
    keep_u = xp.asarray(re.keep_u)
    tree = re.kind == "tree"
    of_unit = None if tree else [xp.asarray(f.of_unit) for f in re.factors]
    acc = SimpleNamespace(gt=xp.zeros(P), Htt=xp.zeros((P, P)), ga=xp.zeros((U, M)),
                          Haa=xp.zeros((U, M, M)),
                          B=xp.zeros((U, M, P)) if tree else None,
                          Bf=None if tree else [xp.zeros((f.G, M, P)) for f in re.factors],
                          Dt=xp.zeros(P) if extra else None,
                          T=xp.zeros((U, d, d, d)) if extra else None)
    Vall = None
    if extra:
        lv = np.flatnonzero(re.lv)
        Vall = xp.asarray(re._Vu[:, lv][:, :, lv], dtype=xp.float64)
    cols = None if len(m.cols) == driver.K0 else xp.asarray(m.cols)
    if driver._dstore is None:
        driver._dstore = DeviceStore(re.store, driver.device,
                                     extra=lambda blk: {"u": re._units(blk)})
    ll = 0.0
    for blk in driver._dstore:
        u = blk["u"].astype(xp.int64)
        keep = keep_u[u]
        sel = None if bool(keep.all()) else xp.nonzero(keep)[0]
        X = blk["X"] if cols is None else blk["X"][:, cols]
        X, y, w, off, u = (None if a is None else a if sel is None else a[sel]
                           for a in (X, blk["y"], blk.get("w"), blk.get("off"), u))
        if not len(y):
            continue
        wt = xp.ones(len(y)) if w is None else w
        Ar = Au[u] if off is None else Au[u] + off[:, None]
        if m.kind == "mlogit":
            ll = ll + _re_mnl(xp, th, Ar, X, y, wt, u, P, M, d, tree, of_unit, acc, Vall)
        else:
            ll = ll + _re_ord(xp, m, th, Ar, X, y, w, wt, u, U, P, tree, of_unit, acc, Vall)
    D = SimpleNamespace(theta=np.array(theta, copy=True), ll=float(ll), gt=_host(xp, acc.gt),
                        Htt=_host(xp, acc.Htt), ga=_host(xp, acc.ga), Haa=_host(xp, acc.Haa),
                        extra=extra, Dt=_host(xp, acc.Dt), T=_host(xp, acc.T),
                        Vu=re._Vu if extra else None)
    if tree:
        D.B = _host(xp, acc.B)
    else:
        D.Bf = [_host(xp, b) for b in acc.Bf]
    return D
