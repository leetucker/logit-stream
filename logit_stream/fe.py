"""One absorbed fixed effect for the multinomial and ordered logit.

With a fixed effect, each level g of the dimension (a PUMA, say) has its own
intercepts: one per outcome level for the multinomial logit (alpha_gj, with
the group's reference level at 0), one location shift for the ordered logit
(alpha_g, added to x'b, with cut1 fixed at 0 since the shifts carry the
location). There are G x (J - 1) or G of them, too many to put in the
information matrix, but each touches only its own group's rows, so the
information has an arrow shape:

    [ H_bb   H_ba ]      H_aa block diagonal, one J x J (or 1 x 1) block
    [ H_ab   H_aa ]      per group; H_ba one P x J block per group

and the Newton step solves it exactly by eliminating the group blocks (the
Schur complement): the step for b solves

    (H_bb - sum_g H_bg H_gg^-1 H_gb) db = g_b - sum_g H_bg H_gg^-1 g_g,

and each group's step follows from db alone. A Newton step is therefore
still one read of the rows, accumulating H_bb as without fixed effects plus
the per-group blocks (G x J x J and G x J x P, the latter the one that
grows: G = 2,400 PUMAs, J = 6 and P = 300 take 35 MB), and the variance of b
is the inverse of the reduced matrix, which is the b block of the inverse of
the whole information matrix. Rows need not be sorted by group.

Groups whose outcome never varies have infinite effects (in the multinomial
logit, a group with one level; in the ordered, a group all at the lowest or
the highest level), and their rows say nothing about b: they are dropped
before the fit, and recorded in diagnostics["separation"], as hdfe-stream
does for the binary logit. In the multinomial logit a level that some
group never has gets probability zero in that group (its effect is minus
infinity, held fixed rather than estimated).

The incidental-parameter problem applies: with few rows per group, b is
biased (as with fixed-effect logits generally). Absorbing a fixed effect is
meant for groups with many rows each.
"""

from __future__ import annotations

from types import SimpleNamespace

import numba as nb
import numpy as np
import polars as pl
from hdfe_stream.core.report import warn
from scipy.optimize import minimize
from scipy.special import xlogy

from .models import _gram, _mnl_info_sparse
from .newton import _chunks, _design, _Driver
from .store import CODE

_MAX_GROUP_GB = 8.0


@nb.njit(parallel=True, cache=True)
def _mnl_group_info(order, seg, segg, X, P, w, Haa, B):
    """Per-group information of the multinomial logit: Haa[g] = sum over the
    group's rows of w (diag(p) - p p') over all J levels' utilities, and
    B[g, l, (j-1) K + k] = sum of w x_k (p_j [j = l] - p_j p_l), the
    information between the effect of level l and the coefficient of x_k in
    equation j. Rows in `order`, grouped in segments `seg` of group `segg`."""
    J = P.shape[1]
    K = X.shape[1]
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
                    v = wi * ((pl_ if l == m else 0.0) - pl_ * P[i, m])
                    Haa[g, l, m] += v
                for j in range(1, J):
                    v = wi * ((P[i, j] if j == l else 0.0) - P[i, j] * pl_)
                    if v != 0.0:
                        base = (j - 1) * K
                        for k in range(K):
                            x = X[i, k]
                            if x != 0.0:
                                B[g, l, base + k] += v * x


def _segments(f):
    order = np.argsort(f, kind="stable")
    fs = f[order]
    st = np.flatnonzero(np.r_[True, fs[1:] != fs[:-1]]) if len(fs) else np.zeros(0, np.int64)
    return order, np.r_[st, len(fs)].astype(np.int64), fs[st].astype(np.int64)


def _group_sum(f, V, G):
    """Column sums of V within each group code f: G x c."""
    out = np.zeros((G, V.shape[1]))
    if not len(f):
        return out
    order, seg, segg = _segments(f)
    out[segg] += np.add.reduceat(V[order], seg[:-1], axis=0)
    return out


def _subset(ch, keep):
    return {k: (v[keep] if k not in ("starts",) else v) for k, v in ch.items()}


def _ordered_table_loglik(n, link):
    """The maximized log-likelihood of an ordered logit with one location
    per row of the count table `n` (G x J, weighted counts) and common
    cutpoints: sum n_gk log(F(c_k - a_g) - F(c_(k-1) - a_g)). Only the
    differences matter, so c_1 = 0; L-BFGS on the table, with the analytic
    gradient."""
    G, J = n.shape

    def unpack(v):
        c = np.r_[0.0, np.cumsum(np.exp(v[G:]))]           # J - 1 cutpoints, increasing
        return v[:G], c

    def f(v):
        a, c = unpack(v)
        U = c[None, :] - a[:, None]                         # G x (J - 1)
        Fu = link.cdf(U)
        fu = link.pdf(U, Fu, link.sf(U))
        Fp = np.c_[np.zeros(G), Fu, np.ones(G)]
        fp = np.c_[np.zeros(G), fu, np.zeros(G)]
        pr = np.maximum(Fp[:, 1:] - Fp[:, :-1], 1e-300)
        r = n / pr
        ll = float(np.sum(xlogy(n, pr)))
        ga = -np.sum(r * (fp[:, 1:] - fp[:, :-1]), axis=1)
        # d/dc_m: + r_(g,m) f_m - r_(g,m+1) f_m, for m = 1..J-1
        gc = np.sum((r[:, :-1] - r[:, 1:]) * fu, axis=0)
        # c_1 = 0, c_m = sum_(l<m) exp(v_l): dc_m / dv_l = exp(v_l) for l < m
        gv = np.array([gc[l + 1:].sum() * np.exp(v[G + l]) for l in range(J - 2)])
        return -ll, -np.r_[ga, gv]

    p = n.sum(axis=0).cumsum()[:-1] / n.sum()
    c0 = np.log(p / (1 - p))
    v0 = np.r_[np.full(G, -c0[0]), np.log(np.maximum(np.diff(c0), 1e-3))]
    res = minimize(f, v0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 10_000, "ftol": 1e-15, "gtol": 1e-9})
    return float(-res.fun)


class _AbsorbedFE:
    """The fixed effect's bookkeeping, passes and linear algebra for the
    Newton driver (see the module docstring)."""

    def __init__(self, model, p0, name, log, logger=None):
        self.model, self.p0, self.name = model, p0, name
        self.store = p0.store
        self.logger = logger
        G = p0.fe_tables[name].height
        J = model.J
        self.G, self.J = G, J
        log(f"fixed effect {name}: {G:,} levels; counting outcomes per level")
        wcnt = np.zeros(G * J)
        ncnt = np.zeros(G * J)
        for blk in self.store.iterate(["y", "fe", "w"]):
            key = blk["fe"] * J + blk["y"]
            ncnt += np.bincount(key, minlength=G * J)
            wcnt += np.bincount(key, weights=blk.get("w"), minlength=G * J)
        wcnt, ncnt = wcnt.reshape(G, J), ncnt.reshape(G, J)
        obs = ncnt > 0
        nlev = obs.sum(axis=1)
        if model.kind == "mlogit":
            drop = nlev < 2
        else:
            drop = (nlev == 1) & (obs[:, 0] | obs[:, -1])
        self.keep_g = ~drop
        self.obs = obs
        self.ncnt = ncnt
        self.wcnt = wcnt
        if model.kind == "mlogit":
            self.m = J
            self.ref = np.argmax(obs, axis=1)
            free = obs & (np.arange(J)[None, :] != self.ref[:, None])
            self.fixed = np.where(obs, 0.0, -np.inf)        # the reference, and absent levels
        else:
            self.m = 1
            free = np.ones((G, 1), bool)
            model.fix_first = True
        self.free = free & self.keep_g[:, None]
        n_drop = int(ncnt[drop].sum())
        self.separation = {"checked": True, "levels": int(drop.sum()), "observations": n_drop}
        if n_drop:
            warn(f"{n_drop:,} observations in {int(drop.sum()):,} levels of {name} dropped "
                 "because their outcome never varies (their effects would be infinite)",
                 logger)
            p0.n_obs -= n_drop
            p0.wsum -= float(wcnt[drop].sum())
            p0.counts = wcnt[self.keep_g].sum(axis=0)
            if p0.n_obs <= 0:
                raise ValueError(f"no observations left after dropping the levels of {name} "
                                 "whose outcome never varies")
            if np.any(p0.counts <= 0):
                gone = [lv for lv, c in zip(p0.levels, p0.counts) if c <= 0]
                raise ValueError(f"outcome levels {gone} occur only in levels of {name} whose "
                                 "outcome never varies")
        self.n_params = int(self.free.sum())
        gb = G * self.m * max(model.P, 1) * 8 * 2 / 1e9
        if gb > _MAX_GROUP_GB:
            raise MemoryError(
                f"absorbing {name} needs about {gb:.1f} GB for its per-level blocks "
                f"({G:,} levels x {self.m} x {model.P} parameters, twice); fit with fewer "
                "covariates, or a coarser fixed effect")

    def loglik_fe_only(self):
        """The log-likelihood of the model with the fixed effect and no
        covariates (for the within pseudo R-squared), from the weighted
        counts of each outcome level in each kept level of the fixed effect:
        for the multinomial logit each level's observed shares (closed
        form); for the ordered logit a location per level and common
        cutpoints, fitted on the G x J table in memory."""
        n = self.wcnt[self.keep_g]
        if self.model.kind == "mlogit":
            return float(np.sum(xlogy(n, n / n.sum(axis=1, keepdims=True))))
        return _ordered_table_loglik(n, self.model.link)

    # ------------------------------------------------------------ parameters
    def start(self):
        """The slopes at zero and each group's effects at its own sample
        shares (multinomial), or the marginal cutpoints with each group at
        -cut1 (ordered): the null model within each group."""
        m = self.model
        th = m.start(self.p0.counts, fe=True)
        if m.kind == "mlogit":
            with np.errstate(divide="ignore", invalid="ignore"):
                lc = np.log(self.ncnt) - np.log(self.ncnt[np.arange(self.G), self.ref])[:, None]
            A = np.where(self.free, lc, 0.0)
        else:
            A = np.full((self.G, 1), -m.marginal_cuts(self.p0.counts)[0])
        return np.r_[th, A.ravel()]

    def _effects(self, theta):
        P = self.model.P
        A = theta[P:].reshape(self.G, self.m)
        if self.model.kind == "mlogit":
            return np.where(self.free, A, self.fixed)
        return A

    # ------------------------------------------------------------ passes
    def numba_pass(self):
        """Whether a pass with the information does its heavy work in numba
        (the multinomial's group blocks, and its sparse slope block), so
        that BLAS should keep to one thread (hdfe_stream.core.utils.single_blas)."""
        return self.model.kind == "mlogit" and self.model.sparse(None)

    def pass_(self, driver, theta, hessian):
        if driver.device is not None:
            return self._device_pass(driver, theta, hessian)
        driver.passes += 1
        m = self.model
        P, G, J = m.P, self.G, self.m
        th = theta[:P]
        A = self._effects(theta)
        ll, gt, ga = 0.0, np.zeros(P), np.zeros((G, J))
        Htt = np.zeros((P, P)) if hessian else None
        Haa = np.zeros((G, J, J)) if hessian else None
        B = np.zeros((G, J, P)) if hessian else None
        rows = driver._rows()
        for blk in self.store:
            for ch in _chunks(blk, rows):
                keep = self.keep_g[ch["fe"]]
                if not keep.all():
                    ch = _subset(ch, keep)
                if not len(ch["y"]):
                    continue
                X = _design(ch, m.cols, driver.K0)
                y, w, f = ch["y"], ch.get("w"), ch["fe"]
                if m.kind == "mlogit":
                    l = self._mnl_chunk(th, A, X, y, w, f, gt, ga, Htt, Haa, B)
                else:
                    l = self._ord_chunk(th, A, X, y, w, f, gt, ga, Htt, Haa, B, ch.get("off"))
                if not np.isfinite(l):
                    return -np.inf, None, None
                ll += l
        return self._finish(ll, gt, ga, Htt, Haa, B, hessian)

    def _device_pass(self, driver, theta, hessian):
        """`pass_` with the sums formed on the GPU (device/)."""
        from . import device
        driver.passes += 1
        if driver._dstore is None:
            driver._dstore = device.DeviceStore(self.store, driver.device)
        ll, gt, ga, Htt, Haa, B = device.fe_pass(driver.device.xp, self, theta, driver._dstore,
                                                 hessian, driver.K0)
        if not np.isfinite(ll):
            return -np.inf, None, None
        return self._finish(ll, gt, ga, Htt, Haa, B, hessian)

    def _finish(self, ll, gt, ga, Htt, Haa, B, hessian):
        """The levels without a free effect (absent outcome levels, the
        reference, dropped levels) held: zero gradient, unit information."""
        ga *= self.free
        g = np.r_[gt, ga.ravel()]
        if not hessian:
            return ll, g, None
        nf = ~self.free
        Haa *= self.free[:, :, None]
        Haa *= self.free[:, None, :]
        gi, li = np.nonzero(nf)
        Haa[gi, li, li] = 1.0
        B *= self.free[:, :, None]
        return ll, g, SimpleNamespace(tt=Htt, aa=Haa, ta=B)

    def _mnl_chunk(self, th, A, X, y, w, f, gt, ga, Htt, Haa, B):
        n, K, J = len(y), X.shape[1], self.J
        E = np.empty((n, J))
        E[:, 0] = 0.0
        E[:, 1:] = X @ th.reshape(J - 1, K).T
        E += A[f]
        mx = E.max(axis=1)
        ex = np.exp(E - mx[:, None])
        den = ex.sum(axis=1)
        Pr = ex / den[:, None]
        r = np.arange(n)
        lp = E[r, y] - mx - np.log(den)
        l = float(lp.sum() if w is None else lp @ w)
        if not np.isfinite(l):
            return l
        R = -Pr
        R[r, y] += 1.0
        if w is not None:
            R *= w[:, None]
        gt += (X.T @ R[:, 1:]).T.ravel()
        ga += _group_sum(f, R, self.G)
        if Htt is not None and self.model.sparse(X):
            ww = np.ones(n) if w is None else w
            H = np.zeros_like(Htt)
            _mnl_info_sparse(np.ascontiguousarray(X), np.ascontiguousarray(Pr[:, 1:]), ww,
                             nb.get_num_threads(), H)
            for j in range(J - 1):
                for k in range(j + 1, J - 1):
                    H[k * K:(k + 1) * K, j * K:(j + 1) * K] = \
                        H[j * K:(j + 1) * K, k * K:(k + 1) * K].T
            Htt += H
            order, seg, segg = _segments(f)
            _mnl_group_info(order, seg, segg, np.ascontiguousarray(X), Pr, ww, Haa, B)
        elif Htt is not None:
            Pw = Pr if w is None else Pr * w[:, None]
            for j in range(1, J):
                sj = slice((j - 1) * K, j * K)
                Htt[sj, sj] += _gram(X, Pw[:, j] * ((den - ex[:, j]) / den))
                for k in range(j + 1, J):
                    sk = slice((k - 1) * K, k * K)
                    Hjk = _gram(X, Pw[:, j] * Pr[:, k])
                    Htt[sj, sk] -= Hjk
                    Htt[sk, sj] -= Hjk
            order, seg, segg = _segments(f)
            _mnl_group_info(order, seg, segg, np.ascontiguousarray(X), Pr,
                            np.ones(n) if w is None else w, Haa, B)
        return l

    def _ord_chunk(self, th, A, X, y, w, f, gt, ga, Htt, Haa, B, off=None):
        m = self.model
        o = A[f, 0] if off is None else A[f, 0] + off
        r = m.rows(th, X, y, w, offset=o, hessian=Htt is not None)
        l, g, H = m._assemble(r, X, y, w, Htt is not None)
        if not np.isfinite(l):
            return l
        gt += g
        ga[:, 0] += np.bincount(f, weights=r["s"], minlength=self.G)
        if Htt is not None:
            Htt += H
            Haa[:, 0, 0] += np.bincount(f, weights=r["h"], minlength=self.G)
            Q = r["Q"][:, 1:] if m.fix_first else r["Q"]
            B[:, 0, :] += _group_sum(f, np.column_stack([X * r["h"][:, None], Q]), self.G)
        return l

    # ------------------------------------------------------------ algebra
    def _schur(self, H):
        if getattr(H, "red", None) is None:
            G, J = self.G, self.m
            P = H.tt.shape[0]
            try:
                Dinv = np.linalg.inv(H.aa)
            except np.linalg.LinAlgError:
                Dinv = np.linalg.pinv(H.aa, hermitian=True)
            BD = Dinv @ H.ta                                  # G x J x P
            Bf = H.ta.reshape(G * J, P)
            red = H.tt - Bf.T @ BD.reshape(G * J, P)
            H.Dinv, H.BD, H.red = Dinv, BD, (red + red.T) / 2
        return H

    def reduced(self, H):
        """The information for b net of the fixed effects."""
        return self._schur(H).red

    def solve(self, H, g):
        H = self._schur(H)
        G, J = self.G, self.m
        P = H.tt.shape[0]
        gt, ga = g[:P], g[P:].reshape(G, J)
        Bf = H.ta.reshape(G * J, P)
        gred = gt - Bf.T @ (H.Dinv @ ga[..., None]).reshape(G * J)
        dt, pd = _Driver.solve(H.red, gred)
        da = (H.Dinv @ (ga - (Bf @ dt).reshape(G, J))[..., None])[..., 0] * self.free
        return np.r_[dt, da.ravel()], pd

    def vcov_beta(self, H):
        return _Driver.inverse(self.reduced(H))

    def sandwich(self, H, M):
        V = self.vcov_beta(H)
        out = V @ M @ V
        return (out + out.T) / 2

    def penalize(self, H, add):
        return SimpleNamespace(tt=H.tt + np.diag(add), aa=H.aa, ta=H.ta)

    def damp(self, H, lam):
        """H + lam diag(H), block by block (the padding of the fixed
        effects' blocks stays the identity)."""
        from .newton import _floored
        dt = _floored(np.abs(np.diag(H.tt)))
        da = _floored(np.abs(np.diagonal(H.aa, axis1=1, axis2=2))) * self.free
        aa = H.aa.copy()
        G, J = da.shape
        aa[:, np.arange(J), np.arange(J)] += lam * da
        return SimpleNamespace(tt=H.tt + lam * np.diag(dt), aa=aa, ta=H.ta)

    def subset(self, theta, g, H, idx):
        P_old = len(theta) - self.G * self.m
        keep = lambda v: np.r_[v[idx], v[P_old:]]           # noqa: E731
        H2 = SimpleNamespace(tt=H.tt[np.ix_(idx, idx)], aa=H.aa, ta=H.ta[:, :, idx])
        return keep(theta), keep(g), H2

    # ------------------------------------------------------------ inference
    def meat(self, driver, theta, H, weights_type, clusters, hetero):
        """The robust variances' middle matrices from the concentrated
        scores: each row's score for b less the part the fixed effects
        absorb, s_b - H_bg H_gg^-1 s_g (the influence of the row on b̂). On
        the device too (`device.fe_meat`) with device=."""
        if driver.device is not None:
            from . import device
            return device.fe_meat(driver.device.xp, self, driver, theta, H, weights_type,
                                  clusters, hetero)
        m = self.model
        H = self._schur(H)
        P, J = m.P, self.m
        th = theta[:P]
        A = self._effects(theta)
        Mt = np.zeros((P, P)) if hetero else None
        U = {name: np.zeros((Gc, P)) for name, (_, Gc) in clusters.items()}
        rows = max(256, int(driver.chunk_bytes // (8 * max(P * (J + 1), 1))))
        for blk in self.store:
            for ch in _chunks(blk, rows):
                keep = self.keep_g[ch["fe"]]
                if not keep.all():
                    ch = _subset(ch, keep)
                if not len(ch["y"]):
                    continue
                X = _design(ch, m.cols, driver.K0)
                y, f = ch["y"], ch["fe"]
                if m.kind == "mlogit":
                    n = len(y)
                    E = np.empty((n, self.J))
                    E[:, 0] = 0.0
                    E[:, 1:] = X @ th.reshape(self.J - 1, -1).T
                    E += A[f]
                    ex = np.exp(E - E.max(axis=1)[:, None])
                    R = -ex / ex.sum(axis=1)[:, None]
                    R[np.arange(n), y] += 1.0
                    St = (R[:, 1:, None] * X[:, None, :]).reshape(n, -1)
                    Sa = R * self.free[f]
                    S = St - np.einsum("nj,njp->np", Sa, H.BD[f])
                else:
                    o = A[f, 0] if ch.get("off") is None else A[f, 0] + ch["off"]
                    St = m.scores(th, X, y, None, offset=o)
                    s = m.rows(th, X, y, None, offset=o, hessian=False)["s"]
                    S = St - s[:, None] * H.BD[f, 0, :]
                driver._add_meat(S, ch.get("w"), ch.get("cl"), weights_type, Mt, U, clusters)
        return Mt, U

    # ------------------------------------------------------------ results
    def result_spec(self, theta):
        """The fixed-effect table (`fixef`), its effect columns, and the
        levels dropped for separation, for the result."""
        table = self.p0.fe_tables[self.name]
        A = self._effects(theta)
        name = self.name
        if self.model.kind == "mlogit":
            vals = {f"fe_{name}:{lv}": pl.Series(np.where(np.isfinite(A[:, j]), A[:, j], np.nan))
                    .fill_nan(None) for j, lv in enumerate(self.p0.levels)}
        else:
            vals = {f"fe_{name}": pl.Series(A[:, 0])}
        n = pl.Series(self.ncnt.sum(axis=1).astype(np.int64))
        full = table.with_columns(**vals, n_obs=n)
        keep = pl.Series(self.keep_g)
        out = {"fixef": {name: full.filter(keep).drop(CODE)},
               "effects": [(name, list(vals), "fixed")]}
        dropped = table.filter(~keep).drop(CODE)
        if dropped.height:
            out["dropped"] = {name: dropped}
        return out


# --------------------------------------------------------------------------
# random intercepts
# --------------------------------------------------------------------------

def _logistic_derivs(z):
    """F, F', F'' and F''' of the logistic distribution at z (infinite z
    allowed: there the derivatives are 0)."""
    from scipy.special import expit
    F = expit(z)
    with np.errstate(invalid="ignore"):
        f = F * (1 - F)
        f1 = f * (1 - 2 * F)
        f2 = f * (1 - 6 * F + 6 * F * F)
    fin = np.isfinite(z)
    return F, np.where(fin, f, 0.0), np.where(fin, f1, 0.0), np.where(fin, f2, 0.0)


def _ordered_info_derivs(cuts, eta, y):
    """For each row of the ordered logit, P(y) = F(b - eta) - F(a - eta)
    with a, b the row's lower and upper cutpoints: its information for the
    linear index, w = -d^2 log P / d eta^2, and the derivatives of w with
    respect to eta and to the upper and lower cutpoints (third derivatives
    of the log-likelihood, for the exact Laplace approximation)."""
    from scipy.special import expit
    full = np.r_[-np.inf, cuts, np.inf]
    up, lo = full[y + 1] - eta, full[y] - eta
    Fu, fu, f1u, f2u = _logistic_derivs(up)
    Fl, fl, f1l, f2l = _logistic_derivs(lo)
    with np.errstate(invalid="ignore", over="ignore"):
        P = np.where(lo > 0, expit(-lo) - expit(-up), Fu - Fl)
    P = np.maximum(P, 1e-300)
    D1, D2 = fu - fl, f1u - f1l
    w = D1 ** 2 / P ** 2 - D2 / P
    dw_up = 2 * D1 * f1u / P ** 2 - 2 * D1 ** 2 * fu / P ** 3 - f2u / P + D2 * fu / P ** 2
    dw_lo = -2 * D1 * f1l / P ** 2 + 2 * D1 ** 2 * fl / P ** 3 + f2l / P - D2 * fl / P ** 2
    return w, -(dw_up + dw_lo), dw_up, dw_lo


def _sym(A):
    return (A + A.T) / 2
