"""Random slopes by region for the conditional logit (experimental):
``clogit_stream("chosen ~ dist | estab", ..., random={"cz": "dist"})``.

**The model.** Each level g of a grouping (a region) has its own
deviation b_g ~ N(0, Sigma) in the coefficients of the covariates z (a
subset of the formula's), independent across regions:

    U_r = x_r'beta + z_r'b_{g(r)} + sum_d alpha^d[lev_d(r)] + e_r,

with r an alternative of a choice set. A row's region may be its choice
set's (the chooser's: a worker's home commuting zone; every alternative of
a set then shares it) or its own (the alternative's: an establishment's
commuting zone), which the data decide: either is allowed, and a set may
mix regions.

**The fit.** The slopes enter like extra coefficients, one per region and
covariate, penalized by their prior: for given Sigma, beta, b and the
absorbed constants are at the joint maximum of

    h = l(beta, b, alpha) - sum_g b_g' Sigma^-1 b_g / 2,

by the absorbed constants' Newton steps (`_AlternativeFE`, its blocks or
contraction), with b among the dense coefficients. Sigma then takes the
Laplace-EM update

    Sigma = mean_g [ b_g b_g' + C_g ],      C = (H_bb + I (x) Sigma^-1)^-1,

C_g the region's block (H_bb the slopes' information, given beta and the
constants), and the two alternate to convergence. The log-likelihood
reported is Laplace's approximation of the marginal likelihood, the
slopes integrated out (the constants and beta are parameters):

    L = h(theta-hat) - R log|Sigma| / 2 - log|H_bb + I (x) Sigma^-1| / 2.

With regions of thousands of choice sets, as regional slopes have, the
approximation is close to exact and the joint mode to the maximum of L;
the tests check the fit against brute-force integration.

**Inference.** beta's variance is the inverse of the penalized information
net of the constants and the slopes (given Sigma). Sigma's: from the
approximate marginal likelihood of the regions' slopes, each region's
unpenalized estimate b~_g = b_g + H_g^-1 Sigma^-1 b_g distributed as
N(0, Sigma + H_g^-1) (jointly, when regions share choice sets), by
differencing in Sigma's log-Cholesky parameters. A clustered variance
needs clusters that contain whole regions; the heteroskedasticity-robust
one is not available (a region's sets share its slopes).

**Downstream.** `ranef()` has each region's slopes (``re_<g>:<z>``) and
their conditional sds; `predict` adds z'b_g (0 for a region the fit did
not see), and so do ``calibrate=`` and `sample_assignments`;
`draw_parameters` draws beta, the slopes and the constants jointly, and
Sigma, with ``newdata=`` regions the fit did not see from N(0, Sigma).
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numba as nb
import numpy as np
import polars as pl
from hdfe_stream.core.report import warn

from .alt_fe import _alt_matvec, _AlternativeFE
from .newton import _Driver, _Fit
from .store import CODE

SLOPES_MAX = 2000           # regions x slopes, at most (the dense coefficients)
SIGMA_START = 0.01          # Sigma's starting diagonal
SIGMA_FLOOR = 1e-6          # a variance's floor (an sd of 0.001)
SIGMA_MAXITER = 500
LAPLACE_MAXITER = 200


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _slope_pass(X, cols, coef, alpha, lev, reg, zc, q, y, w, off, starts, row_alive, set_comp,
                loc, comp_pos, comp_size, mode, ll_t, g_t, H_t, ga, Hta, Haa, Hdiag):
    """`_alt_pass` with the regions' slopes among the dense coefficients:
    coef = (beta (P), b (R x q, region-major)); a row r's dense design is
    its covariates and, at its region's slots P + q reg[r] + j, its slope
    covariates X[r, zc[j]]. Per set, the dense part is handled on the
    covariates plus the slots of the set's regions, then scattered."""
    T = len(ll_t)
    n_sets = len(starts) - 1
    P = len(cols)
    D = lev.shape[1]
    has_w, has_off = len(w) > 0, len(off) > 0
    max_size = 1
    for c in range(n_sets):
        if starts[c + 1] - starts[c] > max_size:
            max_size = starts[c + 1] - starts[c]
    NZ = P + max_size * q
    for t in nb.prange(T):
        eta = np.empty(max_size)
        xl = np.zeros((max_size, NZ))
        regs = np.empty(max_size, np.int64)
        slot = np.empty(max_size, np.int64)
        idx = np.empty(NZ, np.int64)
        xbar = np.empty(NZ)
        dx = np.empty(NZ)
        g = g_t[t]
        H = H_t[t]
        llt = 0.0
        for c in range(n_sets):
            cc = set_comp[c]
            if cc < 0 or cc % T != t:
                continue
            a, b = starts[c], starts[c + 1]
            ws = w[a] if has_w else 1.0
            # the set's regions, and each row's slot among them
            nreg = 0
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                gr = reg[r]
                k = -1
                for u in range(nreg):
                    if regs[u] == gr:
                        k = u
                if k < 0:
                    regs[nreg] = gr
                    k = nreg
                    nreg += 1
                slot[r - a] = k
            nz = P + nreg * q
            for k in range(P):
                idx[k] = k
            for u in range(nreg):
                for j in range(q):
                    idx[P + u * q + j] = P + regs[u] * q + j
            m = -np.inf
            chosen = 0.0
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                e = off[r] if has_off else 0.0
                for k in range(nz):
                    xl[r - a, k] = 0.0
                for k in range(P):
                    x = X[r, cols[k]]
                    xl[r - a, k] = x
                    e += x * coef[k]
                base = P + slot[r - a] * q
                for j in range(q):
                    x = X[r, zc[j]]
                    xl[r - a, base + j] = x
                    e += x * coef[idx[base + j]]
                for d in range(D):
                    e += alpha[lev[r, d]]
                eta[r - a] = e
                if y[r] == 1:
                    chosen += e
                if e > m:
                    m = e
            S = 0.0
            for r in range(a, b):
                if row_alive[r]:
                    eta[r - a] = np.exp(eta[r - a] - m)
                    S += eta[r - a]
            llt += ws * (chosen - m - np.log(S))
            for k in range(nz):
                xbar[k] = 0.0
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                p = eta[r - a] / S
                eta[r - a] = p
                res = ws * ((1.0 if y[r] == 1 else 0.0) - p)
                for k in range(nz):
                    g[idx[k]] += res * xl[r - a, k]
                    xbar[k] += p * xl[r - a, k]
                for d in range(D):
                    ga[lev[r, d]] += res
            if mode == 0:
                continue
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                pw = ws * eta[r - a]
                for k in range(nz):
                    dx[k] = xl[r - a, k] - xbar[k]
                for k in range(nz):
                    v = pw * dx[k]
                    if v == 0.0:
                        continue
                    ik = idx[k]
                    for l2 in range(nz):
                        H[ik, idx[l2]] += v * dx[l2]
                for d in range(D):
                    lv = lev[r, d]
                    for k in range(nz):
                        Hta[lv, idx[k]] += pw * dx[k]
            if mode == 2:
                for r in range(a, b):
                    if not row_alive[r]:
                        continue
                    p = eta[r - a]
                    for d in range(D):
                        Hdiag[lev[r, d]] += ws * p * (1.0 - p)
                continue
            if D == 0:
                continue
            pos, s = comp_pos[cc], comp_size[cc]
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                pr = ws * eta[r - a]
                for d in range(D):
                    ia = loc[lev[r, d]]
                    for e2 in range(D):
                        Haa[pos + ia * s + loc[lev[r, e2]]] += pr
                    for q2 in range(a, b):
                        if not row_alive[q2]:
                            continue
                        v = pr * eta[q2 - a]
                        for e2 in range(D):
                            Haa[pos + ia * s + loc[lev[q2, e2]]] -= v
        ll_t[t] = llt


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _laplace_terms(X, cols, coef, alpha, lev, reg, zc, q, w, off, starts, row_alive, set_comp,
                   Minv, Cd_t, Ca):
    """c_s = d log|M| / d u_s for each row s (M = H_bb + I (x) Sigma^-1),
    with b held: w p_s (Q_s - sum_r p_r Q_r), Q_s = zt_s' M^-1 zt_s and zt
    the row's slope design less its set's mean; summed into the dense
    coefficients' (Cd: c_s times the row's dense design) and the
    constants' (Ca) terms."""
    T = Cd_t.shape[0]
    n_sets = len(starts) - 1
    P = len(cols)
    D = lev.shape[1]
    has_w, has_off = len(w) > 0, len(off) > 0
    max_size = 1
    for c in range(n_sets):
        if starts[c + 1] - starts[c] > max_size:
            max_size = starts[c + 1] - starts[c]
    for t in nb.prange(T):
        eta = np.empty(max_size)
        zz = np.zeros((max_size, max_size * q))
        regs = np.empty(max_size, np.int64)
        slot = np.empty(max_size, np.int64)
        Qr = np.empty(max_size)
        zbar = np.empty(max_size * q)
        Cd = Cd_t[t]
        for c in range(n_sets):
            cc = set_comp[c]
            if cc < 0 or cc % T != t:
                continue
            a, b = starts[c], starts[c + 1]
            ws = w[a] if has_w else 1.0
            nreg = 0
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                gr = reg[r]
                k = -1
                for u in range(nreg):
                    if regs[u] == gr:
                        k = u
                if k < 0:
                    regs[nreg] = gr
                    k = nreg
                    nreg += 1
                slot[r - a] = k
            nzb = nreg * q
            m = -np.inf
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                e = off[r] if has_off else 0.0
                for k in range(P):
                    e += X[r, cols[k]] * coef[k]
                for k in range(nzb):
                    zz[r - a, k] = 0.0
                for j in range(q):
                    x = X[r, zc[j]]
                    zz[r - a, slot[r - a] * q + j] = x
                    e += x * coef[P + reg[r] * q + j]
                for d in range(D):
                    e += alpha[lev[r, d]]
                eta[r - a] = e
                if e > m:
                    m = e
            S = 0.0
            for r in range(a, b):
                if row_alive[r]:
                    eta[r - a] = np.exp(eta[r - a] - m)
                    S += eta[r - a]
            for k in range(nzb):
                zbar[k] = 0.0
            for r in range(a, b):
                if row_alive[r]:
                    eta[r - a] /= S
                    for k in range(nzb):
                        zbar[k] += eta[r - a] * zz[r - a, k]
            Qbar = 0.0
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                v = 0.0
                for k in range(nzb):
                    gk = P + regs[k // q] * q + k % q
                    dk = zz[r - a, k] - zbar[k]
                    for l2 in range(nzb):
                        gl = P + regs[l2 // q] * q + l2 % q
                        v += dk * Minv[gk - P, gl - P] * (zz[r - a, l2] - zbar[l2])
                Qr[r - a] = v
                Qbar += eta[r - a] * v
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                cs = ws * eta[r - a] * (Qr[r - a] - Qbar)
                for k in range(P):
                    Cd[k] += cs * X[r, cols[k]]
                for j in range(q):
                    Cd[P + reg[r] * q + j] += cs * X[r, zc[j]]
                for d in range(D):
                    Ca[lev[r, d]] += cs


@nb.njit(cache=True)
def _region_scan(reg, starts, row_alive, case_alive, n_sets_g, n_rows_g, n_chosen_g, y):
    """Each region's sets, rows and choices among the alive sets; returns
    whether some set has rows of more than one region."""
    mixed = False
    for c in range(len(starts) - 1):
        if not case_alive[c]:
            continue
        first = -1
        for r in range(starts[c], starts[c + 1]):
            if not row_alive[r]:
                continue
            g = reg[r]
            n_rows_g[g] += 1
            if y[r] == 1:
                n_chosen_g[g] += 1
            if first < 0:
                first = g
                n_sets_g[g] += 1
            elif g != first:
                mixed = True
    return mixed


def _psi_of_sigma(S):
    """Sigma's log-Cholesky parameters: the factor's log diagonal, then its
    lower triangle by rows."""
    Lc = np.linalg.cholesky(S)
    q = S.shape[0]
    return np.r_[np.log(np.diag(Lc)), [Lc[i, j] for i in range(q) for j in range(i)]]


def _sigma_of_psi(psi, q):
    Lc = np.diag(np.exp(psi[:q]))
    k = q
    for i in range(q):
        for j in range(i):
            Lc[i, j] = psi[k]
            k += 1
    return Lc @ Lc.T


class _AltSlopes(_AlternativeFE):
    """The conditional logit with absorbed alternative constants (D >= 0
    dimensions) and random slopes by region; see the module docstring."""

    def __init__(self, model, p0, names, region, keys, region_key, slopes, log, logger=None,
                 budget=None, solver="auto"):
        super().__init__(model, p0, names, log, logger=logger, budget=budget, solver=solver,
                         keys=keys)
        self.region, self.rkey = region, region_key
        xn = list(model.xnames)
        missing = [z for z in slopes if z not in xn]
        if missing:
            raise ValueError(f"random slopes: {missing} are not covariates of the formula "
                             f"({xn})")
        self.slopes = list(slopes)
        self.zpos = np.array([xn.index(z) for z in slopes], np.int64)   # in beta
        self.zc = np.asarray(model.cols, np.int64)[self.zpos]           # columns of X
        self.q = q = len(slopes)
        self.R = R = int(p0.fe_tables[region].height)
        if R * q > SLOPES_MAX:
            raise ValueError(f"random slopes: {R:,} regions x {q} slopes is more than "
                             f"{SLOPES_MAX:,} (the slopes are dense coefficients here)")
        ns, nr, nc = np.zeros(R, np.int64), np.zeros(R, np.int64), np.zeros(R, np.int64)
        mixed = False
        for blk, c0 in self._blocks([self.rkey, "y"]):
            st = np.asarray(blk["starts"], np.int64)
            lev = self._lev(blk)
            mixed |= _region_scan(np.asarray(blk[self.rkey], np.int64), st,
                                  self._row_alive(lev), self.case_alive[c0:c0 + len(st) - 1],
                                  ns, nr, nc, blk["y"].astype(np.int64))
        self.mixed = bool(mixed)
        self.region_sets, self.region_rows, self.region_chosen = ns, nr, nc
        self.keep = nr > 0
        self.n_regions = int(self.keep.sum())
        if self.n_regions < 2:
            raise ValueError(f"random slopes by {region}: {self.n_regions} region(s) with "
                             "choice sets; a variance needs several")
        self.Sigma = np.eye(q) * SIGMA_START
        self.Lam = np.linalg.inv(self.Sigma)
        log(f"random slopes: {', '.join(slopes)} by {region} ({self.n_regions:,} regions; the "
            f"{'alternative' if mixed else 'choice set'}'s region)")

    # ------------------------------------------------------------ layout
    @property
    def n_params(self):
        """The estimated parameters beyond beta: the constants' (their rank)
        and Sigma's (not the slopes, which are integrated out)."""
        return self._n_alpha + self.q * (self.q + 1) // 2

    @n_params.setter
    def n_params(self, v):
        self._n_alpha = int(v)

    @property
    def Pd(self):
        return self.model.P + self.R * self.q

    def _bslice(self):
        P = self.model.P
        return slice(P, P + self.R * self.q)

    def start(self):
        base = super().start()
        P = self.model.P
        return np.r_[base[:P], np.zeros(self.R * self.q), base[P:]]

    def _penalty_blocks(self):
        """The slopes' prior precision, (R q) x (R q): Lam per kept region,
        the identity for a region without rows (its slopes stay 0)."""
        R, q = self.R, self.q
        out = np.zeros((R * q, R * q))
        for g in range(R):
            sl = slice(g * q, (g + 1) * q)
            out[sl, sl] = self.Lam if self.keep[g] else np.eye(q)
        return out

    # ------------------------------------------------------------ passes
    def pass_(self, driver, theta, hessian, mode=None):
        """`_AlternativeFE.pass_`, with the slopes and their prior: h, its
        gradient and the penalized information; H.bb the slopes'
        information without the prior."""
        driver.passes += 1
        m = self.model
        Pd, L = self.Pd, self.L
        mode = (1 if self.solver == "blocks" else 2) if hessian and mode is None else \
            (0 if mode is None else mode)
        coef = np.ascontiguousarray(theta[:Pd], np.float64)
        alpha = np.ascontiguousarray(theta[Pd:], np.float64)
        T = nb.get_num_threads()
        ll = 0.0
        g_t, H_t = np.zeros((T, Pd)), np.zeros((T, Pd, Pd) if mode else (T, 0, 0))
        ga, Hta = np.zeros(L), np.zeros((L, Pd) if mode else (0, Pd))
        Haa = np.zeros(int((self.comp_size.astype(np.int64) ** 2).sum()) if mode == 1 else 0)
        Hdiag = np.zeros(L if mode == 2 else 0)
        empty = np.empty(0)
        cols = np.asarray(m.cols, np.int64)
        for blk, c0 in self._blocks(["X", "y", "w", "off", self.rkey]):
            st = np.asarray(blk["starts"], np.int64)
            lev = self._lev(blk)
            ra = self._row_alive(lev)
            sc = self._set_comp(lev, st, ra, self.case_alive[c0:c0 + len(st) - 1])
            ll_t = np.zeros(T)
            w, off = blk.get("w"), blk.get("off")
            _slope_pass(np.ascontiguousarray(blk["X"], np.float64), cols, coef, alpha, lev,
                        np.asarray(blk[self.rkey], np.int64), self.zc, self.q,
                        blk["y"].astype(np.int64),
                        empty if w is None else np.ascontiguousarray(w, np.float64),
                        empty if off is None else np.ascontiguousarray(off, np.float64),
                        st, ra, sc, self.loc, self.comp_pos, self.comp_size, mode,
                        ll_t, g_t, H_t, ga, Hta, Haa, Hdiag)
            ll += ll_t.sum()
        if not np.isfinite(ll):
            return -np.inf, None, None
        b = coef[self._bslice()]
        Pen = self._penalty_blocks()
        ll -= 0.5 * float(b @ Pen @ b)
        gd = g_t.sum(axis=0)
        gd[self._bslice()] -= Pen @ b
        g = np.r_[gd, ga]
        if not hessian:
            return float(ll), g, None
        Hd = H_t.sum(axis=0)
        Hd = (Hd + Hd.T) / 2
        bs = self._bslice()
        bb = Hd[bs, bs].copy()
        Hd[bs, bs] += Pen
        H = SimpleNamespace(tt=Hd, ta=Hta, aa=Haa if mode == 1 else None,
                            diag=Hdiag if mode == 2 else None, red=None, bb=bb)
        return float(ll), g, H

    def _offsets(self, blk, coef):
        """Each row's offset plus its region's slopes' contribution z'b."""
        X = np.asarray(blk["X"], np.float64)
        reg = np.asarray(blk[self.rkey], np.int64)
        B = coef[self._bslice()].reshape(self.R, self.q)
        add = np.einsum("nj,nj->n", X[:, self.zc], B[reg])
        off = blk.get("off")
        return add if off is None else add + np.asarray(off, np.float64)

    def matvec(self, driver, theta, V):
        """H_aa V, by one read (the slopes enter as offsets)."""
        driver.passes += 1
        m = self.model
        P, Pd = m.P, self.Pd
        coef = np.ascontiguousarray(theta[:Pd], np.float64)
        beta = coef[:P].copy()
        alpha = np.ascontiguousarray(theta[Pd:], np.float64)
        out = np.zeros((self.L, V.shape[1]))
        V = np.ascontiguousarray(V, np.float64)
        empty = np.empty(0)
        cols = np.asarray(m.cols, np.int64)
        for blk, c0 in self._blocks(["X", "w", "off", self.rkey]):
            st = np.asarray(blk["starts"], np.int64)
            lev = self._lev(blk)
            ra = self._row_alive(lev)
            sc = self._set_comp(lev, st, ra, self.case_alive[c0:c0 + len(st) - 1])
            w = blk.get("w")
            _alt_matvec(np.ascontiguousarray(blk["X"], np.float64), cols, beta, alpha, lev,
                        empty if w is None else np.ascontiguousarray(w, np.float64),
                        np.ascontiguousarray(self._offsets(blk, coef)),
                        st, ra, sc, V, out, nb.get_num_threads())
        return out

    # ------------------------------------------------------------ variances
    def vcov_full(self, H):
        """The dense coefficients' (beta and the slopes) variance given
        Sigma: the inverse of their penalized information net of the
        constants."""
        return _Driver.inverse(self.reduced(H))

    def vcov_beta(self, H):
        return self.vcov_full(H)[:self.model.P, :self.model.P]

    def sandwich(self, H, M):
        V = self.vcov_full(H)
        out = V @ M @ V
        P = self.model.P
        out = out[:P, :P]
        return (out + out.T) / 2

    def meat(self, driver, theta, H, weights_type, clusters, hetero):
        """The clustered variances' middle matrices from each choice set's
        score for (beta, b) net of the constants: sum_r (y_r - p_r)
        (x~_r - sum_d B[lev_d(r)]), x~_r the row's dense design. Each
        cluster must contain whole regions; the prior's gradient is a
        region's and so a cluster's."""
        if hetero:
            raise ValueError("random slopes: the heteroskedasticity-robust variance is not "
                             "available (a region's choice sets share its slopes); cluster by "
                             "a variable that contains whole regions")
        if self.solver == "blocks":
            H = self._schur(H)
        BD = H.BD
        m = self.model
        P, Pd, q = m.P, self.Pd, self.q
        coef, alpha = theta[:Pd], theta[Pd:]
        U = {name: np.zeros((Gc, Pd)) for name, (_, Gc) in clusters.items()}
        region_cl = {name: np.full(self.R, -1, np.int64) for name in clusters}
        for blk, c0 in self._blocks(["X", "y", "w", "off", "cl", self.rkey]):
            st = np.asarray(blk["starts"], np.int64)
            n_sets = len(st) - 1
            if not n_sets:
                continue
            lev = self._lev(blk)
            ca = self.case_alive[c0:c0 + n_sets]
            ra = self._row_alive(lev) & np.repeat(ca, np.diff(st))
            X = np.asarray(blk["X"], np.float64)
            reg = np.asarray(blk[self.rkey], np.int64)
            eta = X[:, m.cols] @ coef[:P] + self._offsets(blk, coef)
            if self.D:
                eta = eta + alpha[lev].sum(axis=1)
            eta = np.where(ra, eta, -np.inf)
            mx = np.maximum.reduceat(eta, st[:-1])
            mx = np.where(np.isfinite(mx), mx, 0.0)
            e = np.where(ra, np.exp(eta - np.repeat(mx, np.diff(st))), 0.0)
            S = np.add.reduceat(e, st[:-1])
            p = e / np.repeat(np.where(S > 0, S, 1.0), np.diff(st))
            r = np.where(ra, blk["y"] - p, 0.0)
            w = blk.get("w")
            wr = np.ones(len(r)) if w is None else np.asarray(w, np.float64)
            cl = np.asarray(blk["cl"])
            set_of_row = np.repeat(np.arange(n_sets), np.diff(st))
            rr = r * wr
            Xt = X[:, m.cols]
            if self.D:
                Xt = Xt - BD[lev, :P].sum(axis=1)                # beta's part, net of alpha
                Bs = BD[lev, P:].sum(axis=1)                     # rows x (R q)
            for name, (col, _) in clusters.items():
                codes = np.asarray(cl[st[:-1], col], np.int64)[set_of_row]   # row's cluster
                G = U[name].shape[0]
                for k in range(P):
                    U[name][:, k] += np.bincount(codes, weights=rr * Xt[:, k], minlength=G)
                # the slopes' part: at the row's region's slots, net of the constants
                for j in range(q):
                    np.add.at(U[name], (codes, P + reg * q + j), rr * X[:, self.zc[j]])
                if self.D:
                    for k in range(self.R * q):
                        U[name][:, P + k] -= np.bincount(codes, weights=rr * Bs[:, k],
                                                         minlength=G)
                # whole regions within a cluster
                rc = region_cl[name]
                pairs = np.unique(np.column_stack([reg[ra], codes[ra]]), axis=0)
                for g_, c_ in pairs:
                    if rc[g_] >= 0 and rc[g_] != c_:
                        raise ValueError(f"random slopes: the clustering {name!r} splits a "
                                         f"region's ({self.region}) choice sets; clusters must "
                                         "contain whole regions")
                    rc[g_] = c_
                dup = np.unique(pairs[:, 0], return_counts=True)[1]
                if np.any(dup > 1):
                    raise ValueError(f"random slopes: the clustering {name!r} splits a "
                                     f"region's ({self.region}) choice sets; clusters must "
                                     "contain whole regions")
        # the prior's gradient, -Lam b_g, belongs to the region's cluster
        b = coef[self._bslice()].reshape(self.R, q)
        for name in clusters:
            rc = region_cl[name]
            for g_ in np.flatnonzero(rc >= 0):
                U[name][rc[g_], P + g_ * q:P + (g_ + 1) * q] -= self.Lam @ b[g_]
        return None, U

    # ------------------------------------------------------------ the fit
    def _newton(self, driver, theta, fml, halvings=30):
        """Newton's method on h for the current Sigma (the blocks)."""
        ll, g, H = self.pass_(driver, theta, True)
        it, dec, converged = 0, np.inf, False
        while it < driver.maxiter:
            it += 1
            step, pd = self.solve(H, g)
            dec = float(g @ step)
            if pd and 0 <= dec < driver.tol:
                converged = True
                break
            t = 1.0
            for _ in range(halvings):
                th = theta + t * step
                ll_n, _, _ = self.pass_(driver, th, False)
                if np.isfinite(ll_n) and ll_n >= ll - 1e-10 * abs(ll):
                    break
                t /= 2
            else:
                warn(f"{fml}: random slopes: the line search failed", self.logger)
                break
            theta = th
            ll, g, H = self.pass_(driver, theta, True)
        return SimpleNamespace(theta=theta, ll=ll, g=g, H=H, iterations=it, converged=converged,
                               decrement=dec)

    def _at(self, driver, theta):
        """h, its gradient and information at theta (no steps)."""
        ll, g, H = self.pass_(driver, theta, True)
        return SimpleNamespace(theta=theta, ll=ll, g=g, H=H, iterations=0, converged=True,
                               decrement=0.0)

    def _mode(self, driver, theta, fml):
        if self.solver == "contraction":
            f = self.run_contraction(driver, theta, fml)
            _, _, H = self.pass_(driver, f.theta, True, mode=2)
            # (run_contraction's H carries red and BD; keep its bb from this read)
            f.H.bb = H.bb
            return SimpleNamespace(theta=f.theta, ll=f.ll, g=f.g, H=f.H, iterations=f.iterations,
                                   converged=f.converged, decrement=f.decrement)
        return self._newton(driver, theta, fml)

    def _cond_cov(self, H):
        """Each region's slopes' conditional covariance given beta and the
        constants, (R, q, q): blocks of (H_bb + I (x) Lam)^-1; and the
        log-determinant of H_bb + I (x) Lam over the kept regions."""
        R, q = self.R, self.q
        A = H.bb + self._penalty_blocks()
        keep = np.repeat(self.keep, q)
        if not self.mixed:
            C = np.zeros((R, q, q))
            logdet = 0.0
            for g in range(R):
                if not self.keep[g]:
                    continue
                Ag = A[g * q:(g + 1) * q, g * q:(g + 1) * q]
                C[g] = np.linalg.inv(Ag)
                logdet += float(np.linalg.slogdet(Ag)[1])
            return C, logdet
        Ak = A[np.ix_(keep, keep)]
        Ci = np.linalg.inv(Ak)
        full = np.zeros((R * q, R * q))
        full[np.ix_(keep, keep)] = Ci
        C = np.array([full[g * q:(g + 1) * q, g * q:(g + 1) * q] for g in range(R)])
        return C, float(np.linalg.slogdet(Ak)[1])

    def _Minv(self, H):
        """(H_bb + I (x) Sigma^-1)^-1, (R q) x (R q) (the identity for a region
        without rows)."""
        R, q = self.R, self.q
        A = H.bb + self._penalty_blocks()
        if not self.mixed:
            out = np.zeros((R * q, R * q))
            for g in range(R):
                sl = slice(g * q, (g + 1) * q)
                out[sl, sl] = np.linalg.inv(A[sl, sl])
            return out
        return np.linalg.inv(A)

    def _laplace_C(self, driver, theta, Minv):
        """d log|M| / d theta with b held: the dense coefficients' part and
        the constants', by one read."""
        driver.passes += 1
        m = self.model
        Pd = self.Pd
        coef = np.ascontiguousarray(theta[:Pd], np.float64)
        alpha = np.ascontiguousarray(theta[Pd:], np.float64)
        T = nb.get_num_threads()
        Cd_t, Ca = np.zeros((T, Pd)), np.zeros(self.L)
        empty = np.empty(0)
        cols = np.asarray(m.cols, np.int64)
        Minv = np.ascontiguousarray(Minv, np.float64)
        for blk, c0 in self._blocks(["X", "w", "off", self.rkey]):
            st = np.asarray(blk["starts"], np.int64)
            lev = self._lev(blk)
            ra = self._row_alive(lev)
            sc = self._set_comp(lev, st, ra, self.case_alive[c0:c0 + len(st) - 1])
            w, off = blk.get("w"), blk.get("off")
            _laplace_terms(np.ascontiguousarray(blk["X"], np.float64), cols, coef, alpha, lev,
                           np.asarray(blk[self.rkey], np.int64), self.zc, self.q,
                           empty if w is None else np.ascontiguousarray(w, np.float64),
                           empty if off is None else np.ascontiguousarray(off, np.float64),
                           st, ra, sc, Minv, Cd_t, Ca)
        return Cd_t.sum(axis=0), Ca

    def _sigma_step(self, b, C, u):
        """Sigma's fixed-point update, mean over the kept regions of
        b b' + C - sym(b u') (u = M^-1 d log|M| / d b; 0 for Laplace-EM),
        floored."""
        k = self.keep
        S = (np.einsum("gi,gj->ij", b[k], b[k]) + C[k].sum(axis=0)
             - 0.5 * (np.einsum("gi,gj->ij", b[k], u[k]) + np.einsum("gi,gj->ij", u[k], b[k])))
        S = S / self.n_regions
        S = (S + S.T) / 2
        ev, Q = np.linalg.eigh(S)
        return (Q * np.maximum(ev, SIGMA_FLOOR)) @ Q.T

    def _refine(self, driver, theta, fml, tol):
        """The exact Laplace approximation: Newton steps in (beta, b, the
        constants) whose gradient for beta and the constants adds the
        log-determinant's, -(1/2) [C_theta - H_theta,b M^-1 C_b], with b
        following to first order, and Sigma's exact fixed-point update; to
        convergence of both. Needs the blocks."""
        P, Pd, R, q = self.model.P, self.Pd, self.R, self.q
        bs = self._bslice()
        converged, it, dec = False, 0, np.inf
        for it in range(1, LAPLACE_MAXITER + 1):
            ll, g, H = self.pass_(driver, theta, True)
            Minv = self._Minv(H)
            Cd, Ca = self._laplace_C(driver, theta, Minv)
            u = Minv @ Cd[bs]
            gm = g.copy()
            gm[:P] += -0.5 * (Cd[:P] - H.tt[:P, bs] @ u)
            if self.D:
                gm[Pd:] += -0.5 * (Ca - H.ta[:, bs] @ u)
            step, pd = self.solve(H, gm)
            dec = float(gm @ step)
            C = np.array([Minv[gq * q:(gq + 1) * q, gq * q:(gq + 1) * q] for gq in range(R)])
            S_new = self._sigma_step(theta[bs].reshape(R, q), C, u.reshape(R, q))
            change = float(np.max(np.abs(S_new - self.Sigma)) / max(np.max(np.abs(S_new)),
                                                                   SIGMA_FLOOR))
            driver.log(f"  Laplace {it}: decrement {dec:.3e}, Sigma "
                       f"{np.round(np.diag(S_new), 6)}, change {change:.2e}")
            theta = theta + step
            self.Sigma, self.Lam = S_new, np.linalg.inv(S_new)
            if pd and 0 <= dec < driver.tol and change < tol:
                converged = True
                break
        return theta, converged, it

    def fit(self, driver, theta, fml, tol=1e-6):
        """The joint mode for each Sigma with Sigma's Laplace-EM updates (a
        start), then the exact Laplace approximation's maximum (with the
        blocks); returns a `_Fit` (its ll Laplace's) and a summary."""
        t0 = time.time()
        p_start = driver.passes
        q = self.q
        converged, rounds = False, 0
        f = None
        for rounds in range(1, SIGMA_MAXITER + 1):
            f = self._mode(driver, theta, fml)
            theta = f.theta
            b = theta[self._bslice()].reshape(self.R, q)[self.keep]
            C, _ = self._cond_cov(f.H)
            S_new = (np.einsum("gi,gj->ij", b, b) + C[self.keep].sum(axis=0)) / self.n_regions
            S_new = (S_new + S_new.T) / 2
            ev, Q = np.linalg.eigh(S_new)
            S_new = (Q * np.maximum(ev, SIGMA_FLOOR)) @ Q.T
            change = float(np.max(np.abs(S_new - self.Sigma)) / max(np.max(np.abs(S_new)),
                                                                   SIGMA_FLOOR))
            driver.log(f"  random slopes round {rounds}: h {f.ll:.10g}, Sigma "
                       f"{np.round(np.diag(S_new), 6)}, change {change:.2e}")
            self.Sigma, self.Lam = S_new, np.linalg.inv(S_new)
            if change < max(tol, 1e-4):
                converged = True
                break
        method = "laplace-em"
        if self.solver == "blocks":
            theta, conv_l, steps = self._refine(driver, theta, fml, tol)
            converged = conv_l
            method = "laplace"
        else:
            warn(f"{fml}: with fe_solver='contraction' the random slopes are fitted at the joint "
                 "mode with Laplace-EM updates of Sigma (the exact Laplace refinement needs the "
                 "blocks)", self.logger)
        self.method = method
        # the final mode at the final Sigma
        f = self._mode(driver, theta, fml) if method == "laplace-em" else \
            self._at(driver, theta)
        self.theta = f.theta
        C, logdet = self._cond_cov(f.H)
        self.cond_cov = C
        ll_lap = (f.ll - 0.5 * self.n_regions * float(np.linalg.slogdet(self.Sigma)[1])
                  - 0.5 * logdet)
        self.singular = bool(np.min(np.linalg.eigvalsh(self.Sigma)) <= SIGMA_FLOOR * 1.0001)
        self.sigma_info = self.sigma_inference(f.H)
        info = {"loglik": ll_lap, "h": f.ll, "rounds": rounds, "method": method,
                "converged": converged and f.converged, "passes": driver.passes - p_start,
                "seconds": round(time.time() - t0, 3)}
        fit = _Fit(theta=f.theta, ll=ll_lap, g=f.g, H=f.H, ll_start=np.nan,
                   iterations=f.iterations, converged=info["converged"], decrement=f.decrement,
                   halvings=0, path=[], seconds=time.time() - t0)
        return fit, info

    # ------------------------------------------------------------ Sigma's inference
    def _marginal(self, psi, bt, V):
        """The approximate marginal log-likelihood of Sigma (log-Cholesky
        psi) from the regions' unpenalized slopes bt with covariance V."""
        q = self.q
        S = _sigma_of_psi(psi, q)
        if not self.mixed:
            out = 0.0
            for g in range(len(bt)):
                M = S + V[g]
                sign, ld = np.linalg.slogdet(M)
                if sign <= 0:
                    return -np.inf
                out -= 0.5 * (ld + bt[g] @ np.linalg.solve(M, bt[g]))
            return out
        k = len(bt)
        M = np.kron(np.eye(k), S) + V
        sign, ld = np.linalg.slogdet(M)
        if sign <= 0:
            return -np.inf
        x = bt.ravel()
        return -0.5 * (ld + x @ np.linalg.solve(M, x))

    def sigma_inference(self, H):
        """Sigma's elements' variance (the pairs own, diagonal first), by
        differencing the approximate marginal likelihood in psi."""
        q, R = self.q, self.R
        keep = self.keep
        b = self.theta[self._bslice()].reshape(R, q)
        Hb = H.bb
        if not self.mixed:
            bt, V = [], []
            for g in np.flatnonzero(keep):
                Hg = Hb[g * q:(g + 1) * q, g * q:(g + 1) * q]
                Vg = np.linalg.pinv(Hg)
                bt.append(b[g] + Vg @ self.Lam @ b[g])
                V.append(Vg)
            bt, V = np.array(bt), np.array(V)
        else:
            kk = np.repeat(keep, q)
            Hk = Hb[np.ix_(kk, kk)]
            Vk = np.linalg.pinv(Hk)
            bk = b[keep].ravel()
            Lk = np.kron(np.eye(self.n_regions), self.Lam)
            bt = (bk + Vk @ Lk @ bk).reshape(-1, q)
            V = Vk
        psi = _psi_of_sigma(self.Sigma)
        k = len(psi)
        h = 1e-4
        Hp = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                ei, ej = np.eye(k)[i] * h, np.eye(k)[j] * h
                v = (self._marginal(psi + ei + ej, bt, V) - self._marginal(psi + ei - ej, bt, V)
                     - self._marginal(psi - ei + ej, bt, V)
                     + self._marginal(psi - ei - ej, bt, V)) / (4 * h * h)
                Hp[i, j] = Hp[j, i] = v
        try:
            Vpsi = np.linalg.inv(-Hp)
        except np.linalg.LinAlgError:
            Vpsi = np.full((k, k), np.nan)
        own = [(j, j) for j in range(q)] + [(i, j) for i in range(q) for j in range(i)]
        # d Sigma_own / d psi, by differences
        J = np.zeros((len(own), k))
        for i in range(k):
            e = np.eye(k)[i] * 1e-6
            Sp, Sm = _sigma_of_psi(psi + e, q), _sigma_of_psi(psi - e, q)
            J[:, i] = [(Sp[a, c] - Sm[a, c]) / 2e-6 for a, c in own]
        Vf = J @ Vpsi @ J.T
        return own, Vf, psi, Vpsi

    # ------------------------------------------------------------ results
    def result_spec(self, theta):
        out = super().result_spec(theta)
        out.pop("dropped", None)
        q, R = self.q, self.R
        b = theta[self._bslice()].reshape(R, q)
        sd = np.sqrt(np.clip(np.diagonal(self.cond_cov, axis1=1, axis2=2), 0, None))
        g = self.region
        vals = {f"re_{g}:{z}": pl.Series(b[:, j]) for j, z in enumerate(self.slopes)}
        sds = {f"sd_{g}:{z}": pl.Series(sd[:, j]) for j, z in enumerate(self.slopes)}
        n = self.region_rows if self.mixed else self.region_sets
        table = self.p0.fe_tables[g].with_columns(**vals, **sds, n_obs=pl.Series(n),
                                                  n_chosen=pl.Series(self.region_chosen))
        keep = pl.Series(self.keep)
        out["ranef"] = {g: table.filter(keep).drop(CODE)}
        out["ranef_cov"] = {g: np.ascontiguousarray(self.cond_cov[self.keep])}
        out["slopes"] = {g: list(self.slopes)}
        out["effects"].append((g, list(vals), "slope"))
        return out

    def draw_info(self, theta, H):
        """For `draw_parameters`: the constants' pieces (as
        `_AlternativeFE.draw_info`, over the dense coefficients), the dense
        coefficients' joint variance, and Sigma's."""
        if self.D > 1 or (self.D and self.solver != "blocks"):
            return {"kind": None, "why": "with random slopes and absorbed alternative "
                    "constants it needs one dimension of constants and fe_solver='blocks'"}
        if self.D:
            base = super().draw_info(theta, H)
            if base["kind"] is None:
                return base
            base["alpha"] = np.asarray(theta[self.Pd:], float).copy()
        else:
            base = {"alpha": np.zeros(0)}
        own, Vf, psi, Vpsi = self.sigma_info
        base.update(kind="alt_slopes", P=self.model.P, Pd=self.Pd, V=self.vcov_full(H),
                    coef=np.asarray(theta[:self.Pd], float).copy(), R=self.R, q=self.q,
                    keep=self.keep.copy(), psi=psi, Vpsi=Vpsi, region=self.region,
                    D=self.D)
        return base
