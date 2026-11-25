"""Random slopes for the binary, ordered and multinomial logit
(experimental): ``random={"school": ["1", "prior"]}`` -- one random grouping
whose groups each have an intercept and/or slopes in some covariates, drawn
from N(0, Sigma) -- optionally within an absorbed fixed effect the grouping
nests in.

**The model.** Group g's effect vector a_g enters each row's linear
predictor through the row's random design z_i (a 1 for the intercept, then
the slope covariates): the ordered logit's eta_i + z_i'a_g (binary: the
two-level ordered logit), the multinomial's eta_ij + z_i'a_gj for each
non-base level j, a_g then the levels' vectors stacked (level-major), and
Sigma of their joint dimension q = (J - 1) x terms.

**The fit** is the multinomial adaptive quadrature's (`adaptive_mnl`),
generalized to the row design: Newton's method in the coefficients b, the
fixed effect's free levels and Sigma^-1's elements, each group's integral by
the product of K-point Gauss-Hermite rules (K^q points) centred at its mode
and scaled by its conditional covariance, the points held during a step and
moved between steps. Laplace's approximation is not offered: random slopes
are wanted most where groups are small, where it is biased. The start is
the random intercepts' joint-mode fit, with the slopes' variances small.

**Downstream.** `ranef()` has each group's modes (``re_<g>`` for the
intercept, ``re_<g>:<x>`` for a slope; with the multinomial logit
``re_<g>:<level>`` and ``re_<g>:<level>:<x>``) and their conditional sds;
`predict` adds z_i'a_g (0 for a group the fit did not see), and
``effects="average"`` averages over the effects' distribution;
`draw_parameters` draws b, the fixed effect and Sigma, with
``ranef=True`` each group's effects given its rows (importance resampling,
as for the multinomial intercepts).
"""

from __future__ import annotations

import numba as nb
import numpy as np
import polars as pl

from ..core.models import _ord_terms
from .adaptive_mnl import (
    _E,
    HALF_LOG_2PI,
    LARGE_ROWS,
    MAX_POINTS,
    SMALL_NODES,
    _AdaptiveMNL,
    _bysum,
    _rule,
    _sigma_params,
)

VECTOR_NODES = {1: 9, 2: 7}         # the default points per dimension (else 5)
MIN_SLOPE_NODES = 5


def default_nodes(q):
    return VECTOR_NODES.get(q, 5)


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _mnl_kernel_z(E0, A, Z, r, vx, y, w, X, st, kc, logL, pi, Pbar, PP, Ds, Db, score0,
                  curv0):
    """`adaptive_mnl._group_kernel` with a row design: the effect on level
    j's predictor at point k is sum_t Z[i, t] A[g, k, (j - 1) r + t]; the
    centre's score and curvature are in the q = (J - 1) r effects."""
    G, Kt, q = A.shape
    J = E0.shape[1]
    Kx = X.shape[1]
    for g in nb.prange(G):
        r0, r1 = st[g], st[g + 1]
        ng = r1 - r0
        p = np.empty((ng, Kt, J))
        F = np.zeros(Kt)
        eta = np.empty(J)
        for k in range(Kt):
            for ii in range(ng):
                i = r0 + ii
                mx = -np.inf
                for j in range(J):
                    e = E0[i, j]
                    if j > 0:
                        for t in range(r):
                            e += Z[i, t] * A[g, k, (j - 1) * r + t]
                    eta[j] = e
                    if e > mx:
                        mx = e
                den = 0.0
                for j in range(J):
                    x = np.exp(eta[j] - mx)
                    p[ii, k, j] = x
                    den += x
                for j in range(J):
                    p[ii, k, j] /= den
                F[k] += w[i] * (eta[y[i]] - mx - np.log(den))
        vmax = -np.inf
        for k in range(Kt):
            F[k] += vx[g, k]
            if F[k] > vmax:
                vmax = F[k]
        tot = 0.0
        for k in range(Kt):
            tot += np.exp(F[k] - vmax)
        lg = vmax + np.log(tot)
        logL[g] = lg
        for k in range(Kt):
            pi[g, k] = np.exp(F[k] - lg)
        Sk = np.zeros((Kt, J))
        Tk = np.zeros((Kt, J - 1, Kx))
        for ii in range(ng):
            i = r0 + ii
            for a in range(J):
                Pbar[i, a] = 0.0
                for c in range(J):
                    PP[i, a, c] = 0.0
            for k in range(Kt):
                pk = pi[g, k]
                for a in range(J):
                    pa = p[ii, k, a]
                    Pbar[i, a] += pk * pa
                    Sk[k, a] += w[i] * pa
                    qq = pk * pa
                    for c in range(a, J):
                        PP[i, a, c] += qq * p[ii, k, c]
                    if a > 0:
                        wpa = w[i] * pa
                        for c in range(Kx):
                            Tk[k, a - 1, c] += wpa * X[i, c]
            for a in range(J):
                for c in range(a + 1, J):
                    PP[i, c, a] = PP[i, a, c]
        mS = np.zeros(J)
        mT = np.zeros((J - 1, Kx))
        for k in range(Kt):
            pk = pi[g, k]
            for a in range(J):
                mS[a] += pk * Sk[k, a]
            for a in range(J - 1):
                for c in range(Kx):
                    mT[a, c] += pk * Tk[k, a, c]
        for k in range(Kt):
            for a in range(J):
                Ds[g, k, a] = mS[a] - Sk[k, a]
            for a in range(J - 1):
                for c in range(Kx):
                    Db[g, k, a, c] = mT[a, c] - Tk[k, a, c]
        for a in range(q):
            score0[g, a] = 0.0
            for c in range(q):
                curv0[g, a, c] = 0.0
        for ii in range(ng):
            i = r0 + ii
            wi = w[i]
            for la in range(J - 1):
                pa = p[ii, kc, la + 1]
                ra = (1.0 if y[i] == la + 1 else 0.0) - pa
                for ta in range(r):
                    ia = la * r + ta
                    score0[g, ia] += wi * ra * Z[i, ta]
                    for lc in range(J - 1):
                        wac = (pa if la == lc else 0.0) - pa * p[ii, kc, lc + 1]
                        for tc in range(r):
                            curv0[g, ia, lc * r + tc] += wi * wac * Z[i, ta] * Z[i, tc]


def _points_py(mode, Rg, logdetR, gu, z, lw, Lam, c0, A, vx):
    """Each group's quadrature points, a = mode + R z_k (A), and the log of
    their weight times the N(0, Sigma) density over the standard normal's
    (vx: lw_k + log det R - a' Lam a / 2 + c0)."""
    G, Kt, q = A.shape
    for gi in nb.prange(G):         # (nb.prange is range in the serial compilation)
        g = gu[gi]
        for k in range(Kt):
            for a in range(q):
                v = mode[g, a]
                for c in range(q):
                    v += Rg[g, a, c] * z[k, c]
                A[gi, k, a] = v
            quad = 0.0
            for a in range(q):
                t = 0.0
                for c in range(q):
                    t += A[gi, k, c] * Lam[c, a]
                quad += t * A[gi, k, a]
            vx[gi, k] = lw[k] + logdetR[g] + c0 - 0.5 * quad


_points_par = nb.njit(parallel=True, cache=True, error_model="numpy")(_points_py)
_points_ser = nb.njit(cache=True, error_model="numpy")(_points_py)
POINTS_PARALLEL_MIN = 2_000_000     # multiply-adds: below, a parallel launch costs more


def _points(mode, Rg, logdetR, gu, z, lw, Lam, c0, A, vx):
    """`_points_py`, in parallel over the groups when there is enough work
    (the multinomial's chunks hold few groups, each with many points)."""
    G, Kt, q = A.shape
    fn = _points_par if G * Kt * q * q >= POINTS_PARALLEL_MIN else _points_ser
    fn(mode, Rg, logdetR, gu, z, lw, Lam, c0, A, vx)


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _ord_kernel_z(E0, A, Z, vx, cuts, y, w, X, st, kc, probit, hmode, fix_first, coef, Sv, jj,
                  ll_, logL, pi, Wd, slbar, Sbar, vdS, fev, R, Qb, score0, curv0, pmean, pcov):
    """The ordered (and binary) logit's chunk of `_AdaptiveSlopes._accumulate`
    in one pass over its groups: at each group's point k the rows' linear
    predictors are E0_i + sum_t Z[i, t] A[g, k, t], and their pieces those
    of `_Ordered.rows` (`_ord_terms`).

    Per group: the log-integral (logL, NaN where a row's probability is 0)
    and the points' posterior weights pi; at each point the deviation from
    its posterior mean of the score in E = (b, the cutpoints less the first
    with `fix_first`, Sigma^-1's elements), weighted by sqrt(pi) (Wd: its
    sum of outer products is the information's between-point part); the
    posterior mean of the score in Sigma^-1's elements, coef (Sv - a_j a_l)
    (slbar); for the fixed effect, the posterior mean of the score in eta
    (Sbar), its posterior variance (vdS) and its posterior covariance with
    the score in E (fev); the centre point kc's score and curvature in the
    effects; the effects' posterior mean and covariance. Per row: the
    pi-weighted means of `rows`' s, ga, gb, h, haa, hbb and hab (R, in that
    order) and of Q (Qb)."""
    G, Kt, q = A.shape
    K = X.shape[1]
    C = len(cuts)
    nb_ = K + C
    P = nb_ - (1 if fix_first else 0)
    ns = len(coef)
    nE = P + ns
    for g in nb.prange(G):
        r0, r1 = st[g], st[g + 1]
        ng = r1 - r0
        keep = np.empty((ng, Kt, 5))
        F = np.zeros(Kt)
        Gk = np.zeros((Kt, nb_))
        Sk = np.zeros(Kt)
        bad = False
        for k in range(Kt):
            for ii in range(ng):
                i = r0 + ii
                e = E0[i]
                for t in range(q):
                    e += Z[i, t] * A[g, k, t]
                yi = y[i]
                a = cuts[yi] - e if yi < C else np.inf
                b = cuts[yi - 1] - e if yi > 0 else -np.inf
                Pr, ga, gb, haa, hbb, hab = _ord_terms(a, b, probit, hmode)
                lp = np.log(Pr)
                if not np.isfinite(lp):
                    bad = True
                wi = w[i]
                F[k] += wi * lp
                s = -(ga + gb) * wi
                Sk[k] += s
                for c in range(K):
                    Gk[k, c] += s * X[i, c]
                if yi < C:
                    Gk[k, K + yi] += ga * wi
                if yi > 0:
                    Gk[k, K + yi - 1] += gb * wi
                keep[ii, k, 0] = ga
                keep[ii, k, 1] = gb
                keep[ii, k, 2] = haa
                keep[ii, k, 3] = hbb
                keep[ii, k, 4] = hab
        if bad:
            logL[g] = np.nan
            continue
        vmax = -np.inf
        for k in range(Kt):
            F[k] += vx[g, k]
            if F[k] > vmax:
                vmax = F[k]
        tot = 0.0
        for k in range(Kt):
            tot += np.exp(F[k] - vmax)
        lg = vmax + np.log(tot)
        logL[g] = lg
        for k in range(Kt):
            pi[g, k] = np.exp(F[k] - lg)
        # the score in E at each point (the first cutpoint's column dropped
        # with fix_first), its posterior mean, and the deviations
        Ek = np.empty((Kt, nE))
        for k in range(Kt):
            c2 = 0
            for c in range(nb_):
                if fix_first and c == K:
                    continue
                Ek[k, c2] = Gk[k, c]
                c2 += 1
            for p in range(ns):
                Ek[k, P + p] = coef[p] * (Sv[p] - A[g, k, jj[p]] * A[g, k, ll_[p]])
        Eb = np.zeros(nE)
        sbar = 0.0
        for k in range(Kt):
            pk = pi[g, k]
            sbar += pk * Sk[k]
            for c in range(nE):
                Eb[c] += pk * Ek[k, c]
        for p in range(ns):
            slbar[g, p] = Eb[P + p]
        Sbar[g] = sbar
        v = 0.0
        for c in range(nE):
            fev[g, c] = 0.0
        for k in range(Kt):
            pk = pi[g, k]
            rk = np.sqrt(pk)
            ds = Sk[k] - sbar
            v += pk * ds * ds
            for c in range(nE):
                dev = Ek[k, c] - Eb[c]
                Wd[g, k, c] = rk * dev
                fev[g, c] += pk * dev * ds
        vdS[g] = v
        for a in range(q):
            score0[g, a] = 0.0
            for c in range(q):
                curv0[g, a, c] = 0.0
        for ii in range(ng):
            i = r0 + ii
            wi = w[i]
            yi = y[i]
            sb = 0.0
            gab = 0.0
            gbb = 0.0
            hb = 0.0
            haab = 0.0
            hbbb = 0.0
            habb = 0.0
            for k in range(Kt):
                pk = pi[g, k]
                ga = keep[ii, k, 0]
                gb = keep[ii, k, 1]
                haa = keep[ii, k, 2]
                hbb = keep[ii, k, 3]
                hab = keep[ii, k, 4]
                sb += pk * (ga + gb)
                gab += pk * ga
                gbb += pk * gb
                hb += pk * max(-(haa + 2.0 * hab + hbb), 0.0)
                haab += pk * haa
                hbbb += pk * hbb
                habb += pk * hab
            R[i, 0] = -sb * wi
            R[i, 1] = gab * wi
            R[i, 2] = gbb * wi
            R[i, 3] = hb * wi
            R[i, 4] = -haab * wi
            R[i, 5] = -hbbb * wi
            R[i, 6] = -habb * wi
            for c in range(C):
                Qb[i, c] = 0.0
            if yi < C:
                Qb[i, yi] += wi * (haab + habb)
            if yi > 0:
                Qb[i, yi - 1] += wi * (habb + hbbb)
            ga = keep[ii, kc, 0]
            gb = keep[ii, kc, 1]
            s = -(ga + gb) * wi
            h = wi * max(-(keep[ii, kc, 2] + 2.0 * keep[ii, kc, 4] + keep[ii, kc, 3]), 0.0)
            for a in range(q):
                score0[g, a] += s * Z[i, a]
                for c in range(q):
                    curv0[g, a, c] += h * Z[i, a] * Z[i, c]
        for a in range(q):
            mm = 0.0
            for k in range(Kt):
                mm += pi[g, k] * A[g, k, a]
            pmean[g, a] = mm
        for a in range(q):
            for c in range(q):
                vv = 0.0
                for k in range(Kt):
                    vv += pi[g, k] * A[g, k, a] * A[g, k, c]
                pcov[g, a, c] = vv - pmean[g, a] * pmean[g, c]


class _AdaptiveSlopes(_AdaptiveMNL):
    """Adaptive quadrature for one random grouping with intercepts and/or
    slopes, for the multinomial and the ordered (and binary) logit (see the
    module docstring). `re` is the random intercepts' `_RandomEffects` (its
    store, groups and fixed effect); `terms` the random design, "1" for the
    intercept and covariate names."""

    def __init__(self, re, driver, n_nodes, log, logger=None, terms=("1",), large_rows=None):
        large_rows = LARGE_ROWS if large_rows is None else large_rows
        if len(re.random) != 1 or re.kind != "tree" or re.factors[0] is not re.random[0]:
            raise ValueError("random slopes take one random grouping")
        if getattr(re.model, "own", None):
            raise ValueError("random slopes do not take own_covariates= yet")
        if not getattr(re.p0, "fe_sorted", False):
            raise ValueError(
                "random slopes need each group's rows in one block of the store, and they could "
                "not be grouped (a group larger than a block, or too little memory to sort "
                "them; see the log): raise block_mb or max_memory_gb")
        self.re, self.driver, self.log, self.logger = re, driver, log, logger
        m = self.m = re.model
        self.kind = m.kind
        self.f, self.fe = re.random[0], re.fixed_factor
        self.J = re.J
        self.dl = re.d                       # levels with effects (J - 1, or 1)
        self.Jf = re.J if self.kind == "mlogit" else 1
        xn = list(m.xnames)
        self.terms = list(terms)
        bad = [t for t in self.terms if t != "1" and t not in xn]
        if bad:
            raise ValueError(f"random slopes: {bad} are not covariates of the formula ({xn})")
        # each term's column of the design (None: the intercept); X from _design
        # has the model's columns in xnames' order
        self.zcol = [None if t == "1" else xn.index(t) for t in self.terms]
        self.r = len(self.terms)
        self.d = q = self.dl * self.r
        self.own = _sigma_params(q, re.cov)
        self.Ep = _E(q, self.own)
        self.ns = len(self.own)
        self.K = int(n_nodes)
        if self.K ** q > MAX_POINTS:
            raise ValueError(
                f"random slopes with {q} effects per group and random_nodes={self.K} need "
                f"{self.K ** q:,} points per group; at most {MAX_POINTS:,} are supported: use "
                "fewer slopes or outcome levels, or fewer nodes")
        self.z, self.lw, self.kc = _rule(self.K, q)
        size = re.ncnt_u.sum(axis=1)
        self.large = size >= large_rows
        self.rules = [(self.z, self.lw, self.kc, False)]
        if self.K > SMALL_NODES and self.large.any():
            self.rules.append((*_rule(SMALL_NODES, q), True))
        else:
            self.large[:] = False

    # ------------------------------------------------------------ layout
    def labels(self):
        """Each effect's name: the term (ordered), or level:term."""
        def tn(t):
            return "(Intercept)" if t == "1" else t
        if self.kind == "mlogit":
            return [f"{lv}:{tn(t)}" for lv in self.re.p0.levels[1:] for t in self.terms]
        return [tn(t) for t in self.terms]

    def Z(self, X):
        """The rows' random design (n x terms)."""
        return np.column_stack([np.ones(len(X)) if c is None else X[:, c] for c in self.zcol])

    def start(self, theta, S_int):
        """b, the fixed effect, lambda and the modes from the random
        intercepts' joint-mode fit: their intercepts where the design has
        one, the slopes' at 0, with variances of SLOPE_VAR."""
        re, q, r = self.re, self.d, self.r
        P = self.m.P
        b = theta[:P].copy()
        alpha = re._theta_of(theta, self.fe).copy() if self.fe is not None else None
        if self.kind != "mlogit" and alpha is not None:
            alpha = alpha[:, :1]
        A0 = re._theta_of(theta, self.f)[:, re.lv]                 # G x dl
        mode = np.zeros((self.f.G, q))
        S = np.zeros((q, q))
        ti = [t for t, c in enumerate(self.zcol) if c is None]
        for la in range(self.dl):
            for t in range(r):
                if self.zcol[t] is None:
                    mode[:, la * r + t] = A0[:, la]
        for t in range(r):
            for la in range(self.dl):
                i = la * r + t
                if self.zcol[t] is None:
                    for lb in range(self.dl):
                        S[i, lb * r + ti[0]] = S_int[la, lb]
                else:
                    S[i, i] = 0.05
        return b, alpha, self.lam_of(S), mode

    def unpack(self, theta):            # (fit() calls it; the start is set by start())
        return self._start

    def pack(self, b, alpha, mode):
        return b

    # ------------------------------------------------------------ the rows
    def _base(self, b, alpha, X, u, off=None):
        if self.kind == "mlogit":
            return super()._base(b, alpha, X, u)
        base = np.zeros(len(u)) if off is None else np.asarray(off, float).copy()
        if alpha is not None:
            base += alpha[self.fe.of_unit[u], 0]
        return base

    def centre(self, b, alpha, mode):
        G, q, r = self.f.G, self.d, self.r
        score, curv = np.zeros((G, q)), np.zeros((G, q, q))
        self.driver.passes += 1
        for X, y, w, u, off in self._chunks(200_000):
            Z = self.Z(X)
            if self.kind == "mlogit":
                E = self._base(b, alpha, X, u)
                M = mode[u].reshape(len(u), self.dl, r)
                E[:, 1:] += np.einsum("nlt,nt->nl", M, Z)
                Pr, _ = self._probs(E)
                R = -Pr
                R[np.arange(len(y)), y] += 1.0
                p = Pr[:, 1:]
                Wm = -p[:, :, None] * p[:, None, :]
                Wm[:, np.arange(self.dl), np.arange(self.dl)] += p
                sc = (R[:, 1:, None] * Z[:, None, :]).reshape(len(y), q) * w[:, None]
                cv = (Wm[:, :, None, :, None] * Z[:, None, :, None, None]
                      * Z[:, None, None, None, :]).reshape(len(y), q, q) * w[:, None, None]
            else:
                eta = self._base(b, alpha, X, u, off) + (mode[u] * Z).sum(axis=1)
                rr = self.m.rows(b, X, y, w, offset=eta, hessian=True)
                sc = rr["s"][:, None] * Z
                cv = rr["h"][:, None, None] * Z[:, :, None] * Z[:, None, :]
            score += _bysum(u, sc, G)
            curv += _bysum(u, cv.reshape(len(y), -1), G).reshape(G, q, q)
        return score, curv

    def _accumulate(self, out, c, X, y, w, u, z, lw, kc, off=None):
        q = self.d
        Lam, ldL, mode, Rg, logdetR = c.Lam, c.ldL, c.mode, c.Rg, c.logdetR
        st = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
        gu = u[st]
        gc = len(st)
        A, vx = np.empty((gc, len(z), q)), np.empty((gc, len(z)))
        _points(mode, Rg, logdetR, gu.astype(np.int64), np.ascontiguousarray(z, dtype=np.float64),
                np.ascontiguousarray(lw, dtype=np.float64), np.ascontiguousarray(Lam),
                0.5 * ldL - q * HALF_LOG_2PI, A, vx)
        Z = np.ascontiguousarray(self.Z(X))
        if self.kind == "mlogit":
            return self._accumulate_mnl(out, c, X, y, w, u, A, vx, Z, st, gu, gc, kc)
        return self._accumulate_ord(out, c, X, y, w, u, A, vx, Z, st, gu, gc, kc, off)

    def _accumulate_ord(self, out, c, X, y, w, u, A, vx, Z, st, gu, gc, kc, off):
        """The ordered (and binary) chunk: `_ord_kernel_z`, then the
        coefficients' and Sigma's gradient and information, and the fixed
        effect's blocks."""
        q, P = self.d, self.m.P
        Kt = A.shape[1]
        coef, jj, ll_, Sv, nE = c.coef, c.jj, c.ll_, c.Sv, c.nE
        b, alpha = c.b, c.alpha
        m = self.m
        n = len(y)
        K, C = m.K, m.C
        X = np.ascontiguousarray(X, dtype=np.float64)
        E0 = self._base(b, alpha, X, u, off) + X @ b[:K]
        logL, pi = np.empty(gc), np.empty((gc, Kt))
        Wd, slbar = np.empty((gc, Kt, nE)), np.empty((gc, len(coef)))
        Sbar, vdS, fev = np.empty(gc), np.empty(gc), np.empty((gc, nE))
        R, Qb = np.empty((n, 7)), np.empty((n, C))
        sc0, cv0 = np.empty((gc, q)), np.empty((gc, q, q))
        pm, pc = np.empty((gc, q)), np.empty((gc, q, q))
        wv = np.ones(n) if w is None else np.ascontiguousarray(w, dtype=np.float64)
        _ord_kernel_z(np.ascontiguousarray(E0), np.ascontiguousarray(A), Z,
                      np.ascontiguousarray(vx), np.ascontiguousarray(m.cuts(b), dtype=np.float64),
                      y.astype(np.int64), wv, X, np.r_[st, n].astype(np.int64), kc,
                      m.link.name == "probit", m.hmode(True), bool(m.fix_first),
                      np.ascontiguousarray(coef, dtype=np.float64),
                      np.ascontiguousarray(Sv, dtype=np.float64), np.asarray(jj, np.int64),
                      np.asarray(ll_, np.int64), logL, pi, Wd, slbar, Sbar, vdS, fev, R, Qb,
                      sc0, cv0, pm, pc)
        if not np.all(np.isfinite(logL)):
            return False
        out.ll += float(logL.sum())
        out.n_groups += gc
        rbar = dict(zip(("s", "ga", "gb", "h", "haa", "hbb", "hab"), R.T))
        rbar["Q"] = Qb
        rbar["lp"] = np.zeros(n)
        _, g_c, H_c = m._assemble(rbar, X, y, None, True)
        out.gE[:P] += g_c
        out.JEE[:P, :P] += H_c
        out.gE[P:] += slbar.sum(axis=0)
        Wd = Wd.reshape(gc * Kt, nE)
        out.JEE -= Wd.T @ Wd
        fe = self.fe
        if fe is not None:
            Gf = fe.G
            cell = fe.of_unit[gu]
            rcell = fe.of_unit[u]
            out.ga[:, 0] += np.bincount(cell, Sbar, Gf)
            out.Jaa[:, 0, 0] += np.bincount(rcell, rbar["h"], Gf) - np.bincount(cell, vdS, Gf)
            Qm = Qb[:, 1:] if m.fix_first else Qb
            out.JaE[:, 0, :P] += _bysum(rcell, np.column_stack([X * rbar["h"][:, None], Qm]), Gf)
            out.JaE[:, 0, :] -= _bysum(cell, fev, Gf)
        out.score0[gu] = sc0
        out.curv0[gu] = cv0
        out.pmean[gu] = pm
        out.pcov[gu] = pc
        return True

    def _accumulate_mnl(self, out, c, X, y, w, u, A, vx, Z, st, gu, gc, kc):
        """The multinomial chunk (as `_AdaptiveMNL._accumulate`, the kernel
        with the row design)."""
        J, q, P = self.J, self.d, self.m.P
        Kx = P // (J - 1)
        Kt = A.shape[1]
        coef, jj, ll_, Sv, nE = c.coef, c.jj, c.ll_, c.Sv, c.nE
        b, alpha = c.b, c.alpha
        fe = self.fe
        n = len(y)
        logL, pi = np.empty(gc), np.empty((gc, Kt))
        Pbar, Wbar = np.empty((n, J)), np.empty((n, J, J))
        Ds, Db = np.empty((gc, Kt, J)), np.empty((gc, Kt, J - 1, Kx))
        sc0, cv0 = np.empty((gc, q)), np.empty((gc, q, q))
        _mnl_kernel_z(np.ascontiguousarray(self._base(b, alpha, X, u)), np.ascontiguousarray(A),
                      Z, self.r, np.ascontiguousarray(vx), y.astype(np.int64),
                      np.ascontiguousarray(w, dtype=np.float64),
                      np.ascontiguousarray(X, dtype=np.float64), np.r_[st, n].astype(np.int64),
                      kc, logL, pi, Pbar, Wbar, Ds, Db, sc0, cv0)
        if not np.all(np.isfinite(logL)):
            return False
        out.ll += float(logL.sum())
        out.n_groups += gc
        Wbar *= -1.0
        Wbar[:, np.arange(J), np.arange(J)] += Pbar
        Rbar = -Pbar
        Rbar[np.arange(n), y] += 1.0
        out.gE[:P] += (X.T @ (Rbar[:, 1:] * w[:, None])).T.ravel()
        Ww = Wbar * w[:, None, None]
        for a in range(1, J):
            sa = slice((a - 1) * Kx, a * Kx)
            for cc in range(a, J):
                sc = slice((cc - 1) * Kx, cc * Kx)
                H = (X * Ww[:, a, cc][:, None]).T @ X
                out.JEE[sa, sc] += H
                if cc != a:
                    out.JEE[sc, sa] += H.T
        sl = coef[None, None, :] * (Sv[None, None, :] - A[:, :, jj] * A[:, :, ll_])
        slbar = (pi[:, None, :] @ sl)[:, 0, :]
        out.gE[P:] += slbar.sum(axis=0)
        Db = Db.reshape(gc, Kt, P)
        Dl = sl - slbar[:, None, :]
        DE = np.concatenate([Db, Dl], axis=2)
        Wd = (np.sqrt(pi)[:, :, None] * DE).reshape(gc * Kt, nE)
        out.JEE -= Wd.T @ Wd
        if fe is not None:
            Gf = fe.G
            cell = fe.of_unit[gu]
            rcell = fe.of_unit[u]
            out.ga += _bysum(rcell, Rbar * w[:, None], Gf)
            out.Jaa += _bysum(rcell, Ww.reshape(n, -1), Gf).reshape(Gf, J, J)
            Bv = (Ww[:, :, 1:, None] * X[:, None, None, :]).reshape(n, J, P)
            out.JaE[:, :, :P] += _bysum(rcell, Bv.reshape(n, -1), Gf).reshape(Gf, J, P)
            Dsp = np.swapaxes(Ds * pi[:, :, None], 1, 2)
            out.Jaa -= _bysum(cell, (Dsp @ Ds).reshape(gc, -1), Gf).reshape(Gf, J, J)
            out.JaE -= _bysum(cell, (Dsp @ DE).reshape(gc, -1), Gf).reshape(Gf, J, nE)
        out.score0[gu] = sc0
        out.curv0[gu] = cv0
        Ap = np.swapaxes(A * pi[:, :, None], 1, 2)
        pm = Ap.sum(axis=2)
        out.pmean[gu] = pm
        out.pcov[gu] = Ap @ A - pm[:, :, None] * pm[:, None, :]
        return True

    # ------------------------------------------------------------ results
    def result_spec(self, res):
        """The `ranef` table (each group's modes and conditional sds, and
        its rows), the effects as `predict` adds them, and each group's
        conditional covariance."""
        re, f = self.re, self.f
        name = f.name
        mode, cov = res["mode"], res["moments"].pcov
        labels = self.labels()
        cols = [f"re_{name}" if lab == "(Intercept)" else
                f"re_{name}:{lab.replace(':(Intercept)', '')}" for lab in labels]
        sds = ["sd_" + c[3:] for c in cols]
        sd = np.sqrt(np.clip(np.diagonal(cov, axis1=1, axis2=2), 0, None))
        n = pl.Series(f.ncnt.sum(axis=1).astype(np.int64))
        from ..core.store import CODE
        table = re.p0.fe_tables[name].with_columns(
            **{c: pl.Series(mode[:, k]) for k, c in enumerate(cols)},
            **{c: pl.Series(sd[:, k]) for k, c in enumerate(sds)}, n_obs=n)
        table = table.filter(pl.Series(f.keep)).drop(CODE)
        # each column's level (1.. for the multinomial, 0 for the ordered) and term
        layout = [((k // self.r) + (1 if self.kind == "mlogit" else 0), self.terms[k % self.r])
                  for k in range(self.d)]
        return {"ranef": {name: table}, "ranef_cov": {name: np.ascontiguousarray(cov[f.keep])},
                "slope_layout": {name: layout}, "slope_values": {name: cols}}


def _fit_slopes(driver, re, fit, re_info, fml, tol, n_nodes, terms, log, logger):
    """`_AdaptiveSlopes` from the random intercepts' joint-mode fit `fit`;
    sets Sigma, the coefficients' variance and the results' pieces on `re`
    and returns the fit and `re_info` updated as `_fit_one` uses them."""
    from dataclasses import replace

    from .fit import _sigma_summary

    ims = _AdaptiveSlopes(re, driver, n_nodes, log, logger, terms)
    q = ims.d
    log(f"random slopes: {', '.join(ims.labels())} by {ims.f.name}; adaptive Gauss-Hermite "
        f"quadrature, {n_nodes} nodes per dimension ({len(ims.z) - (n_nodes % 2 == 0):,} "
        "points per group)")
    ims._start = ims.start(fit.theta, re.random[0].S)
    res = ims.fit(fit.theta, tol)
    own, Ep = ims.own, ims.Ep
    S = np.linalg.inv(ims._Lam(res["lam"]))
    S = 0.5 * (S + S.T)
    f = ims.f
    f.S = S
    re.adaptive_V = res["Vb_cond"]
    re.method = "adaptive"
    T = -np.einsum("ab,pbc,cd->pad", S, Ep, S)
    T = np.array([[T[p, j, l] for p in range(len(own))] for j, l in own])
    VS = T @ res["V_lam"] @ T.T
    Ti = np.linalg.inv(T)
    Jb = res["dbdlam"] @ Ti
    E = res["E"]
    gS = Ti.T @ E.gE[ims.m.P:]
    sigma = {"by_factor": {f.name: _sigma_summary(S, VS, own)}, "vcov": VS, "J": Jb,
             "gradient": [float(x) for x in gS]}
    # the parameter vector of `re` with these estimates (the fixed effect's levels)
    P = ims.m.P
    theta = fit.theta.copy()
    theta[:P] = res["b"]
    if ims.fe is not None:
        k = re.factors.index(ims.fe)
        o = P + re.offsets[k]
        fe = ims.fe
        a = res["alpha"]
        full = np.zeros((fe.G, re.m))
        full[:, :a.shape[1]] = a
        theta[o:o + fe.G * re.m] = np.where(fe.free, full, fe.fixed).ravel()
    re.n_params = ims.ns + (int(ims.fe.free.sum()) if ims.fe is not None else 0)
    re.slope_spec = ims.result_spec(res)
    re.slope_labels = ims.labels()
    re.slope_terms = list(terms)
    from ..draws import adaptive_mnl_draw_info
    re.draw_info = adaptive_mnl_draw_info(ims, res["E"], res["b"], res["alpha"], res["lam"])
    info = res["info"]
    re_info = dict(re_info)
    re_info.update(info)
    re_info["converged"] = info["adaptive_converged"]
    re_info["passes"] += info["adaptive_passes"]
    re_info["newton_steps"] += info["adaptive_steps"]
    re_info["seconds"] = round(re_info["seconds"] + info["adaptive_seconds"], 3)
    re_info["loglik"] = res["ll"]
    re_info["sigma"] = sigma
    re_info["random_dim"] = q
    fit = replace(fit, theta=theta, converged=info["adaptive_converged"],
                  decrement=info["adaptive_decrement"])
    return fit, re_info


def _squarem_fit(self, theta, tol, maxiter=300, node_tol=1e-4, halvings=30):
    """`_AdaptiveMNL.fit`'s rounds -- a Newton step with the points held,
    then the points moved -- accelerated by SQUAREM (Varadhan and Roland,
    2008) on the parameters: with tiny groups each group's posterior moves
    with Sigma, so the rounds alone converge linearly and slowly. Every two
    rounds the parameters are extrapolated along their path, the points
    re-centred there, and the extrapolation kept if its log-likelihood is
    at least the second round's."""
    import time

    from hdfe_stream.core.report import warn
    t0 = time.time()
    p_start = self.driver.passes
    b, alpha, lam, mode = self.unpack(theta)
    keep = self.f.keep
    P = self.m.P

    def centred(b, alpha, lam, mode, steps=2):
        for _ in range(steps):
            st_, Rg_, sd_ = self._proposal(b, alpha, lam, mode)
            mode = np.where(keep[:, None], mode + np.clip(st_, -3 * sd_, 3 * sd_), 0.0)
        _, Rg_, sd_ = self._proposal(b, alpha, lam, mode)
        return mode, Rg_, sd_

    mode, Rg, sd = centred(b, alpha, lam, mode)
    logdetR = np.log(np.diagonal(Rg, axis1=1, axis2=2)).sum(axis=1)
    E = self.evaluate(b, alpha, lam, mode, Rg, logdetR)
    if E is None:
        raise RuntimeError("adaptive quadrature: a probability of 0 at the start")

    def round_(b, alpha, lam, mode, Rg, sd, logdetR, E):
        """One round: (state after, decrement, points moved, converged)."""
        step_m, Rn, sdn = self._proposal(b, alpha, lam, mode, E.score0, E.curv0)
        moved = float(np.max(np.abs(step_m[keep]) / sd[keep])) if keep.any() else 0.0
        st = self.step(E)
        if st is None:
            raise RuntimeError("adaptive quadrature: the fixed effect's information is not "
                               "positive")
        dE, da, dec, pd = st
        if pd and dec < tol and moved < node_tol:
            return (b, alpha, lam, mode, Rg, sd, logdetR, E), dec, moved, True
        if not (pd and dec < tol):
            t = 1.0
            for _ in range(halvings):
                b_c, lam_c = b + t * dE[:P], lam + t * dE[P:]
                alpha_c = alpha + t * da if alpha is not None else None
                if self.feasible(lam_c):
                    Ec = self.evaluate(b_c, alpha_c, lam_c, mode, Rg, logdetR)
                    if Ec is not None and Ec.ll >= E.ll - 1e-10 * abs(E.ll):
                        break
                t /= 2
            else:
                return None, dec, moved, False
            b, alpha, lam, E = b_c, alpha_c, lam_c, Ec
            step_m, Rn, sdn = self._proposal(b, alpha, lam, mode, E.score0, E.curv0)
        mode = np.where(keep[:, None], mode + np.clip(step_m, -3 * sd, 3 * sd), 0.0)
        Rg, sd = Rn, sdn
        logdetR = np.log(np.diagonal(Rg, axis1=1, axis2=2)).sum(axis=1)
        E = self.evaluate(b, alpha, lam, mode, Rg, logdetR)
        if E is None:
            raise RuntimeError("adaptive quadrature: a probability of 0 after moving the points")
        return (b, alpha, lam, mode, Rg, sd, logdetR, E), dec, moved, False

    def vec(s):
        return np.r_[s[0], s[1].ravel() if s[1] is not None else [], s[2]]

    def unvec(v, like):
        b_ = v[:P]
        o = P
        a_ = None
        if like[1] is not None:
            a_ = v[o:o + like[1].size].reshape(like[1].shape)
            o += like[1].size
        return b_, a_, v[o:]

    state = (b, alpha, lam, mode, Rg, sd, logdetR, E)
    converged, dec, it, path, n_acc = False, np.inf, 0, [], 0
    while it < maxiter:
        it += 1
        s1, dec, moved, done = round_(*state)
        path.append(state[7].ll)
        self.log(f"  adaptive round {it}: log-likelihood {state[7].ll:.12g}, decrement "
                 f"{dec:.3e}, points moved {moved:.2e} (sd)")
        if s1 is None:
            warn("adaptive quadrature: the line search failed", self.logger)
            break
        if done:
            converged = True
            state = s1
            break
        it += 1
        s2, dec2, moved2, done2 = round_(*s1)
        if s2 is None:
            state = s1
            warn("adaptive quadrature: the line search failed", self.logger)
            break
        if done2:
            converged, state = True, s2
            break
        v0, v1, v2 = vec(state), vec(s1), vec(s2)
        r, v = v1 - v0, v2 - 2 * v1 + v0
        nr, nv = np.linalg.norm(r), np.linalg.norm(v)
        state = s2
        if nv > 0:
            alpha_ = min(-nr / nv, -1.0)
            v3 = v0 - 2 * alpha_ * r + alpha_ ** 2 * v
            b3, a3, l3 = unvec(v3, s2)
            if alpha_ < -1.0 and self.feasible(l3):
                try:
                    m3, R3, sd3 = centred(b3, a3, l3, s2[3], steps=1)
                    ld3 = np.log(np.diagonal(R3, axis1=1, axis2=2)).sum(axis=1)
                    E3 = self.evaluate(b3, a3, l3, m3, R3, ld3)
                except np.linalg.LinAlgError:
                    E3 = None
                if E3 is not None and E3.ll >= s2[7].ll:
                    state = (b3, a3, l3, m3, R3, sd3, ld3, E3)
                    n_acc += 1
    if not converged:
        warn(f"adaptive quadrature did not converge in {it} rounds (decrement {dec:.2g})",
             self.logger)
    b, alpha, lam, mode, Rg, sd, logdetR, E = state
    Vb_cond, V_lam, dbdlam = self.inference(E)
    ll, moments = E.ll, E
    if len(self.rules) > 1:
        rules, self.rules = self.rules, self.rules[:1]
        try:
            Ef = self.evaluate(b, alpha, lam, mode, Rg, logdetR)
        finally:
            self.rules = rules
        if Ef is not None:
            ll, moments = Ef.ll, Ef
    return dict(theta=b, b=b, alpha=alpha, lam=lam, mode=mode, E=E, moments=moments,
                Vb_cond=Vb_cond, V_lam=V_lam, dbdlam=dbdlam, ll=ll,
                info={"adaptive_rounds": it, "adaptive_steps": it,
                      "adaptive_converged": converged, "adaptive_decrement": dec,
                      "adaptive_nodes": self.K, "adaptive_loglik_path": path,
                      "adaptive_extrapolations": n_acc,
                      "adaptive_passes": self.driver.passes - p_start,
                      "adaptive_seconds": round(time.time() - t0, 3)})


_AdaptiveSlopes.fit = _squarem_fit
