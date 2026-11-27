"""Fused numba passes for the multinomial and ordered logit with one absorbed
fixed effect (see fe.py for the model and its arrow-shaped information).

A pass without a fixed effect (`models._mnl_pass_tri`, `models._ord_pass`)
needs sums over rows only. With one it also needs, for every group g, the
group's score (`ga`), its information block (`Haa`) and the block against the
slopes (`B`): sums over the group's rows that, in a block of rows in any
order, many threads would update at once. Pass 0 therefore writes the rows
grouped by the fixed effect (`store._build(sort_fe=True)`, hdfe-stream's
`iter_groups`): each group is a run of consecutive rows of one block. A pass
cuts a block into parts at group boundaries and hands the parts to the
threads; a thread owns whole groups, sums a group's rows in a few local
arrays, and adds them to the group's entries once, at the end of its run, so
no entry is ever shared and nothing is locked or copied per thread. What is
per thread is only what the passes without a fixed effect also keep: the
slopes' information as the outer-product accumulators.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from .models import _assemble_tri, _ord_terms, _parts


def part_starts(f, n_parts):
    """Row positions that cut the block into `n_parts` parts of about equal
    numbers of rows, each cut moved forward to the start of a group (the
    codes `f` are in runs): ``n_parts + 1`` nondecreasing positions from 0
    to n."""
    n = len(f)
    cuts = np.flatnonzero(f[1:] != f[:-1]) + 1
    targets = (np.arange(1, n_parts, dtype=np.int64) * n) // n_parts
    idx = np.searchsorted(cuts, targets)
    snapped = np.where(idx < len(cuts), cuts[np.minimum(idx, max(len(cuts) - 1, 0))], n) \
        if len(cuts) else np.full(len(targets), n)
    return np.r_[0, snapped, n].astype(np.int64)


# ---------------------------------------------------------------------- ordered
@nb.njit(parallel=True, cache=True, error_model="numpy")
def _ord_fe_pass(X, theta, colmap, cuts, eff, keep, f, y, w, off, probit, hessian, starts,
                 ll_t, g_t, A_t, XC_t, CC_t, ga, Haa, Bg):
    """`models._ord_pass` with a group effect in the linear predictor (``eff[g]``,
    added to x'b) and the groups' sums: for each group g in a run of rows, its
    score in eta (``ga[g]``, summed ``s``), its information in eta (``Haa[g]``,
    summed ``h``), and the information between it and each slope and
    cutpoint (``Bg[g]``: summed ``h x``, then summed ``Q``), added once per
    run. Rows of a group with ``keep[g]`` false are skipped. `starts` cut the
    rows into parts at group boundaries."""
    K = len(colmap)
    C = len(cuts)
    P = K + C
    nacc = A_t.shape[1]
    n_parts = len(ll_t)
    has_w = len(w) > 0
    has_off = len(off) > 0
    for t in nb.prange(n_parts):
        u = np.empty(K)
        g = np.zeros(P)
        acc = np.zeros(nacc)
        xc = np.zeros((C, K))
        cc = np.zeros((C, C))
        bloc = np.zeros(P)
        sloc = 0.0
        hloc = 0.0
        cur = -1
        llt = 0.0
        for i in range(starts[t], starts[t + 1]):
            gi = f[i]
            if gi != cur:
                if cur >= 0 and keep[cur]:
                    ga[cur] += sloc
                    if hessian:
                        Haa[cur] += hloc
                        for k in range(P):
                            Bg[cur, k] += bloc[k]
                cur = gi
                sloc = 0.0
                hloc = 0.0
                for k in range(P):
                    bloc[k] = 0.0
            if not keep[gi]:
                continue
            wt = w[i] if has_w else 1.0
            yi = y[i]
            eta = eff[gi] + (off[i] if has_off else 0.0)
            for k in range(K):
                u[k] = X[i, colmap[k]]
                eta += u[k] * theta[k]
            a = cuts[yi] - eta if yi < C else np.inf
            b = cuts[yi - 1] - eta if yi > 0 else -np.inf
            Pr, ga_, gb_, haa, hbb, hab = _ord_terms(a, b, probit, hessian)
            llt += wt * np.log(Pr)
            s = -(ga_ + gb_) * wt
            sloc += s
            for k in range(K):
                g[k] += s * u[k]
            if yi < C:
                g[K + yi] += ga_ * wt
            if yi > 0:
                g[K + yi - 1] += gb_ * wt
            if hessian:
                h = wt * max(-(haa + 2.0 * hab + hbb), 0.0)
                hloc += h
                k = 0
                for c1 in range(K):
                    hx = h * u[c1]
                    bloc[c1] += hx
                    for c2 in range(c1, K):
                        acc[k] += hx * u[c2]
                        k += 1
                if yi < C:
                    q = wt * (haa + hab)
                    for k in range(K):
                        xc[yi, k] += q * u[k]
                    cc[yi, yi] -= wt * haa
                    bloc[K + yi] += q
                if yi > 0:
                    q = wt * (hab + hbb)
                    for k in range(K):
                        xc[yi - 1, k] += q * u[k]
                    cc[yi - 1, yi - 1] -= wt * hbb
                    bloc[K + yi - 1] += q
                    if yi < C:
                        cc[yi - 1, yi] -= wt * hab
        if cur >= 0 and keep[cur]:
            ga[cur] += sloc
            if hessian:
                Haa[cur] += hloc
                for k in range(P):
                    Bg[cur, k] += bloc[k]
        ll_t[t] = llt
        for k in range(P):
            g_t[t, k] = g[k]
        if hessian:
            for k in range(nacc):
                A_t[t, k] = acc[k]
            for c in range(C):
                for k in range(K):
                    XC_t[t, c, k] = xc[c, k]
                for d in range(C):
                    CC_t[t, c, d] = cc[c, d]


class OrderedPass:
    """One pass of the ordered logit (or probit) with an absorbed fixed
    effect, block by block through the kernel: `add` each block, then
    `finish` into the arrays the Newton driver expects (fe.py)."""

    def __init__(self, fe, theta, A, hessian):
        m = fe.model
        self.fe, self.m, self.hessian = fe, m, hessian
        K, C = m.K, m.C
        th = np.ascontiguousarray(theta[:m.P], np.float64)
        self.th, self.cuts = th, np.ascontiguousarray(m.cuts(th))
        self.eff = np.ascontiguousarray(A[:, 0], np.float64)
        self.keep = np.ascontiguousarray(fe.keep_g)
        G = fe.G
        self.ll = 0.0
        self.g = np.zeros(K + C)
        self.ga = np.zeros(G)
        self.Haa = np.zeros(G if hessian else 0)
        self.Bg = np.zeros((G if hessian else 0, K + C))
        self.S = np.zeros(K * (K + 1) // 2 if hessian else 0)
        self.XC = np.zeros((C if hessian else 0, K))
        self.CC = np.zeros((C if hessian else 0, C))

    def add(self, blk):
        m = self.m
        K, C = m.K, m.C
        f = np.ascontiguousarray(blk["fe"], np.int64)
        n = len(f)
        w, off = blk.get("w"), blk.get("off")
        nacc = K * (K + 1) // 2 if self.hessian else 0
        n_parts = _parts(max(1, n // 256), int(np.sqrt(nacc + C * K + C * C + K + C)))
        starts = part_starts(f, n_parts)
        ll_t, g_t = np.zeros(n_parts), np.zeros((n_parts, K + C))
        A_t = np.zeros((n_parts, nacc))
        XC_t = np.zeros((n_parts, C if self.hessian else 0, K))
        CC_t = np.zeros((n_parts, C if self.hessian else 0, C))
        empty = np.empty(0)
        with nb.parallel_chunksize(1):
            _ord_fe_pass(np.ascontiguousarray(blk["X"]), self.th, np.asarray(m.cols, np.int64),
                         self.cuts, self.eff, self.keep, f, blk["y"],
                         empty if w is None else np.ascontiguousarray(w, np.float64),
                         empty if off is None else np.ascontiguousarray(off, np.float64),
                         m.link.name == "probit", m.hmode(self.hessian), starts, ll_t, g_t,
                         A_t, XC_t, CC_t, self.ga, self.Haa, self.Bg)
        ll = float(ll_t.sum())
        if not np.isfinite(ll):
            return -np.inf
        self.g += g_t.sum(axis=0)
        if self.hessian:
            self.S += A_t.sum(axis=0)
            self.XC += XC_t.sum(axis=0)
            self.CC += CC_t.sum(axis=0)
        return ll

    def finish(self, gt, ga, Htt, Haa, B):
        """Add the pass's sums to the driver's arrays (`gt` P, `ga` G x 1,
        `Htt` P x P, `Haa` G x 1 x 1, `B` G x 1 x P); with cut1 fixed at 0,
        its parameter is left out of each."""
        m = self.m
        K, C = m.K, m.C
        keep = np.r_[np.arange(K), K + np.arange(1 if m.fix_first else 0, C)]
        gt += self.g[keep]
        ga[:, 0] += self.ga
        if self.hessian:
            S = np.zeros((K, K))
            S[np.triu_indices(K)] = self.S
            H = np.zeros((K + C, K + C))
            H[:K, :K] = S + np.triu(S, 1).T
            H[:K, K:] = self.XC.T
            H[K:, :K] = self.XC
            H[K:, K:] = np.triu(self.CC) + np.triu(self.CC, 1).T
            Htt += H[np.ix_(keep, keep)]
            Haa[:, 0, 0] += self.Haa
            B[:, 0, :] += self.Bg[:, keep]


# ------------------------------------------------------------------ multinomial
@nb.njit(parallel=True, cache=True, error_model="numpy")
def _mnl_fe_pass(X, theta, colmap, A, keep, f, y, w, hessian, starts, ll_t, g_t, acc_t,
                 ga, Haa, B):
    """`models._mnl_pass_tri` with the group effects ``A[g, l]`` of all J
    outcome levels in the linear predictors (level l = 0 has x'b = 0, the
    others x'b_l, each with the K columns `colmap`; an absent level has
    effect -inf), and the groups' sums, added once per run of a group's rows:
    the score in the effects ``ga[g, l]`` = sum w (1[y = l] - p_l); their
    information ``Haa[g, l, m]`` = sum w p_l (1[l = m] - p_m); and its block
    against the slopes ``B[g, l, (j - 1) K + k]`` = sum w x_k p_j (1[j = l] -
    p_l). Rows of a group with ``keep[g]`` false are skipped."""
    J = A.shape[1]
    J1 = J - 1
    K = len(colmap)
    P = J1 * K
    ntri = K * (K + 1) // 2
    n_parts = len(ll_t)
    has_w = len(w) > 0
    for t in nb.prange(n_parts):
        u = np.empty(K)
        xxt = np.empty(ntri)
        E = np.empty(J)
        p = np.empty(J)
        g = np.zeros(P)
        acc = np.zeros((J1 * (J1 + 1) // 2, ntri))
        gl = np.zeros(J)
        Hl = np.zeros((J, J))
        Bl = np.zeros((J, P))
        cur = -1
        llt = 0.0
        for i in range(starts[t], starts[t + 1]):
            gi = f[i]
            if gi != cur:
                if cur >= 0 and keep[cur]:
                    for l in range(J):
                        ga[cur, l] += gl[l]
                    if hessian:
                        for l in range(J):
                            for q in range(J):
                                Haa[cur, l, q] += Hl[l, q]
                            for k in range(P):
                                B[cur, l, k] += Bl[l, k]
                cur = gi
                for l in range(J):
                    gl[l] = 0.0
                    for q in range(J):
                        Hl[l, q] = 0.0
                    for k in range(P):
                        Bl[l, k] = 0.0
            if not keep[gi]:
                continue
            wi = w[i] if has_w else 1.0
            for c in range(K):
                u[c] = X[i, colmap[c]]
            E[0] = A[gi, 0]
            m = E[0]
            for j in range(1, J):
                e = 0.0
                o = (j - 1) * K
                for c in range(K):
                    e += u[c] * theta[o + c]
                E[j] = e + A[gi, j]
                if E[j] > m:
                    m = E[j]
            s = 0.0
            for j in range(J):
                p[j] = np.exp(E[j] - m)
                s += p[j]
            yi = y[i]
            llt += wi * (E[yi] - m - np.log(s))
            inv = 1.0 / s
            for j in range(J):
                p[j] *= inv
            for j in range(J):
                r = wi * ((1.0 if yi == j else 0.0) - p[j])
                gl[j] += r
                if j > 0:
                    o = (j - 1) * K
                    for c in range(K):
                        g[o + c] += r * u[c]
            if hessian:
                k = 0
                for c1 in range(K):
                    x1 = u[c1]
                    for c2 in range(c1, K):
                        xxt[k] = x1 * u[c2]
                        k += 1
                pr = 0
                for j in range(1, J):
                    pj = wi * p[j]
                    for l in range(j, J):
                        v = pj * ((1.0 if j == l else 0.0) - p[l])
                        for k in range(ntri):
                            acc[pr, k] += v * xxt[k]
                        pr += 1
                for l in range(J):
                    pl_ = wi * p[l]
                    for q in range(J):
                        Hl[l, q] += pl_ * ((1.0 if l == q else 0.0) - p[q])
                    for j in range(1, J):
                        v = wi * p[j] * ((1.0 if j == l else 0.0) - p[l])
                        o = (j - 1) * K
                        for c in range(K):
                            Bl[l, o + c] += v * u[c]
        if cur >= 0 and keep[cur]:
            for l in range(J):
                ga[cur, l] += gl[l]
            if hessian:
                for l in range(J):
                    for q in range(J):
                        Haa[cur, l, q] += Hl[l, q]
                    for k in range(P):
                        B[cur, l, k] += Bl[l, k]
        ll_t[t] = llt
        for k in range(P):
            g_t[t, k] = g[k]
        if hessian:
            acc_t[t] = acc


class MultinomialPass:
    """`OrderedPass` for the multinomial logit: the slopes' sums are kept
    here, the groups' go straight into the driver's arrays."""

    def __init__(self, fe, theta, A, hessian, ga, Haa, B):
        m = fe.model
        self.fe, self.m, self.hessian = fe, m, hessian
        self.A = np.ascontiguousarray(A, np.float64)
        self.th = np.ascontiguousarray(theta[:m.P], np.float64)
        self.keep = np.ascontiguousarray(fe.keep_g)
        none = np.zeros((0, 0, 0))
        self.ga, self.Haa, self.B = ga, (Haa if hessian else none), (B if hessian else none)
        J1, K = m.J - 1, m.K
        self.ll = 0.0
        self.g = np.zeros(m.P)
        self.acc = np.zeros((J1 * (J1 + 1) // 2, K * (K + 1) // 2)) if hessian else None

    def add(self, blk):
        m = self.m
        J1, K = m.J - 1, m.K
        f = np.ascontiguousarray(blk["fe"], np.int64)
        n = len(f)
        w = blk.get("w")
        ntri = K * (K + 1) // 2
        npair = J1 * (J1 + 1) // 2
        n_parts = _parts(max(1, n // 256), int(np.sqrt(npair * ntri)) if self.hessian else m.P)
        starts = part_starts(f, n_parts)
        ll_t, g_t = np.zeros(n_parts), np.zeros((n_parts, m.P))
        acc_t = np.zeros((n_parts, npair, ntri) if self.hessian else (n_parts, 0, 0))
        with nb.parallel_chunksize(1):
            _mnl_fe_pass(np.ascontiguousarray(blk["X"]), self.th, np.asarray(m.cols, np.int64),
                         self.A, self.keep, f, blk["y"],
                         np.empty(0) if w is None else np.ascontiguousarray(w, np.float64),
                         bool(self.hessian), starts, ll_t, g_t, acc_t, self.ga, self.Haa,
                         self.B)
        ll = float(ll_t.sum())
        if not np.isfinite(ll):
            return -np.inf
        self.g += g_t.sum(axis=0)
        if self.hessian:
            self.acc += acc_t.sum(axis=0)
        return ll

    def finish(self, gt, Htt):
        m = self.m
        J1, K = m.J - 1, m.K
        gt += self.g
        if self.hessian:
            pos = np.tile(np.arange(K, dtype=np.int64), (J1, 1))
            Htt += _assemble_tri(self.acc, pos, np.full(J1, K, np.int64),
                                 np.arange(J1 + 1, dtype=np.int64) * K, K)


# ------------------------------------------------- the variances, with a fixed effect
# A row's score for the slopes, net of the part the fixed effects absorb
# (fe.py, `_AbsorbedFE.meat`): s_b - H_bg H_gg^-1 s_g, with BD[g] = H_gg^-1 H_gb
# the group's block. One dense vector of P entries per row, formed while the
# group's block is hot (the rows are grouped), and added to the outer-product
# sums and to each clustering's cluster sums, as meat_kernels.py does without
# a fixed effect.

@nb.njit(cache=True, inline="always", error_model="numpy")
def _mnl_fe_score(X, i, colmap, theta, Arow, frow, BDg, yi, u, E, p, sv):
    """Row i's concentrated score in `sv` (P entries, equation by equation):
    R_j x for the J - 1 equations, less sum_l R_l free_gl BD[g, l, :]; R_l =
    1[y = l] - p_l over all J levels, with the group's effects in the linear
    predictors."""
    J = len(Arow)
    K = len(colmap)
    P = (J - 1) * K
    for c in range(K):
        u[c] = X[i, colmap[c]]
    E[0] = Arow[0]
    m = E[0]
    for j in range(1, J):
        e = 0.0
        o = (j - 1) * K
        for c in range(K):
            e += u[c] * theta[o + c]
        E[j] = e + Arow[j]
        if E[j] > m:
            m = E[j]
    s = 0.0
    for j in range(J):
        p[j] = np.exp(E[j] - m)
        s += p[j]
    inv = 1.0 / s
    for j in range(J):
        p[j] = (1.0 if yi == j else 0.0) - p[j] * inv            # R_j
    for j in range(1, J):
        o = (j - 1) * K
        for c in range(K):
            sv[o + c] = p[j] * u[c]
    for l in range(J):
        sa = p[l] * frow[l]
        if sa != 0.0:
            for k in range(P):
                sv[k] -= sa * BDg[l, k]


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _mnl_fe_meat_pass(X, theta, colmap, A, free, BD, keep, f, y, w, wmode, cl, cloff, hetero,
                      starts, M_t, U_t):
    """The outer products of the concentrated scores (``M_t[t]``, upper
    triangle, weighted by m: 1, w or w^2 for `wmode` 0, 2, 1) and the clusters'
    sums (``U_t[t]``, weighted by w) over the parts of a block of rows grouped
    by the fixed effect (`starts` cut at group boundaries)."""
    J = A.shape[1]
    K = len(colmap)
    P = (J - 1) * K
    ncl = cl.shape[1]
    n_parts = len(starts) - 1
    has_w = wmode > 0
    for t in nb.prange(n_parts):
        u = np.empty(K)
        E = np.empty(J)
        p = np.empty(J)
        sv = np.empty(P)
        M = np.zeros((P, P) if hetero else (0, 0))
        for i in range(starts[t], starts[t + 1]):
            gi = f[i]
            if not keep[gi]:
                continue
            wi = w[i] if has_w else 1.0
            _mnl_fe_score(X, i, colmap, theta, A[gi], free[gi], BD[gi], y[i], u, E, p, sv)
            for k in range(ncl):
                row = cloff[k] + cl[i, k]
                for a in range(P):
                    U_t[t, row, a] += wi * sv[a]
            if hetero:
                mw = 1.0 if wmode == 0 else (wi * wi if wmode == 1 else wi)
                for a in range(P):
                    x = mw * sv[a]
                    for b in range(a, P):
                        M[a, b] += x * sv[b]
        if hetero:
            M_t[t] = M


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _mnl_fe_cluster_pass(X, theta, colmap, A, free, BD, keep, f, y, w, has_w, code, bounds, U):
    """`meat_kernels._mnl_cluster_pass` with a fixed effect: part t owns the
    cluster codes ``bounds[t]`` to ``bounds[t + 1]``. (The rows are grouped by
    the fixed effect, so a part's rows come in runs of one group, whose block
    stays in cache.)"""
    n = len(y)
    J = A.shape[1]
    K = len(colmap)
    P = (J - 1) * K
    n_parts = len(bounds) - 1
    for t in nb.prange(n_parts):
        lo = bounds[t]
        hi = bounds[t + 1]
        u = np.empty(K)
        E = np.empty(J)
        p = np.empty(J)
        sv = np.empty(P)
        for i in range(n):
            c = code[i]
            if c < lo or c >= hi:
                continue
            gi = f[i]
            if not keep[gi]:
                continue
            wi = w[i] if has_w else 1.0
            _mnl_fe_score(X, i, colmap, theta, A[gi], free[gi], BD[gi], y[i], u, E, p, sv)
            for a in range(P):
                U[c, a] += wi * sv[a]


@nb.njit(cache=True, inline="always", error_model="numpy")
def _ord_fe_score(X, i, colmap, theta, cuts, eff_g, offi, BDg, yi, probit, u, sv):
    """Row i's concentrated score in `sv`: s x for the slopes, g_a at the cutpoint
    above the row's level and g_b at the one below (the first cutpoint is fixed
    at 0 and has no entry, so cutpoint c is at K + c - 1), less s BD[g, 0, :];
    s = -(g_a + g_b)."""
    K = len(colmap)
    C = len(cuts)
    eta = eff_g + offi
    for k in range(K):
        u[k] = X[i, colmap[k]]
        eta += u[k] * theta[k]
    a = cuts[yi] - eta if yi < C else np.inf
    b = cuts[yi - 1] - eta if yi > 0 else -np.inf
    Pr, ga, gb, haa, hbb, hab = _ord_terms(a, b, probit, False)
    s = -(ga + gb)
    for k in range(K):
        sv[k] = s * u[k] - s * BDg[k]
    for q in range(C - 1):
        sv[K + q] = -s * BDg[K + q]
    if yi < C and yi >= 1:
        sv[K + yi - 1] += ga
    if yi > 0 and yi - 1 >= 1:
        sv[K + yi - 2] += gb


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _ord_fe_meat_pass(X, theta, colmap, cuts, eff, BD, keep, f, y, w, off, wmode, probit, cl,
                      cloff, hetero, starts, M_t, U_t):
    """`_mnl_fe_meat_pass` for the ordered models (`BD` is G x P)."""
    K = len(colmap)
    C = len(cuts)
    P = K + C - 1
    ncl = cl.shape[1]
    n_parts = len(starts) - 1
    has_w = wmode > 0
    has_off = len(off) > 0
    for t in nb.prange(n_parts):
        u = np.empty(K)
        sv = np.empty(P)
        M = np.zeros((P, P) if hetero else (0, 0))
        for i in range(starts[t], starts[t + 1]):
            gi = f[i]
            if not keep[gi]:
                continue
            wi = w[i] if has_w else 1.0
            _ord_fe_score(X, i, colmap, theta, cuts, eff[gi], off[i] if has_off else 0.0,
                          BD[gi], y[i], probit, u, sv)
            for k in range(ncl):
                row = cloff[k] + cl[i, k]
                for a in range(P):
                    U_t[t, row, a] += wi * sv[a]
            if hetero:
                mw = 1.0 if wmode == 0 else (wi * wi if wmode == 1 else wi)
                for a in range(P):
                    x = mw * sv[a]
                    for b in range(a, P):
                        M[a, b] += x * sv[b]
        if hetero:
            M_t[t] = M


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _ord_fe_cluster_pass(X, theta, colmap, cuts, eff, BD, keep, f, y, w, off, has_w, probit,
                         code, bounds, U):
    n = len(y)
    K = len(colmap)
    C = len(cuts)
    P = K + C - 1
    has_off = len(off) > 0
    n_parts = len(bounds) - 1
    for t in nb.prange(n_parts):
        lo = bounds[t]
        hi = bounds[t + 1]
        u = np.empty(K)
        sv = np.empty(P)
        for i in range(n):
            c = code[i]
            if c < lo or c >= hi:
                continue
            gi = f[i]
            if not keep[gi]:
                continue
            wi = w[i] if has_w else 1.0
            _ord_fe_score(X, i, colmap, theta, cuts, eff[gi], off[i] if has_off else 0.0,
                          BD[gi], y[i], probit, u, sv)
            for a in range(P):
                U[c, a] += wi * sv[a]


class _FEMeat:
    """The variances' pass for an absorbed fixed effect, block by block: `add`
    each block of grouped rows, then `result`. Like meat_kernels.py, the
    clusters' sums are kept per part when the copies fit, else each part owns
    a range of the cluster codes."""

    def __init__(self, fe, theta, A, H, weights_type, clusters, hetero):
        from . import meat_kernels as mk
        self.mk = mk
        m = fe.model
        self.fe, self.m, self.hetero = fe, m, bool(hetero)
        self.th = np.ascontiguousarray(theta[:m.P], np.float64)
        self.keep = np.ascontiguousarray(fe.keep_g)
        self.P = m.P
        self.wmode = {"aweights": 1, "fweights": 2}[weights_type]
        self.names = list(clusters)
        self.cols = [clusters[nm][0] for nm in self.names]
        self.sizes = [clusters[nm][1] for nm in self.names]
        self.cloff = np.r_[0, np.cumsum(self.sizes)].astype(np.int64)
        self.sumG = int(self.cloff[-1])
        self.U = np.zeros((self.sumG, self.P))
        self.Mt = np.zeros((self.P, self.P)) if hetero else None

    def combined(self, n):
        per = self.sumG * self.P * 8
        return bool(self.sumG) and per * nb.get_num_threads() <= self.mk.MEAT_CLUSTER_MB * 1e6 \
            and self.sumG * 8 <= n

    def n_parts(self, n, combined):
        parts = _parts(max(1, n // 256), int(np.sqrt(max(self.P * self.P, 1))))
        if combined:
            parts = min(parts, max(1, int(self.mk.MEAT_CLUSTER_MB * 1e6
                                          // (self.sumG * self.P * 8))))
        return parts

    def owner_bounds(self, k):
        return np.linspace(0, self.sizes[k], 4 * nb.get_num_threads() + 1).astype(np.int64)

    def result(self):
        U = {nm: self.U[self.cloff[k]:self.cloff[k + 1]] for k, nm in enumerate(self.names)}
        M = None
        if self.hetero:
            M = np.triu(self.Mt) + np.triu(self.Mt, 1).T
        return M, U

    def _run(self, blk, call_parts, call_owner):
        n = len(blk["y"])
        f = np.ascontiguousarray(blk["fe"], np.int64)
        combined = self.combined(n)
        n_parts = self.n_parts(n, combined)
        starts = part_starts(f, n_parts)
        cl = (np.ascontiguousarray(blk["cl"][:, self.cols], np.int64) if self.names
              else np.zeros((n, 0), np.int64))
        P = self.P
        M_t = np.zeros((n_parts, P, P) if self.hetero else (n_parts, 0, 0))
        U_t = np.zeros((n_parts, self.sumG if combined else 0, P))
        with nb.parallel_chunksize(1):
            if self.hetero or combined:
                call_parts(f, starts, cl if combined else cl[:, :0], M_t, U_t)
            if not combined:
                for k in range(len(self.names)):
                    call_owner(f, np.ascontiguousarray(cl[:, k]), self.owner_bounds(k),
                               self.U[self.cloff[k]:self.cloff[k + 1]])
        if self.hetero:
            self.Mt += M_t.sum(axis=0)
        if combined:
            self.U += U_t.sum(axis=0)
        return True


class MultinomialFEMeat(_FEMeat):
    def __init__(self, fe, theta, A, H, weights_type, clusters, hetero):
        super().__init__(fe, theta, A, H, weights_type, clusters, hetero)
        self.A = np.ascontiguousarray(A, np.float64)
        self.free = np.ascontiguousarray(fe.free, np.float64)
        self.BD = np.ascontiguousarray(H.BD, np.float64)

    def add(self, blk):
        m = self.m
        w = blk.get("w")
        wv = np.empty(0) if w is None else np.ascontiguousarray(w, np.float64)
        X = np.ascontiguousarray(blk["X"])
        cols = np.asarray(m.cols, np.int64)
        wmode = self.wmode if w is not None else 0

        def parts(f, starts, cl, M_t, U_t):
            _mnl_fe_meat_pass(X, self.th, cols, self.A, self.free, self.BD, self.keep, f,
                              blk["y"], wv, wmode, cl, self.cloff, self.hetero, starts, M_t, U_t)

        def owner(f, code, bounds, U):
            _mnl_fe_cluster_pass(X, self.th, cols, self.A, self.free, self.BD, self.keep, f,
                                 blk["y"], wv, w is not None, code, bounds, U)
        return self._run(blk, parts, owner)


class OrderedFEMeat(_FEMeat):
    def __init__(self, fe, theta, A, H, weights_type, clusters, hetero):
        super().__init__(fe, theta, A, H, weights_type, clusters, hetero)
        m = fe.model
        self.cuts = np.ascontiguousarray(m.cuts(self.th))
        self.eff = np.ascontiguousarray(A[:, 0], np.float64)
        self.BD = np.ascontiguousarray(H.BD[:, 0, :], np.float64)

    def add(self, blk):
        m = self.m
        w, off = blk.get("w"), blk.get("off")
        empty = np.empty(0)
        wv = empty if w is None else np.ascontiguousarray(w, np.float64)
        ov = empty if off is None else np.ascontiguousarray(off, np.float64)
        X = np.ascontiguousarray(blk["X"])
        cols = np.asarray(m.cols, np.int64)
        wmode = self.wmode if w is not None else 0
        probit = m.link.name == "probit"

        def parts(f, starts, cl, M_t, U_t):
            _ord_fe_meat_pass(X, self.th, cols, self.cuts, self.eff, self.BD, self.keep, f,
                              blk["y"], wv, ov, wmode, probit, cl, self.cloff, self.hetero,
                              starts, M_t, U_t)

        def owner(f, code, bounds, U):
            _ord_fe_cluster_pass(X, self.th, cols, self.cuts, self.eff, self.BD, self.keep, f,
                                 blk["y"], wv, ov, w is not None, probit, code, bounds, U)
        return self._run(blk, parts, owner)
