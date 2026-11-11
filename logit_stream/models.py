"""The three likelihoods: for a chunk of rows, the log-likelihood, its
gradient, minus its Hessian (the information, positive definite), and the
scores the robust and clustered variances need.

Every model here has a globally concave log-likelihood, which is what makes
a plain Newton iteration with step-halving the right optimizer for them, and
what lets the iteration be one read of the rows per step: a step needs only
sums over rows (log-likelihood, gradient, information), and each is
accumulated block by block.

The work per chunk is dense linear algebra -- one product X @ B for the
linear predictors, elementwise functions of them, and weighted Gram matrices
X' diag(v) X, each computed as (sqrt(v) X)' (sqrt(v) X) so that BLAS's
symmetric rank-k update does it -- so it ports to a GPU array library
unchanged but for the segment sums of the conditional logit (``reduceat``).

Parameter layouts (the order of `names`):

multinomial
    equation-major: the K covariates of the first non-base level, then of the
    second, ...; named ``"<level>:<covariate>"``.
ordered
    the K covariates, then the J - 1 cutpoints ``cut1`` ... ``cut{J-1}``,
    with P(y <= j) = F(cut_j - x'b).
conditional
    the K covariates.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np
from scipy.special import expit, ndtr, ndtri

# The multinomial information from the rows' nonzeros (`_mnl_info_sparse`)
# rather than dense Gram products, when the design is at most this dense on
# average, and the parameters few enough for a per-thread accumulator each
SPARSE_DENSITY = 0.5
SPARSE_MAX_PARAMS = 1200
# The conditional and fixed-effects logits' numba passes (`_clogit_pass`,
# `_cfe_units`) split a block's choice sets or units into this many parts per
# thread, handed out as threads come free (`_parts`), and keep the parts'
# information accumulators together to at most PARTS_MB (but at least one
# part per thread). The conditional logit's is used up to CLOGIT_KERNEL_MAX_P
# parameters: it adds each
# row into a P x P accumulator per thread, which is fast while that fits in a
# core's L2 cache (0.5 MB at 256; current cores have 1-2.5 MB) and several
# times slower once it does not, where BLAS's blocked products are not
# (measured, 4 threads: 0.17 of the NumPy pass's time at P = 128, 0.43 at
# 256, 0.7-0.85 at 384-512, 2.3 at 768 with a 2.5 MB L2). Above it the pass
# is NumPy's (`_Conditional._accumulate_blas`).
PARTS_PER_THREAD = 4
PARTS_MB = 64.0
# The multinomial logit's fused numba passes (`_mnl_pass`, `_mnl_pass_tri`),
# used for a block of at least MNL_KERNEL_MIN_ROWS rows (fewer: the per-call
# cost of the parallel loop and its accumulators is not worth it). The
# information is formed either from each row's nonzeros, when at most
# MNL_NNZ_DENSITY of the design's entries are nonzero (up to SPARSE_MAX_PARAMS
# parameters), or from its outer product, whose cost does not depend on the
# data: the nonzero form costs about (density x columns)^2 per pair of
# equations, the outer product columns^2 / 2 but several times faster per
# entry, which puts the crossover at a density of about 0.2 for any number of
# columns (measured, 6 outcome levels, 11-96 columns). The outer product's
# accumulators leave the core's cache as the columns grow, but it was
# still faster than BLAS's products at 96 columns (480 parameters; at 64:
# 0.5 s against 1.3-1.8 s per million rows), so it is used up to
# MNL_KERNEL_MAX_K columns.
MNL_KERNEL_MIN_ROWS = 4096
MNL_NNZ_DENSITY = 0.2
MNL_KERNEL_MAX_K = 96
# The ordered logit's and probit's fused pass (`_ord_pass`), from the same
# number of rows. Its information is one weighted Gram matrix of the
# covariates, from each row's outer product (up to ORD_KERNEL_MAX_K
# covariates: beyond, BLAS's products are faster; measured 1M rows, 5 levels:
# 0.30 s against 0.44 at 100, 0.66 against 0.52 at 128) or, with at most
# ORD_NNZ_DENSITY of the entries nonzero (about the crossover at 40-110
# covariates), from its nonzeros (up to ORD_NNZ_MAX_K, where the accumulator
# is 2 MB).
ORD_KERNEL_MAX_K = 96
ORD_NNZ_DENSITY = 0.3
ORD_NNZ_MAX_K = 512
# the fixed-effects logit's parts take runs of this many consecutive units
CFE_RUN = 64
CLOGIT_KERNEL_MAX_P = 256


def _parts(n_items: int, P: int) -> int:
    """How many parts to split `n_items` sets or units into, for a pass
    whose parts each keep a P x P accumulator (see PARTS_PER_THREAD)."""
    threads = nb.get_num_threads()
    cap = max(threads, int(PARTS_MB * 1e6 // (8 * max(P * P, 1))))
    return max(1, min(n_items, PARTS_PER_THREAD * threads, cap))


def _gram(X, v):
    """X' diag(v) X for v >= 0."""
    Xs = X * np.sqrt(v)[:, None]
    return Xs.T @ Xs


@nb.njit(parallel=True, cache=True)
def _mnl_info_sparse(X, P, w, n_threads, out):
    """The multinomial logit's information, from each row's nonzeros:
    out[(j, a), (l, b)] = sum_i w_i p_ij ([j = l] - p_il) x_ia x_ib for the
    upper blocks (j <= l), the lower ones left for the caller to mirror.

    A design of indicators has a few nonzeros per row, so this does
    (J - 1) J / 2 x nnz^2 operations per row where the dense Gram products
    do (J - 1) J / 2 x K^2. Rows are split among `n_threads` accumulators,
    one P x P matrix each."""
    n, K = X.shape
    J1 = P.shape[1]
    Pn = J1 * K
    acc = np.zeros((n_threads, Pn, Pn))
    step = (n + n_threads - 1) // n_threads
    for t in nb.prange(n_threads):
        idx = np.empty(K, np.int64)
        val = np.empty(K)
        a_ = acc[t]
        for i in range(t * step, min(n, (t + 1) * step)):
            cnt = 0
            for k in range(K):
                x = X[i, k]
                if x != 0.0:
                    idx[cnt] = k
                    val[cnt] = x
                    cnt += 1
            wi = w[i]
            for j in range(J1):
                pj = P[i, j] * wi
                if pj == 0.0:
                    continue
                for l in range(j, J1):
                    v = pj * ((1.0 if j == l else 0.0) - P[i, l])
                    if v == 0.0:
                        continue
                    jo, lo = j * K, l * K
                    for aa in range(cnt):
                        ra = jo + idx[aa]
                        xa = v * val[aa]
                        for bb in range(cnt):
                            a_[ra, lo + idx[bb]] += xa * val[bb]
    for t in range(n_threads):
        out += acc[t]


@nb.njit(parallel=True, cache=True)
def _mnl_info_sparse_own(X, P, w, own, n_threads, out):
    """`_mnl_info_sparse` for a design with own covariates: `own[k]` is the
    equation (0-based, among the non-base levels) that column k enters, or
    -1 for a column in every equation. A pair of columns contributes to
    block (j, l) only if each is in its equation, so the off-diagonal
    entries of the own columns are left zero (the caller takes the free
    parameters' rows and columns)."""
    n, K = X.shape
    J1 = P.shape[1]
    Pn = J1 * K
    acc = np.zeros((n_threads, Pn, Pn))
    step = (n + n_threads - 1) // n_threads
    for t in nb.prange(n_threads):
        idx = np.empty(K, np.int64)
        val = np.empty(K)
        a_ = acc[t]
        for i in range(t * step, min(n, (t + 1) * step)):
            cnt = 0
            for k in range(K):
                x = X[i, k]
                if x != 0.0:
                    idx[cnt] = k
                    val[cnt] = x
                    cnt += 1
            wi = w[i]
            for j in range(J1):
                pj = P[i, j] * wi
                if pj == 0.0:
                    continue
                for l in range(j, J1):
                    v = pj * ((1.0 if j == l else 0.0) - P[i, l])
                    if v == 0.0:
                        continue
                    jo, lo = j * K, l * K
                    for aa in range(cnt):
                        oa = own[idx[aa]]
                        if oa >= 0 and oa != j:
                            continue
                        ra = jo + idx[aa]
                        xa = v * val[aa]
                        for bb in range(cnt):
                            ob = own[idx[bb]]
                            if ob >= 0 and ob != l:
                                continue
                            a_[ra, lo + idx[bb]] += xa * val[bb]
    for t in range(n_threads):
        out += acc[t]


@nb.njit(parallel=True, cache=True)
def _mnl_pass(X, theta, cols, ncols, off, y, w, hessian, ll_t, g_t, H_t):
    """The multinomial logit's log-likelihood, gradient and, with `hessian`,
    information over the rows of a block, in one pass, for a design of mostly
    zeros: the rows are split into ``len(ll_t)`` parts, and part t's sums are
    written to ``ll_t[t]``, ``g_t[t]`` and ``H_t[t]``, for the caller to add
    up. Per row: the linear predictors, the softmax, the log-likelihood, the
    gradient w (1[y = j] - p_j) x, and the information
    sum w p_j (1[j = l] - p_l) x x' for the blocks j <= l, over the row's
    nonzero entries (a design of indicators has a few per equation); only the
    upper triangle is filled. Without `hessian` this is also the pass for a
    dense design (`_mnl_pass_tri` forms its information).

    Equation j (the j-th non-base level) has the columns ``cols[j, :ncols[j]]``
    of `X`, its parameters ``off[j] + a`` in `theta`: every column in every
    equation, or fewer with own covariates.

    Like `_clogit_pass`, there is no row-sized temporary and the only parallel
    loop is over the parts, whose accumulators the caller allocates."""
    n = len(y)
    J1 = len(ncols)
    P = off[J1]
    n_parts = len(ll_t)
    has_w = len(w) > 0
    step = (n + n_parts - 1) // n_parts
    for t in nb.prange(n_parts):
        xg = np.empty(P)                    # the row's entries, in parameter order
        nzi = np.empty(P, np.int64)         # parameter indices of its nonzeros, by equation
        cnt = np.zeros(J1, np.int64)
        et = np.empty(J1)
        p = np.empty(J1)
        g = np.zeros(P)
        Ht = np.zeros((P, P) if hessian else (0, 0))
        llt = 0.0
        for i in range(t * step, min(n, (t + 1) * step)):
            wi = w[i] if has_w else 1.0
            m = 0.0
            for j in range(J1):
                e = 0.0
                o = off[j]
                for a in range(ncols[j]):
                    x = X[i, cols[j, a]]
                    xg[o + a] = x
                    e += x * theta[o + a]
                et[j] = e
                if e > m:
                    m = e
            s = np.exp(-m)
            for j in range(J1):
                p[j] = np.exp(et[j] - m)
                s += p[j]
            yi = y[i]
            lp = -m - np.log(s)
            if yi > 0:
                lp += et[yi - 1]
            llt += wi * lp
            inv = 1.0 / s
            for j in range(J1):
                p[j] *= inv
            for j in range(J1):
                r = wi * ((1.0 if yi == j + 1 else 0.0) - p[j])
                o = off[j]
                if hessian:
                    c = 0
                    for a in range(ncols[j]):
                        if xg[o + a] != 0.0:
                            g[o + a] += r * xg[o + a]
                            nzi[o + c] = o + a
                            c += 1
                    cnt[j] = c
                else:
                    for a in range(ncols[j]):
                        g[o + a] += r * xg[o + a]
            if hessian:
                for j in range(J1):
                    pj = wi * p[j]
                    for l in range(j, J1):
                        v = pj * ((1.0 if j == l else 0.0) - p[l])
                        for aa in range(cnt[j]):
                            ra = nzi[off[j] + aa]
                            va = v * xg[ra]
                            for bb in range(aa if j == l else 0, cnt[l]):
                                cb = nzi[off[l] + bb]
                                Ht[ra, cb] += va * xg[cb]
        ll_t[t] = llt
        for k in range(P):
            g_t[t, k] = g[k]
        if hessian:
            for k in range(P):
                for l in range(P):
                    H_t[t, k, l] = Ht[k, l]


@nb.njit(parallel=True, cache=True)
def _mnl_pass_tri(X, theta, colmap, pos, ncols, off, y, w, ll_t, g_t, A_t):
    """`_mnl_pass` with the information for a dense design. Per row, the
    outer product x x' of the design's K columns (colmap: their columns of
    `X`) is formed once, as a vector of its K (K + 1) / 2 upper-triangle
    entries, and added, times the weight w p_j (1[j = l] - p_l), to the
    accumulator of each pair of equations j <= l: an add of contiguous
    vectors, which the compiler vectorizes (a scatter into the P x P matrix
    does not). `_assemble_tri` forms the information from the accumulators
    `A_t[t]`, keeping each equation's own columns, `pos[j, :ncols[j]]` (their
    positions among the K) at ``off[j] + a`` in `theta`; for own covariates
    it also accumulates entries the model does not have, at the cost of
    some wasted adds."""
    n = len(y)
    J1 = len(ncols)
    K = len(colmap)
    P = off[J1]
    ntri = K * (K + 1) // 2
    n_parts = len(ll_t)
    has_w = len(w) > 0
    step = (n + n_parts - 1) // n_parts
    for t in nb.prange(n_parts):
        u = np.empty(K)
        xxt = np.empty(ntri)
        et = np.empty(J1)
        p = np.empty(J1)
        g = np.zeros(P)
        llt = 0.0
        acc = np.zeros((J1 * (J1 + 1) // 2, ntri))
        for i in range(t * step, min(n, (t + 1) * step)):
            wi = w[i] if has_w else 1.0
            for c in range(K):
                u[c] = X[i, colmap[c]]
            m = 0.0
            for j in range(J1):
                o = off[j]
                e = 0.0
                for a in range(ncols[j]):
                    e += u[pos[j, a]] * theta[o + a]
                et[j] = e
                if e > m:
                    m = e
            s = np.exp(-m)
            for j in range(J1):
                p[j] = np.exp(et[j] - m)
                s += p[j]
            yi = y[i]
            lp = -m - np.log(s)
            if yi > 0:
                lp += et[yi - 1]
            llt += wi * lp
            inv = 1.0 / s
            for j in range(J1):
                p[j] *= inv
            k = 0
            for c1 in range(K):
                x1 = u[c1]
                for c2 in range(c1, K):
                    xxt[k] = x1 * u[c2]
                    k += 1
            pr = 0
            for j in range(J1):
                r = wi * ((1.0 if yi == j + 1 else 0.0) - p[j])
                o = off[j]
                for a in range(ncols[j]):
                    g[o + a] += r * u[pos[j, a]]
                pj = wi * p[j]
                for l in range(j, J1):
                    v = pj * ((1.0 if j == l else 0.0) - p[l])
                    for k in range(ntri):
                        acc[pr, k] += v * xxt[k]
                    pr += 1
        ll_t[t] = llt
        for k in range(P):
            g_t[t, k] = g[k]
        A_t[t] = acc


def _assemble_tri(A, pos, ncols, off, K):
    """The information from `_mnl_pass_tri`'s accumulators (summed over the
    parts): block (j, l) is the pair's K x K matrix at the rows of equation
    j's columns and the columns of equation l's."""
    J1 = len(ncols)
    tri = np.triu_indices(K)
    H = np.zeros((off[-1], off[-1]))
    pr = 0
    for j in range(J1):
        pj = pos[j, :ncols[j]]
        for l in range(j, J1):
            S = np.zeros((K, K))
            S[tri] = A[pr]
            S = S + np.triu(S, 1).T
            pr += 1
            B = S[np.ix_(pj, pos[l, :ncols[l]])]
            H[off[j]:off[j + 1], off[l]:off[l + 1]] = B
            if l != j:
                H[off[l]:off[l + 1], off[j]:off[j + 1]] = B.T
    return H


@nb.njit(cache=True, inline="always", error_model="numpy")
def _ord_cdf(x, probit):
    """The error distribution's cdf: the logistic (as `scipy.special.expit`,
    stable in both tails) or the standard normal."""
    if probit:
        return 0.5 * math.erfc(-x * 0.7071067811865476)
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@nb.njit(cache=True, inline="always", error_model="numpy")
def _ord_terms(a, b, probit, hessian):
    """A row of the ordered model, from the limits a and b of its level (the
    cutpoints less the linear predictor): P = F(a) - F(b) (as S(b) - S(a) in
    the upper tail), g_a = f(a) / P and g_b = -f(b) / P, and with `hessian`
    the second derivatives h_aa, h_bb and h_ab (zero without). The pieces of
    `_Ordered.rows`; returns (P, g_a, g_b, h_aa, h_bb, h_ab)."""
    Fa = _ord_cdf(a, probit)
    Sa = _ord_cdf(-a, probit)
    Fb = _ord_cdf(b, probit)
    Sb = _ord_cdf(-b, probit)
    Pr = Sb - Sa if b > 0 else Fa - Fb
    ac = min(max(a, -40.0), 40.0)
    bc = min(max(b, -40.0), 40.0)
    if probit:
        fa = np.exp(-0.5 * ac * ac) / 2.5066282746310002
        fb = np.exp(-0.5 * bc * bc) / 2.5066282746310002
    else:
        fa = Fa * Sa
        fb = Fb * Sb
    ga = fa / Pr
    gb = -fb / Pr
    haa = 0.0
    hbb = 0.0
    if hessian:
        if probit:
            fpa = -ac * fa
            fpb = -bc * fb
        else:
            fpa = fa * (1.0 - 2.0 * Fa)
            fpb = fb * (1.0 - 2.0 * Fb)
        haa = fpa / Pr - ga * ga
        hbb = -fpb / Pr - gb * gb
    return Pr, ga, gb, haa, hbb, -ga * gb


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _ord_pass(X, theta, colmap, cuts, y, w, off, probit, hessian, use_nnz, ll_t, g_t, A_t,
              XC_t, CC_t):
    """The ordered logit's (probit's, with `probit`) log-likelihood, gradient
    and information over the rows of a block, in one pass, as `_mnl_pass_tri`
    does for the multinomial logit: the rows are split into ``len(ll_t)``
    parts whose sums the caller adds up. The row's level y lies between the
    cutpoints ``cuts[y - 1]`` and ``cuts[y]`` (open at the ends), a = cut_y -
    x'b and b = cut_{y-1} - x'b are its limits, and P = F(a) - F(b), taken as
    S(b) - S(a) in the upper tail. The pieces are those of `_Ordered.rows`:
    with g_a = f(a) / P, g_b = -f(b) / P, the score in eta is -(g_a + g_b),
    and the information is h x x' for the covariates (A_t: the upper triangle
    by rows, K (K + 1) / 2 entries, or with `use_nnz` the entries ``a K + b``
    of a K x K matrix, formed from the row's nonzeros), q_c x for the
    covariates against cutpoint c (XC_t), and a tridiagonal block for the
    cutpoints (CC_t, the superdiagonal in the upper triangle)."""
    n = len(y)
    K = len(colmap)
    C = len(cuts)
    P = K + C
    nacc = A_t.shape[1]
    n_parts = len(ll_t)
    has_w = len(w) > 0
    has_off = len(off) > 0
    step = (n + n_parts - 1) // n_parts
    for t in nb.prange(n_parts):
        u = np.empty(K)
        nz = np.empty(K, np.int64)
        g = np.zeros(P)
        acc = np.zeros(nacc)
        xc = np.zeros((C, K))
        cc = np.zeros((C, C))
        llt = 0.0
        for i in range(t * step, min(n, (t + 1) * step)):
            wt = w[i] if has_w else 1.0
            yi = y[i]
            eta = off[i] if has_off else 0.0
            for k in range(K):
                u[k] = X[i, colmap[k]]
                eta += u[k] * theta[k]
            a = cuts[yi] - eta if yi < C else np.inf
            b = cuts[yi - 1] - eta if yi > 0 else -np.inf
            Pr, ga, gb, haa, hbb, hab = _ord_terms(a, b, probit, hessian)
            llt += wt * np.log(Pr)
            s = -(ga + gb) * wt
            for k in range(K):
                g[k] += s * u[k]
            if yi < C:
                g[K + yi] += ga * wt
            if yi > 0:
                g[K + yi - 1] += gb * wt
            if hessian:
                h = wt * max(-(haa + 2.0 * hab + hbb), 0.0)
                cnt = 0
                if use_nnz:
                    for k in range(K):
                        if u[k] != 0.0:
                            nz[cnt] = k
                            cnt += 1
                    for aa in range(cnt):
                        ra = nz[aa]
                        hx = h * u[ra]
                        for bb in range(aa, cnt):
                            rb = nz[bb]
                            acc[ra * K + rb] += hx * u[rb]
                else:
                    k = 0
                    for c1 in range(K):
                        hx = h * u[c1]
                        for c2 in range(c1, K):
                            acc[k] += hx * u[c2]
                            k += 1
                if yi < C:
                    q = wt * (haa + hab)
                    if use_nnz:
                        for aa in range(cnt):
                            xc[yi, nz[aa]] += q * u[nz[aa]]
                    else:
                        for k in range(K):
                            xc[yi, k] += q * u[k]
                    cc[yi, yi] -= wt * haa
                if yi > 0:
                    q = wt * (hab + hbb)
                    if use_nnz:
                        for aa in range(cnt):
                            xc[yi - 1, nz[aa]] += q * u[nz[aa]]
                    else:
                        for k in range(K):
                            xc[yi - 1, k] += q * u[k]
                    cc[yi - 1, yi - 1] -= wt * hbb
                    if yi < C:
                        cc[yi - 1, yi] -= wt * hab
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


@nb.njit(parallel=True, cache=True)
def _clogit_pass(X, cols, theta, y, w, off, starts, hessian, max_size, ll_t, g_t, H_t):
    """The conditional logit's log-likelihood, gradient and information over
    the choice sets of a block, in one pass over its rows: the sets are
    split into ``len(ll_t)`` parts, and part t's sums are written to
    ``ll_t[t]``, ``g_t[t]`` and (with `hessian`) ``H_t[t]``, for the caller
    to add up (`_Conditional.accumulate_full`). `cols` are the columns of
    `X` in the model; `w` and `off` are empty when there are no weights or
    offset.

    The only parallel loop is the one over the parts: the accumulators are
    allocated and summed by the caller, since with `parallel_chunksize`
    numba would hand out every element of an array operation here as a
    chunk of its own.

    Per set: p = softmax(eta), the log-likelihood eta_chosen - log sum
    exp(eta), the gradient sum (y - p) x, and the information
    sum p (x - xbar)(x - xbar)' with xbar = sum p x (the within-set
    covariance, centred before the products: the same matrix as
    sum p x x' - xbar xbar', without the cancellation). Each is weighted by
    the set's weight. Only the upper triangle of the information is filled."""
    n_sets = len(starts) - 1
    P = len(cols)
    n_parts = len(ll_t)
    has_w, has_off = len(w) > 0, len(off) > 0
    step = (n_sets + n_parts - 1) // n_parts
    for t in nb.prange(n_parts):
        # local accumulators, copied out at the end: rows of the shared
        # arrays would share cache lines between threads
        eta = np.empty(max_size)
        xs = np.empty((max_size, P))         # the set's rows, read once
        xbar = np.empty(P)
        d = np.empty(P)
        g = np.zeros(P)
        Ht = np.zeros((P, P) if hessian else (0, 0))
        llt = 0.0
        for s in range(t * step, min(n_sets, (t + 1) * step)):
            a, b = starts[s], starts[s + 1]
            ws = w[a] if has_w else 1.0
            m = -np.inf
            chosen = 0.0
            for i in range(a, b):
                e = off[i] if has_off else 0.0
                for k in range(P):
                    x = X[i, cols[k]]
                    xs[i - a, k] = x
                    e += x * theta[k]
                eta[i - a] = e
                if y[i] == 1:
                    chosen += e
                if e > m:
                    m = e
            S = 0.0
            for i in range(a, b):
                eta[i - a] = np.exp(eta[i - a] - m)
                S += eta[i - a]
            llt += ws * (chosen - m - np.log(S))
            for k in range(P):
                xbar[k] = 0.0
            for i in range(a, b):
                p = eta[i - a] / S
                eta[i - a] = p
                r = ws * ((1.0 if y[i] == 1 else 0.0) - p)
                for k in range(P):
                    x = xs[i - a, k]
                    g[k] += r * x
                    xbar[k] += p * x
            if hessian:
                for i in range(a, b):
                    p = ws * eta[i - a]
                    for k in range(P):
                        d[k] = xs[i - a, k] - xbar[k]
                    # the full square (a fixed-length inner loop, which
                    # vectorizes), though only the upper triangle is used
                    for k in range(P):
                        v = p * d[k]
                        for l in range(P):
                            Ht[k, l] += v * d[l]
        ll_t[t] = llt
        for k in range(P):
            g_t[t, k] = g[k]
        if hessian:
            for k in range(P):
                for l in range(P):
                    H_t[t, k, l] = Ht[k, l]


def _bincount(idx, weights, n):
    return np.bincount(idx, weights=weights, minlength=n)[:n]


class _Model:
    """What the Newton driver needs from a model. `cols` are the design's
    columns still in the model (collinear ones are dropped after the first
    pass)."""

    kind: str
    #: the store's rows (rows) or choice sets (cases) are the independent units
    unit = "rows"

    def __init__(self, xnames, K):
        self.xnames = list(xnames)
        self.cols = np.arange(K)

    # parameter bookkeeping ---------------------------------------------------
    @property
    def K(self):
        return len(self.cols)

    def names(self):
        raise NotImplementedError

    def penalized(self):
        """Which parameters an L2 penalty applies to: not intercepts or
        cutpoints, as in scikit-learn (whose intercept is unpenalized)."""
        raise NotImplementedError

    def feasible(self, theta):
        return bool(np.all(np.isfinite(theta)))

    def x_block(self, H):
        """The part of the first step's information that is a weighted Gram
        matrix of the design's current columns, whose rank deficiency is the
        design's collinearity; and the indices of the other parameters that
        must be kept ahead of the covariates (cutpoints)."""
        raise NotImplementedError

    def drop(self, keep):
        """Keep only the design columns `keep` (booleans over `cols`);
        returns the index of the parameters kept."""
        raise NotImplementedError

    def width(self, scores=False):
        """Columns of the widest per-row temporary, for chunk sizing."""
        return self.K

    def sparse(self, X):
        """Whether the information is computed by a numba kernel from the
        rows' nonzeros (the multinomial's choice)."""
        return False


# ------------------------------------------------------------------ multinomial
class _Multinomial(_Model):
    kind = "mlogit"

    def __init__(self, xnames, K, levels, intercept, own=None):
        super().__init__(xnames, K)
        self.levels = list(levels)
        self.J = len(levels)
        self.intercept = intercept          # index of the Intercept column, or None
        # own covariates: {design column: the equation (0-based, among the
        # non-base levels) it enters}. Every other column enters all of them.
        # The own columns come after the others in the design.
        self.own = dict(own or {})
        self._free_cache = None

    # own covariates ----------------------------------------------------------
    def free(self):
        """Which of the J - 1 x K coefficients (equation by current column)
        are estimated: all, but for an own covariate's other equations."""
        if self._free_cache is None or self._free_cache.shape[1] != self.K:
            M = np.ones((self.J - 1, self.K), bool)
            for p, c in enumerate(self.cols):
                e = self.own.get(int(c))
                if e is not None:
                    M[:, p] = False
                    M[e, p] = True
            self._free_cache = M
        return self._free_cache

    def param_cols(self):
        """The design column (an index into the original columns) of each
        parameter, in the order of `names`."""
        if not self.own:
            return np.tile(self.cols, self.J - 1)
        return np.broadcast_to(self.cols, (self.J - 1, self.K))[self.free()]

    def _unpack(self, theta):
        if not self.own:
            return theta.reshape(self.J - 1, self.K)
        B = np.zeros((self.J - 1, self.K))
        B[self.free()] = theta
        return B

    def keep_columns(self, H, collinear):
        """Which design columns to keep, from the first step's information:
        each equation's block (the columns it has) is checked for collinear
        columns by `collinear`, and a column is dropped if any equation
        finds it so."""
        M = self.free()
        keep = np.ones(self.K, bool)
        off = 0
        for e in range(self.J - 1):
            pos = np.flatnonzero(M[e])
            n = len(pos)
            keep[pos] &= collinear(H[off:off + n, off:off + n])
            off += n
        return keep

    @property
    def P(self):
        return int(self.free().sum()) if self.own else self.K * (self.J - 1)

    def names(self):
        if self.own:
            M = self.free()
            return [f"{lv}:{self.xnames[c]}" for e, lv in enumerate(self.levels[1:])
                    for p, c in enumerate(self.cols) if M[e, p]]
        return [f"{lv}:{self.xnames[c]}" for lv in self.levels[1:] for c in self.cols]

    def penalized(self):
        pen = np.ones((self.J - 1, self.K), bool)
        if self.intercept is not None and self.intercept in self.cols:
            pen[:, int(np.flatnonzero(self.cols == self.intercept)[0])] = False
        return pen[self.free()] if self.own else pen.ravel()

    def start(self, counts, fe=False):
        """Zero slopes, and the intercepts that give each level its sample
        share: the null model, so that the first log-likelihood is the null
        log-likelihood."""
        B = np.zeros((self.J - 1, self.K))
        if self.intercept is not None and not fe:
            B[:, self.intercept] = np.log(counts[1:] / counts[0])
        return B[self.free()] if self.own else B.ravel()

    def x_block(self, H):
        return H[:self.K, :self.K], []

    def drop(self, keep):
        if self.own:
            M = self.free()
            idx = np.flatnonzero((M & keep[None, :])[M])
            self.cols = self.cols[keep]
            if self.intercept is not None and self.intercept not in self.cols:
                self.intercept = None
            return idx
        idx = np.flatnonzero(np.tile(keep, self.J - 1))
        self.cols = self.cols[keep]
        if self.intercept is not None and self.intercept not in self.cols:
            self.intercept = None
        return idx

    def width(self, scores=False):
        return self.K * (self.J - 1) if scores else self.K + self.J

    def eta(self, theta, X):
        return X @ self._unpack(theta).T

    @staticmethod
    def probs(E):
        """Probabilities of levels 1..J-1 and of the base level, from the
        n x (J-1) linear predictors, stably; and log of the denominator."""
        m = np.maximum(E.max(axis=1), 0.0) if E.shape[1] else np.zeros(len(E))
        ex = np.exp(E - m[:, None])
        e0 = np.exp(-m)
        den = e0 + ex.sum(axis=1)
        return ex, e0, den, m + np.log(den)

    def nonzero_form(self):
        """Whether the fused pass forms the information from each row's
        nonzeros (a design that is mostly zeros), not its outer product.
        `density` is set from pass 0's moments."""
        d = getattr(self, "density", None)
        return (d is not None and d <= MNL_NNZ_DENSITY and self.P <= SPARSE_MAX_PARAMS
                and not getattr(self, "dense_only", False))

    def information_form(self, n):
        """How the information is formed for blocks of about `n` rows, for
        the fit's diagnostics: ``"outer-product"`` or ``"nonzero"`` (the
        fused numba pass, from each row's outer product or from its
        nonzeros), else ``"array"`` (NumPy and BLAS; ``"array-nonzero"``
        with the older numba kernel for a mostly-zero design)."""
        if self.kernel_ok(n):
            return "nonzero" if self.nonzero_form() else "outer-product"
        return "array-nonzero" if self.sparse(None) else "array"

    def kernel_ok(self, n):
        """Whether a block of `n` rows is summed by the fused numba pass:
        not a few rows, and not a dense design of very many columns (BLAS's
        blocked products win there)."""
        if n < MNL_KERNEL_MIN_ROWS or self.P == 0:
            return False
        return self.nonzero_form() or self.K <= MNL_KERNEL_MAX_K

    def accumulate_full(self, theta, X, cols, y, w, starts, hessian, offset=None):
        """`accumulate` by the fused numba pass, on the columns `cols` of
        `X` (a whole block, read in place): the information from each row's
        nonzeros (`_mnl_pass`) for a mostly-zero design, else from its
        outer product (`_mnl_pass_tri`)."""
        M = self.free()
        J1, K = self.J - 1, M.shape[1]
        cols = np.asarray(cols)
        ncols = M.sum(axis=1).astype(np.int64)
        pos = np.zeros((J1, K), np.int64)
        for j in range(J1):
            pos[j, :ncols[j]] = np.flatnonzero(M[j])
        off = np.r_[0, np.cumsum(ncols)].astype(np.int64)
        P = int(off[-1])
        X = np.ascontiguousarray(X)
        th = np.ascontiguousarray(theta, np.float64)
        wv = np.empty(0) if w is None else np.ascontiguousarray(w, np.float64)
        tri = hessian and not self.nonzero_form()
        ntri = K * (K + 1) // 2
        # the parts' accumulators are kept to PARTS_MB together: as P x P matrices, or
        # as the pairs' triangles
        n_parts = _parts(max(1, len(y) // 256),
                         int(np.sqrt(J1 * (J1 + 1) // 2 * ntri)) if tri else P)
        ll_t, g_t = np.zeros(n_parts), np.zeros((n_parts, P))
        with nb.parallel_chunksize(1):
            if tri:
                A_t = np.zeros((n_parts, J1 * (J1 + 1) // 2, ntri))
                _mnl_pass_tri(X, th, cols, pos, ncols, off, y, wv, ll_t, g_t, A_t)
                return float(ll_t.sum()), g_t.sum(axis=0), \
                    _assemble_tri(A_t.sum(axis=0), pos, ncols, off, K)
            H_t = np.zeros((n_parts, P, P) if hessian else (n_parts, 0, 0))
            _mnl_pass(X, th, np.where(np.arange(K) < ncols[:, None], cols[pos], 0), ncols, off,
                      y, wv, bool(hessian), ll_t, g_t, H_t)
        ll, g = float(ll_t.sum()), g_t.sum(axis=0)
        if not hessian:
            return ll, g, None
        H = H_t.sum(axis=0)
        return ll, g, np.triu(H) + np.triu(H, 1).T

    def accumulate(self, theta, X, y, w, starts, hessian, offset=None):
        if offset is None and self.kernel_ok(len(y)):
            return self.accumulate_full(theta, X, np.arange(X.shape[1]), y, w, starts, hessian)
        J1, K = self.J - 1, self.K
        E = self.eta(theta, X)
        if offset is not None:
            E += offset
        ex, e0, den, logden = self.probs(E)
        P = ex / den[:, None]
        idx = np.flatnonzero(y > 0)
        lp = -logden
        lp[idx] += E[idx, y[idx] - 1]
        ll = float(lp.sum() if w is None else lp @ w)
        R = -P
        R[idx, y[idx] - 1] += 1.0
        if w is not None:
            R *= w[:, None]
        g = (X.T @ R).T
        g = g[self.free()] if self.own else g.ravel()
        if not hessian:
            return ll, g, None
        if self.own:
            return ll, g, self._information_own(X, P, ex, den, w)
        if self.sparse(X):
            H = np.zeros((J1 * K, J1 * K))
            _mnl_info_sparse(np.ascontiguousarray(X), np.ascontiguousarray(P),
                             np.ones(len(y)) if w is None else w,
                             nb.get_num_threads(), H)
            for j in range(J1):
                for k in range(j + 1, J1):
                    H[k * K:(k + 1) * K, j * K:(j + 1) * K] = \
                        H[j * K:(j + 1) * K, k * K:(k + 1) * K].T
            return ll, g, H
        H = np.empty((J1 * K, J1 * K))
        Pw = P if w is None else P * w[:, None]
        for j in range(J1):
            sj = slice(j * K, (j + 1) * K)
            # p_j (1 - p_j), with 1 - p_j from the other terms of the
            # denominator rather than by subtraction
            H[sj, sj] = _gram(X, Pw[:, j] * ((den - ex[:, j]) / den))
            for k in range(j + 1, J1):
                sk = slice(k * K, (k + 1) * K)
                Hjk = -_gram(X, Pw[:, j] * P[:, k])
                H[sj, sk] = Hjk
                H[sk, sj] = Hjk
        return ll, g, H

    def _information_own(self, X, P, ex, den, w):
        """The information with own covariates: the free parameters' rows and
        columns only. The common columns are a prefix of the design (the own
        ones follow), so block (j, l) is the Gram product of the common
        columns, plus the thin products with the equations' own columns."""
        J1, K = self.J - 1, self.K
        M = self.free()
        if self.sparse(X):
            own = np.full(K, -1, np.int64)
            for p, c in enumerate(self.cols):
                own[p] = self.own.get(int(c), -1)
            H = np.zeros((J1 * K, J1 * K))
            _mnl_info_sparse_own(np.ascontiguousarray(X), np.ascontiguousarray(P),
                                 np.ones(len(X)) if w is None else w, own,
                                 nb.get_num_threads(), H)
            for j in range(J1):
                for k in range(j + 1, J1):
                    H[k * K:(k + 1) * K, j * K:(j + 1) * K] = \
                        H[j * K:(j + 1) * K, k * K:(k + 1) * K].T
            fi = np.flatnonzero(M.ravel())
            return H[np.ix_(fi, fi)]
        nc = int(sum(int(c) not in self.own for c in self.cols))
        Xc = np.ascontiguousarray(X[:, :nc])
        Xo = np.ascontiguousarray(X[:, nc:])
        opos = [np.flatnonzero(M[e, nc:]) for e in range(J1)]       # into Xo's columns
        size = [nc + len(o) for o in opos]
        off = np.r_[0, np.cumsum(size)]
        H = np.zeros((off[-1], off[-1]))
        Pw = P if w is None else P * w[:, None]
        one_minus = (den[:, None] - ex) / den[:, None]              # 1 - p_j, without cancellation
        # the common columns' blocks: X_c' diag(v_jk) X_c, v_jk = w p_j (1[j = k] - p_k)
        if nc:
            for j in range(J1):
                for k in range(j, J1):
                    v = Pw[:, j] * one_minus[:, j] if j == k else Pw[:, j] * P[:, k]
                    G = _gram(Xc, v) * (1.0 if j == k else -1.0)
                    H[off[j]:off[j] + nc, off[k]:off[k] + nc] = G
                    if k != j:
                        H[off[k]:off[k] + nc, off[j]:off[j] + nc] = G.T
        # own column b (of equation k) against the common columns and the other own
        # columns, for every equation j at once: the weights v_jk of all j form one
        # n x (J-1) matrix, so each own column costs two thin products
        eq_of = M[:, nc:].argmax(axis=0) if Xo.shape[1] else np.zeros(0, int)
        pos_of = np.empty(Xo.shape[1], np.int64)           # an own column's parameter index
        for e in range(J1):
            pos_of[opos[e]] = off[e] + nc + np.arange(len(opos[e]))
        for k in range(J1):
            if not len(opos[k]):
                continue
            Vk = -Pw * P[:, [k]]
            Vk[:, k] = Pw[:, k] * one_minus[:, k]
            for b in opos[k]:
                Vb = Vk * Xo[:, [b]]
                pb = pos_of[b]
                if nc:
                    T = Xc.T @ Vb                           # nc x (J-1): column j is block (j, k)
                    for j in range(J1):
                        H[off[j]:off[j] + nc, pb] = T[:, j]
                        H[pb, off[j]:off[j] + nc] = T[:, j]
                R = Vb.T @ Xo                               # (J-1) x Mo, own against own
                for a in range(Xo.shape[1]):
                    H[pos_of[a], pb] = R[eq_of[a], a]
        return H

    def firth_adjustment(self, theta, X, w, A):
        """A chunk's part of Firth's adjustment to the score, the gradient of
        log|I| / 2, given A = I^-1: with Q_i the J-1 x J-1 block of
        (I (x) x_i)' A (I (x) x_i) and W_i = diag p_i - p_i p_i', the
        derivative of log|I| / 2 in eta_i is c_i / 2 with
        c_i = W_i (diag Q_i - 2 Q_i p_i) (the derivative of W_i in eta_il is
        linear in W_i's columns), and the adjustment sums w_i c_i (x) x_i / 2.
        The same form as the exact Laplace term of random_effects/model.py."""
        J1, K = self.J - 1, self.K
        ex, _, den, _ = self.probs(self.eta(theta, X))
        P = ex / den[:, None]
        Q = np.empty((len(X), J1, J1))
        for j in range(J1):
            for l in range(j, J1):
                q = np.einsum("ik,ik->i", X @ A[j * K:(j + 1) * K, l * K:(l + 1) * K], X)
                Q[:, j, l] = q
                Q[:, l, j] = q
        v = np.einsum("ijj->ij", Q) - 2 * np.einsum("ijl,il->ij", Q, P)
        C = P * v - P * (P * v).sum(axis=1, keepdims=True)
        if w is not None:
            C *= w[:, None]
        return 0.5 * (X.T @ C).T.ravel()

    def sparse(self, X):
        """Whether to compute the information from the rows' nonzeros: a
        design that is mostly zeros (indicators), and few enough parameters
        for one accumulator per thread. `density` is set from pass 0's
        moments; a chunk is not inspected."""
        d = getattr(self, "density", None)
        return (d is not None and d <= SPARSE_DENSITY and self.P <= SPARSE_MAX_PARAMS
                and not getattr(self, "dense_only", False))

    def scores(self, theta, X, y, starts, offset=None):
        """Each row's score, n x P."""
        E = self.eta(theta, X)
        if offset is not None:
            E += offset
        ex, _, den, _ = self.probs(E)
        R = -ex / den[:, None]
        idx = np.flatnonzero(y > 0)
        R[idx, y[idx] - 1] += 1.0
        if self.own:
            M = self.free()
            return np.hstack([R[:, [e]] * X[:, M[e]] for e in range(self.J - 1)])
        return (R[:, :, None] * X[:, None, :]).reshape(len(y), -1)


# ------------------------------------------------------------------ ordered
class _LogisticLink:
    """The ordered logit's error distribution: the logistic."""
    name = "logit"

    @staticmethod
    def cdf(x):
        return expit(x)

    @staticmethod
    def sf(x):
        return expit(-x)

    @staticmethod
    def pdf(x, F, S):
        return F * S

    @staticmethod
    def dpdf(x, f, F):
        return f * (1 - 2 * F)

    @staticmethod
    def ppf(p):
        return np.log(p / (1 - p))


class _NormalLink:
    """The ordered probit's: the standard normal."""
    name = "probit"

    @staticmethod
    def cdf(x):
        return ndtr(x)

    @staticmethod
    def sf(x):
        return ndtr(-x)

    @staticmethod
    def pdf(x, F, S):
        xc = np.clip(x, -40.0, 40.0)            # the open ends, +-inf: density 0
        return np.exp(-0.5 * xc * xc) / np.sqrt(2 * np.pi)

    @staticmethod
    def dpdf(x, f, F):
        return -np.clip(x, -40.0, 40.0) * f

    @staticmethod
    def ppf(p):
        return ndtri(p)


LINKS = {"logit": _LogisticLink, "probit": _NormalLink}


class _Ordered(_Model):
    kind = "ologit"

    def __init__(self, xnames, K, levels, link="logit"):
        super().__init__(xnames, K)
        self.link = LINKS[link]
        self.levels = list(levels)
        self.J = len(levels)
        self.C = self.J - 1                 # cutpoints
        self.fix_first = False              # cut1 = 0 (with an absorbed fixed effect)

    @property
    def n_cuts(self):
        return self.C - self.fix_first

    @property
    def P(self):
        return self.K + self.n_cuts

    def names(self):
        first = 2 if self.fix_first else 1
        return [self.xnames[c] for c in self.cols] + [f"cut{j}" for j in range(first, self.J)]

    def penalized(self):
        return np.r_[np.ones(self.K, bool), np.zeros(self.n_cuts, bool)]

    def marginal_cuts(self, counts):
        cum = np.cumsum(counts)[:-1] / counts.sum()
        return self.link.ppf(cum)

    def start(self, counts, fe=False):
        """Zero slopes, and the cutpoints that give each level its sample
        share (the null model); with cut1 fixed at 0, those less cut1 (the
        fixed effects then start at -cut1)."""
        c = self.marginal_cuts(counts)
        return np.r_[np.zeros(self.K), c[1:] - c[0] if self.fix_first else c]

    def cuts(self, theta):
        c = theta[self.K:self.K + self.n_cuts]
        return np.r_[0.0, c] if self.fix_first else c

    def feasible(self, theta):
        return bool(np.all(np.isfinite(theta)) and np.all(np.diff(self.cuts(theta)) > 0))

    def x_block(self, H):
        # the cutpoints first: a covariate collinear with them (a constant,
        # a full set of indicators) is the one dropped
        K = self.K
        order = np.r_[np.arange(K, K + self.n_cuts), np.arange(K)]
        return H[np.ix_(order, order)], list(range(self.n_cuts))

    def drop(self, keep):
        idx = np.r_[np.flatnonzero(keep), self.K + np.arange(self.n_cuts)]
        self.cols = self.cols[keep]
        return idx

    def _parts(self, theta, X, y, offset):
        K = self.K
        eta = X @ theta[:K]
        if offset is not None:
            eta = eta + offset
        cp = np.r_[-np.inf, self.cuts(theta), np.inf]
        a = cp[y + 1] - eta                 # upper limit of the row's level
        b = cp[y] - eta                     # lower limit
        L = self.link
        Fa, Fb = L.cdf(a), L.cdf(b)
        Sa, Sb = L.sf(a), L.sf(b)
        # F(a) - F(b), as S(b) - S(a) when both are in the upper tail
        P = np.where(b > 0, Sb - Sa, Fa - Fb)
        fa, fb = L.pdf(a, Fa, Sa), L.pdf(b, Fb, Sb)
        with np.errstate(divide="ignore", invalid="ignore"):
            ga, gb = fa / P, -fb / P
        return P, Fa, Fb, fa, fb, ga, gb, a, b

    def rows(self, theta, X, y, w, offset=None, hessian=True):
        """Per-row pieces: the log-likelihood terms; d l / d eta (`s`); and,
        with `hessian`, the information in eta (`h`, >= 0) and between eta
        and each cutpoint (`Q`, n x C), all weighted. (A fixed effect enters
        like eta, so these are also its derivatives.)"""
        P, Fa, Fb, fa, fb, ga, gb, a, b = self._parts(theta, X, y, offset)
        with np.errstate(divide="ignore"):
            lp = np.log(P)
        wt = np.ones(len(y)) if w is None else w
        out = {"lp": lp, "s": -(ga + gb) * wt, "ga": ga * wt, "gb": gb * wt}
        if not hessian or not np.all(np.isfinite(lp)):
            return out
        C = self.C
        fpa, fpb = self.link.dpdf(a, fa, Fa), self.link.dpdf(b, fb, Fb)
        haa = fpa / P - ga ** 2
        hbb = -fpb / P - gb ** 2
        hab = -ga * gb
        up, lo = y < C, y > 0
        iu, il = np.flatnonzero(up), np.flatnonzero(lo)
        Q = np.zeros((len(y), C))
        Q[iu, y[iu]] += (wt * (haa + hab))[iu]
        Q[il, y[il] - 1] += (wt * (hab + hbb))[il]
        out.update(h=wt * np.maximum(-(haa + 2 * hab + hbb), 0.0), Q=Q,
                   haa=-(wt * haa), hbb=-(wt * hbb), hab=-(wt * hab))
        return out

    def nonzero_form(self):
        """Whether the fused pass forms the information from each row's
        nonzeros (a design that is mostly zeros), not its outer product."""
        d = getattr(self, "density", None)
        return d is not None and d <= ORD_NNZ_DENSITY

    def kernel_ok(self, n):
        """Whether a block of `n` rows is summed by the fused numba pass
        (`_ord_pass`). With an absorbed fixed effect (`fix_first`) the fixed
        effect's own passes run, not this one."""
        if n < MNL_KERNEL_MIN_ROWS or self.fix_first or self.K == 0:
            return False
        return self.K <= (ORD_NNZ_MAX_K if self.nonzero_form() else ORD_KERNEL_MAX_K)

    def information_form(self, n):
        """How the information is formed, for the fit's diagnostics:
        ``"outer-product"`` or ``"nonzero"`` (the fused numba pass), else
        ``"array"`` (NumPy and BLAS)."""
        if self.kernel_ok(n):
            return "nonzero" if self.nonzero_form() else "outer-product"
        return "array"

    def accumulate_full(self, theta, X, cols, y, w, starts, hessian, offset=None):
        """`accumulate` by the fused numba pass, on the columns `cols` of `X`
        (a whole block, read in place)."""
        K, C = self.K, self.C
        use_nnz = bool(hessian and self.nonzero_form())
        nacc = (K * K if use_nnz else K * (K + 1) // 2) if hessian else 0
        P = K + C
        n_parts = _parts(max(1, len(y) // 256), int(np.sqrt(nacc + C * K + C * C + P)))
        ll_t, g_t = np.zeros(n_parts), np.zeros((n_parts, P))
        A_t = np.zeros((n_parts, nacc))
        XC_t = np.zeros((n_parts, C if hessian else 0, K))
        CC_t = np.zeros((n_parts, C if hessian else 0, C))
        empty = np.empty(0)
        with nb.parallel_chunksize(1):
            _ord_pass(np.ascontiguousarray(X), np.ascontiguousarray(theta, np.float64),
                      np.asarray(cols, np.int64), np.ascontiguousarray(self.cuts(theta)), y,
                      empty if w is None else np.ascontiguousarray(w, np.float64),
                      empty if offset is None else np.ascontiguousarray(offset, np.float64),
                      self.link.name == "probit", bool(hessian), use_nnz, ll_t, g_t, A_t, XC_t,
                      CC_t)
        ll = float(ll_t.sum())
        if not np.isfinite(ll):
            return -np.inf, None, None
        g = g_t.sum(axis=0)
        if not hessian:
            return ll, g, None
        H = np.zeros((P, P))
        if use_nnz:
            S = A_t.sum(axis=0).reshape(K, K)
        else:
            S = np.zeros((K, K))
            S[np.triu_indices(K)] = A_t.sum(axis=0)
        H[:K, :K] = S + np.triu(S, 1).T
        xc = XC_t.sum(axis=0)
        H[:K, K:] = xc.T
        H[K:, :K] = xc
        cc = CC_t.sum(axis=0)
        H[K:, K:] = np.triu(cc) + np.triu(cc, 1).T
        return ll, g, H

    def accumulate(self, theta, X, y, w, starts, hessian, offset=None):
        if self.kernel_ok(len(y)):
            return self.accumulate_full(theta, X, np.arange(X.shape[1]), y, w, starts, hessian,
                                        offset)
        return self._assemble(self.rows(theta, X, y, w, offset, hessian), X, y, w, hessian)

    def _assemble(self, r, X, y, w, hessian):
        """Log-likelihood, gradient and information from `rows`' pieces."""
        K, C = self.K, self.C
        ll = float(r["lp"].sum() if w is None else r["lp"] @ w)
        if not np.isfinite(ll):
            return -np.inf, None, None
        up, lo = y < C, y > 0
        gcut = (_bincount(y[up], r["ga"][up], C) + _bincount(y[lo] - 1, r["gb"][lo], C))
        g = np.r_[X.T @ r["s"], gcut]
        H = None
        if hessian:
            H = np.empty((K + C, K + C))
            H[:K, :K] = _gram(X, r["h"])
            Hbc = X.T @ r["Q"]
            H[:K, K:] = Hbc
            H[K:, :K] = Hbc.T
            Hcc = np.diag(_bincount(y[up], r["haa"][up], C) + _bincount(y[lo] - 1, r["hbb"][lo], C))
            both = up & lo
            off = _bincount(y[both] - 1, r["hab"][both], C)     # (cut y-1, cut y)
            for j in range(C - 1):
                Hcc[j, j + 1] = Hcc[j + 1, j] = off[j]
            H[K:, K:] = Hcc
        if self.fix_first:
            keep = np.r_[np.arange(K), np.arange(K + 1, K + C)]
            g = g[keep]
            H = H[np.ix_(keep, keep)] if H is not None else None
        return ll, g, H

    def scores(self, theta, X, y, starts, offset=None):
        C = self.C
        r = self.rows(theta, X, y, None, offset, hessian=False)
        S = np.zeros((len(y), self.K + C))
        S[:, :self.K] = X * r["s"][:, None]
        iu, il = np.flatnonzero(y < C), np.flatnonzero(y > 0)
        S[iu, self.K + y[iu]] += r["ga"][iu]
        S[il, self.K + y[il] - 1] += r["gb"][il]
        if self.fix_first:
            S = np.delete(S, self.K, axis=1)
        return S

    def width(self, scores=False):
        return self.K + self.C


# ------------------------------------------------------------------ conditional
class _Conditional(_Model):
    kind = "clogit"
    unit = "cases"

    @property
    def P(self):
        return self.K

    def names(self):
        return [self.xnames[c] for c in self.cols]

    def penalized(self):
        return np.ones(self.K, bool)

    def start(self, counts=None, fe=False):
        return np.zeros(self.K)

    def x_block(self, H):
        # the information at zero is the design's within-set covariance, so
        # a variable that does not vary within choice sets (a characteristic
        # of the chooser not interacted with the alternatives) is dropped
        return H, []

    def drop(self, keep):
        self.cols = self.cols[keep]
        return np.flatnonzero(keep)

    @staticmethod
    def _probs(eta, starts):
        st = starts[:-1]
        sizes = np.diff(starts)
        m = np.maximum.reduceat(eta, st)
        e = np.exp(eta - np.repeat(m, sizes))
        S = np.add.reduceat(e, st)
        return e / np.repeat(S, sizes), m, S

    def accumulate(self, theta, X, y, w, starts, hessian, offset=None):
        return self.accumulate_full(theta, X, np.arange(X.shape[1]), y, w, starts, hessian,
                                    offset)

    def accumulate_full(self, theta, X, cols, y, w, starts, hessian, offset=None):
        """`accumulate` on the columns `cols` of `X`: a whole block, by
        numba's pass, which holds no row-sized temporaries and reads the
        columns in place rather than copying them; above
        CLOGIT_KERNEL_MAX_P parameters, by NumPy's."""
        P = len(cols)
        if P > CLOGIT_KERNEL_MAX_P:
            Xc = X if P == X.shape[1] and np.array_equal(cols, np.arange(P)) else X[:, cols]
            return self._accumulate_blas(theta, Xc, y, w, starts, hessian, offset)
        empty = np.empty(0)
        starts = np.asarray(starts, np.int64)
        # parts handed out one at a time as threads come free, so that a
        # thread the machine slows (another process, the store's reader, a
        # slower core) does not hold up the rest; their accumulators are
        # capped together, at one part per thread
        n_parts = _parts(len(starts) - 1, P)
        ll_t, g_t = np.zeros(n_parts), np.zeros((n_parts, P))
        H_t = np.zeros((n_parts, P, P) if hessian else (n_parts, 0, 0))
        with nb.parallel_chunksize(1):
            _clogit_pass(
                np.ascontiguousarray(X, np.float64), np.asarray(cols, np.int64),
                np.ascontiguousarray(theta, np.float64), y,
                empty if w is None else np.ascontiguousarray(w, np.float64),
                empty if offset is None else np.ascontiguousarray(offset, np.float64),
                starts, bool(hessian), int(np.diff(starts).max()) if len(starts) > 1 else 1,
                ll_t, g_t, H_t)
        ll, g, H = ll_t.sum(), g_t.sum(axis=0), H_t.sum(axis=0)
        if not hessian:
            return float(ll), g, None
        return float(ll), g, np.triu(H) + np.triu(H, 1).T

    def _accumulate_blas(self, theta, X, y, w, starts, hessian, offset=None):
        """`accumulate` by NumPy array operations, the information by BLAS's
        Gram products: for a wide design (`CLOGIT_KERNEL_MAX_P`)."""
        st = starts[:-1]
        eta = X @ theta
        if offset is not None:
            eta = eta + offset
        p, m, S = self._probs(eta, starts)
        lc = np.add.reduceat(np.where(y == 1, eta, 0.0), st) - m - np.log(S)
        wc = None if w is None else w[st]
        ll = float(lc.sum() if wc is None else lc @ wc)
        r = y - p
        if w is not None:
            r = r * w
        g = X.T @ r
        if not hessian:
            return ll, g, None
        H = _gram(X, p if w is None else p * w)
        Xbar = np.add.reduceat(X * p[:, None], st, axis=0)
        H -= _gram(Xbar, np.ones(len(st)) if wc is None else wc)
        return ll, g, H

    def scores(self, theta, X, y, starts, offset=None):
        """Each choice set's score, n_sets x K."""
        eta = X @ theta
        if offset is not None:
            eta = eta + offset
        p, _, _ = self._probs(eta, starts)
        return np.add.reduceat(X * (y - p)[:, None], starts[:-1], axis=0)


# ------------------------------------------------------------------ conditional FE logit
@nb.njit(parallel=True, cache=True)
def _cfe_units(X, eta, y, w, starts, hessian, ll_u, g_u, acc):
    """Chamberlain's conditional likelihood, unit by unit (the rows of unit u
    are starts[u]:starts[u+1]): with k the unit's number of 1s among its T
    rows, log P(y | k) = sum y eta - log e_k(exp(eta)), e_k the elementary
    symmetric polynomial. One pass over the unit's rows builds, for j <= k,
    e_j, G_j = the sum over subsets of size j of their product times their x
    sum, and (with `hessian`) H_j = the same with the x sum's outer product:
    then E[sum d x | k] = G_k / e_k and Cov = H_k / e_k - (G_k / e_k)^2, the
    gradient sum y x - G_k / e_k and the information Cov. All the sums are of
    positive terms until that last subtraction; exp(eta) is scaled by the
    unit's largest, and a unit with k > T / 2 is fitted as its complement
    (1 - y, -x, -eta), so that k <= T / 2. Units with k = 0 or k = T carry no
    information and are skipped. Writes each unit's (weighted) log-likelihood
    and gradient; the information is summed over ``len(acc)`` parts of the
    units, part t's into ``acc[t]``, for the caller to add up.

    A unit's work grows with T k P^2, so units differ in cost: the parts are
    more than the threads, handed out as threads come free (`_CFELogit._units`
    runs this under ``parallel_chunksize(1)``), and each takes every
    ``len(acc)``-th run of CFE_RUN units, so that each has a mix of short and
    long panels. The only parallel loop is the
    one over the parts, so the accumulators are allocated and summed by the
    caller (numba would otherwise hand out every element of an array
    operation here as a chunk of its own)."""
    U, P = len(starts) - 1, X.shape[1]
    n_parts = len(acc)
    for th in nb.prange(n_parts):
        # (a part's own accumulator, copied out at the end: the parts'
        # slices of `acc` are too close together to add into unit by unit)
        a_ = np.zeros((P, P))
        # runs of CFE_RUN units, every n_parts-th run: data in order of panel
        # length (a cohort, an entry date) would otherwise put the longest,
        # costliest units together in the last parts, while single units
        # dealt out would scatter the reads
        for u0 in range(th * CFE_RUN, U, n_parts * CFE_RUN):
            for u in range(u0, min(U, u0 + CFE_RUN)):
                r0, r1 = starts[u], starts[u + 1]
                T = r1 - r0
                k = 0
                for r in range(r0, r1):
                    k += 1 if y[r] > 0 else 0
                ll_u[u] = 0.0
                for p in range(P):
                    g_u[u, p] = 0.0
                if k == 0 or k == T:
                    continue
                s = 1.0
                if 2 * k > T:
                    s, k = -1.0, T - k
                c = -np.inf
                for r in range(r0, r1):
                    c = max(c, s * eta[r])
                e = np.zeros(k + 1)
                G = np.zeros((k + 1, P))
                Hm = np.zeros((k + 1, P, P)) if hessian else np.zeros((1, 1, 1))
                e[0] = 1.0
                obs = 0.0
                for r in range(r0, r1):
                    z = np.exp(s * eta[r] - c)
                    yr = (y[r] > 0) if s > 0 else (y[r] <= 0)
                    if yr:
                        obs += s * eta[r]
                        for p in range(P):
                            g_u[u, p] += s * X[r, p]
                    for j in range(k, 0, -1):
                        if hessian:
                            for p in range(P):
                                xp = s * X[r, p]
                                for q in range(P):
                                    xq = s * X[r, q]
                                    Hm[j, p, q] += z * (Hm[j - 1, p, q] + xp * G[j - 1, q]
                                                        + G[j - 1, p] * xq + xp * xq * e[j - 1])
                        for p in range(P):
                            G[j, p] += z * (G[j - 1, p] + s * X[r, p] * e[j - 1])
                        e[j] += z * e[j - 1]
                wu = w[r0]
                ll_u[u] = wu * (obs - np.log(e[k]) - k * c)
                for p in range(P):
                    m = G[k, p] / e[k]
                    g_u[u, p] = wu * (g_u[u, p] - m)
                if hessian:
                    for p in range(P):
                        mp = G[k, p] / e[k]
                        for q in range(P):
                            a_[p, q] += wu * (Hm[k, p, q] / e[k] - mp * G[k, q] / e[k])
        for p in range(P):
            for q in range(P):
                acc[th, p, q] = a_[p, q]


class _CFELogit(_Model):
    """Chamberlain's (1980) conditional fixed-effects logit: each unit's
    rows (a "case") are conditioned on their number of 1s, which removes the
    unit's effect from the likelihood -- consistent for a fixed number of
    periods, unlike the logit with the effects estimated. Weights are per
    unit."""
    kind = "cfelogit"
    unit = "cases"

    @property
    def P(self):
        return self.K

    def names(self):
        return [self.xnames[c] for c in self.cols]

    def penalized(self):
        return np.ones(self.K, bool)

    def start(self, counts=None, fe=False):
        return np.zeros(self.K)

    def x_block(self, H):
        # a covariate constant within every unit has no information: dropped
        return H, []

    def drop(self, keep):
        self.cols = self.cols[keep]
        return np.flatnonzero(keep)

    def _units(self, theta, X, y, w, starts, hessian, offset):
        eta = X @ theta
        if offset is not None:
            eta = eta + offset
        U, P = len(starts) - 1, X.shape[1]
        ll_u, g_u = np.empty(U), np.empty((U, P))
        acc = np.zeros((_parts(U, P), P, P))
        ww = np.ones(len(y)) if w is None else np.ascontiguousarray(w, dtype=np.float64)
        with nb.parallel_chunksize(1):
            _cfe_units(np.ascontiguousarray(X), np.ascontiguousarray(eta), y, ww,
                       np.ascontiguousarray(starts, dtype=np.int64), hessian, ll_u, g_u, acc)
        return ll_u, g_u, acc.sum(axis=0)

    def accumulate(self, theta, X, y, w, starts, hessian, offset=None):
        ll_u, g_u, H = self._units(theta, X, y, w, starts, hessian, offset)
        return float(ll_u.sum()), g_u.sum(axis=0), (H if hessian else None)

    def scores(self, theta, X, y, starts, offset=None):
        """Each unit's score, n_units x K (unweighted: the meat weights it)."""
        return self._units(theta, X, y, None, starts, False, offset)[1]
