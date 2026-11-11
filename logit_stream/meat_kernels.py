"""Fused numba passes for the robust and clustered variances of the
multinomial and ordered logit (without an absorbed fixed effect).

`_Driver.meat` reads the rows once more to form the sandwich's middle
matrices: sum w' s s' over the rows ("hetero"), and for each clustering the
sums of w s within each cluster. In NumPy that builds every row's score as a
rows-by-parameters array and, per clustering, sorts each chunk by cluster to
sum runs of it. Here one kernel does it per row, in registers, in parallel:
the row's probabilities and residuals, the outer-product accumulators of the
Hessian passes for s s' (a multinomial row's score is its residuals times
its covariates, so the block of equations j and l is R_j R_l x x'; an ordered
row's is the linear predictor's score times x, and at most two cutpoint
entries), and one add of w s to the row's cluster in each clustering. The
clusters' sums are kept per part (thread): `MEAT_CLUSTER_MB` bounds them
together, and with more clusters than that allows the NumPy code runs.

The weights: s s' is weighted by w (frequency weights) or w^2 (analytic
weights) and the clusters' sums by w, as `_Driver._add_meat`.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from . import models as M_
from .models import _assemble_tri, _ord_terms, _parts

# the clusters' sums, per part, all together at most this much
MEAT_CLUSTER_MB = 256.0


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _mnl_meat_pass(X, theta, colmap, pos, ncols, off, y, w, wmode, cl, cloff, hetero,
                   acc_t, U_t):
    """Multinomial: the scores' outer products (``acc_t[t]``: for each pair of
    equations j <= l, the upper triangle of the K design columns' products,
    weighted by m w R_j R_l) and the clusters' sums of scores (``U_t[t]``: the
    rows ``cloff[k] + cl[i, k]`` for clustering k, P columns). R_j = 1[y = j]
    - p_j. `wmode`: 0 no weights, 1 analytic, 2 frequency."""
    n = len(y)
    J1 = len(ncols)
    K = len(colmap)
    ncl = cl.shape[1]
    ntri = K * (K + 1) // 2
    n_parts = acc_t.shape[0]
    has_w = wmode > 0
    step = (n + n_parts - 1) // n_parts
    for t in nb.prange(n_parts):
        u = np.empty(K)
        xxt = np.empty(ntri)
        et = np.empty(J1)
        r = np.empty(J1)
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
                r[j] = np.exp(et[j] - m)
                s += r[j]
            inv = 1.0 / s
            yi = y[i]
            for j in range(J1):
                r[j] = (1.0 if yi == j + 1 else 0.0) - r[j] * inv
            for k in range(ncl):
                row = cloff[k] + cl[i, k]
                for j in range(J1):
                    o = off[j]
                    rj = wi * r[j]
                    for a in range(ncols[j]):
                        U_t[t, row, o + a] += rj * u[pos[j, a]]
            if hetero:
                mw = 1.0 if wmode == 0 else (wi * wi if wmode == 1 else wi)
                k = 0
                for c1 in range(K):
                    x1 = u[c1]
                    for c2 in range(c1, K):
                        xxt[k] = x1 * u[c2]
                        k += 1
                pr = 0
                for j in range(J1):
                    mj = mw * r[j]
                    for l in range(j, J1):
                        v = mj * r[l]
                        for k in range(ntri):
                            acc_t[t, pr, k] += v * xxt[k]
                        pr += 1


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _ord_meat_pass(X, theta, colmap, cuts, y, w, off, wmode, probit, cl, cloff, hetero,
                   A_t, XC_t, CC_t, U_t):
    """Ordered: a row's score is s = -(g_a + g_b) times x for the slopes, g_a
    at cutpoint y and g_b at cutpoint y - 1 (`_Ordered.scores`). The
    outer products (``A_t``: x x' times m s^2, upper triangle; ``XC_t``:
    x against the cutpoints; ``CC_t``: the cutpoints, the superdiagonal in
    the upper triangle) and the clusters' sums (``U_t``), as the
    multinomial's."""
    n = len(y)
    K = len(colmap)
    C = len(cuts)
    ncl = cl.shape[1]
    ntri = K * (K + 1) // 2
    n_parts = U_t.shape[0]
    has_off = len(off) > 0
    has_w = wmode > 0
    step = (n + n_parts - 1) // n_parts
    for t in nb.prange(n_parts):
        u = np.empty(K)
        acc = np.zeros(ntri)
        xc = np.zeros((C, K))
        cc = np.zeros((C, C))
        for i in range(t * step, min(n, (t + 1) * step)):
            wi = w[i] if has_w else 1.0
            yi = y[i]
            eta = off[i] if has_off else 0.0
            for k in range(K):
                u[k] = X[i, colmap[k]]
                eta += u[k] * theta[k]
            a = cuts[yi] - eta if yi < C else np.inf
            b = cuts[yi - 1] - eta if yi > 0 else -np.inf
            Pr, ga, gb, haa, hbb, hab = _ord_terms(a, b, probit, False)
            s = -(ga + gb)
            for k in range(ncl):
                row = cloff[k] + cl[i, k]
                for c in range(K):
                    U_t[t, row, c] += wi * s * u[c]
                if yi < C:
                    U_t[t, row, K + yi] += wi * ga
                if yi > 0:
                    U_t[t, row, K + yi - 1] += wi * gb
            if hetero:
                mw = 1.0 if wmode == 0 else (wi * wi if wmode == 1 else wi)
                k = 0
                for c1 in range(K):
                    hx = mw * s * s * u[c1]
                    for c2 in range(c1, K):
                        acc[k] += hx * u[c2]
                        k += 1
                if yi < C:
                    q = mw * s * ga
                    for k in range(K):
                        xc[yi, k] += q * u[k]
                    cc[yi, yi] += mw * ga * ga
                if yi > 0:
                    q = mw * s * gb
                    for k in range(K):
                        xc[yi - 1, k] += q * u[k]
                    cc[yi - 1, yi - 1] += mw * gb * gb
                    if yi < C:
                        cc[yi - 1, yi] += mw * ga * gb
        if hetero:
            for k in range(ntri):
                A_t[t, k] = acc[k]
            for c in range(C):
                for k in range(K):
                    XC_t[t, c, k] = xc[c, k]
                for d in range(C):
                    CC_t[t, c, d] = cc[c, d]


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _mnl_cluster_pass(X, theta, colmap, pos, ncols, off, y, w, has_w, code, bounds, U):
    """The multinomial's clusters' sums of scores for one clustering, for
    many clusters: part t owns the cluster codes ``bounds[t]`` to
    ``bounds[t + 1]`` and adds only the rows of those clusters to the rows of
    `U` (G x P, shared), so that nothing is kept per part. Every part reads
    every row's code and computes the scores of its own rows."""
    n = len(y)
    J1 = len(ncols)
    K = len(colmap)
    n_parts = len(bounds) - 1
    for t in nb.prange(n_parts):
        lo = bounds[t]
        hi = bounds[t + 1]
        u = np.empty(K)
        r = np.empty(J1)
        for i in range(n):
            c = code[i]
            if c < lo or c >= hi:
                continue
            wi = w[i] if has_w else 1.0
            for k in range(K):
                u[k] = X[i, colmap[k]]
            m = 0.0
            for j in range(J1):
                o = off[j]
                e = 0.0
                for a in range(ncols[j]):
                    e += u[pos[j, a]] * theta[o + a]
                r[j] = e
                if e > m:
                    m = e
            s = np.exp(-m)
            for j in range(J1):
                r[j] = np.exp(r[j] - m)
                s += r[j]
            inv = 1.0 / s
            yi = y[i]
            for j in range(J1):
                rj = wi * ((1.0 if yi == j + 1 else 0.0) - r[j] * inv)
                o = off[j]
                for a in range(ncols[j]):
                    U[c, o + a] += rj * u[pos[j, a]]


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _ord_cluster_pass(X, theta, colmap, cuts, y, w, off, has_w, probit, code, bounds, U):
    """`_mnl_cluster_pass` for the ordered models."""
    n = len(y)
    K = len(colmap)
    C = len(cuts)
    has_off = len(off) > 0
    n_parts = len(bounds) - 1
    for t in nb.prange(n_parts):
        lo = bounds[t]
        hi = bounds[t + 1]
        for i in range(n):
            c = code[i]
            if c < lo or c >= hi:
                continue
            wi = w[i] if has_w else 1.0
            yi = y[i]
            eta = off[i] if has_off else 0.0
            for k in range(K):
                eta += X[i, colmap[k]] * theta[k]
            a = cuts[yi] - eta if yi < C else np.inf
            b = cuts[yi - 1] - eta if yi > 0 else -np.inf
            Pr, ga, gb, haa, hbb, hab = _ord_terms(a, b, probit, False)
            s = wi * -(ga + gb)
            for k in range(K):
                U[c, k] += s * X[i, colmap[k]]
            if yi < C:
                U[c, K + yi] += wi * ga
            if yi > 0:
                U[c, K + yi - 1] += wi * gb


class _MeatPass:
    """Block by block through the kernel: `add` each block, then `result`."""

    def __init__(self, driver, theta, weights_type, clusters, hetero):
        m = driver.model
        self.m, self.theta = m, np.ascontiguousarray(theta, np.float64)
        self.P = len(theta)
        self.hetero = bool(hetero)
        self.wmode = {"aweights": 1, "fweights": 2}[weights_type]
        self.names = list(clusters)
        self.cols = [clusters[nm][0] for nm in self.names]
        sizes = [clusters[nm][1] for nm in self.names]
        self.cloff = np.r_[0, np.cumsum(sizes)].astype(np.int64)
        self.sumG = int(self.cloff[-1])
        self.U = np.zeros((self.sumG, self.P))
        self.sizes = sizes

    @property
    def combined(self):
        """Whether each part may keep its own copy of the clusters' sums (and one
        kernel forms them with the outer products), which needs a copy per
        thread within `MEAT_CLUSTER_MB` and, per block, no more clusters than
        an eighth of its rows (zeroing and adding up the copies costs about
        that much); else each part owns a range of the clusters
        (`_mnl_cluster_pass`) after a pass for the outer products."""
        return self.sumG * self.P * 8 * nb.get_num_threads() <= MEAT_CLUSTER_MB * 1e6

    def n_parts(self, n, acc_elems):
        """Parts for a block of `n` rows; with a copy of the clusters' sums per
        part, no more than `MEAT_CLUSTER_MB` holds."""
        parts = _parts(max(1, n // 256), int(np.sqrt(max(acc_elems, 1))))
        if self.sumG and self.combined and self.sumG * 8 <= n:
            parts = min(parts, max(1, int(MEAT_CLUSTER_MB * 1e6 // (self.sumG * self.P * 8))))
        return parts

    def owner_bounds(self, k):
        """The cluster codes of clustering `k` cut into one range per part."""
        parts = 4 * nb.get_num_threads()
        return np.linspace(0, self.sizes[k], parts + 1).astype(np.int64)

    def _clusters_of(self, blk, n):
        if not self.names:
            return np.zeros((n, 0), np.int64)
        cl = blk["cl"]
        return np.ascontiguousarray(cl[:, self.cols], np.int64)

    def result(self):
        U = {nm: self.U[self.cloff[k]:self.cloff[k + 1]] for k, nm in enumerate(self.names)}
        return (self.M if self.hetero else None), U


class MultinomialMeat(_MeatPass):
    def __init__(self, driver, theta, weights_type, clusters, hetero):
        super().__init__(driver, theta, weights_type, clusters, hetero)
        m = self.m
        J1, K = m.J - 1, m.K
        self.J1, self.K = J1, K
        M = m.free()
        self.ncols = M.sum(axis=1).astype(np.int64)
        self.pos = np.zeros((J1, K), np.int64)
        for j in range(J1):
            self.pos[j, :self.ncols[j]] = np.flatnonzero(M[j])
        self.off = np.r_[0, np.cumsum(self.ncols)].astype(np.int64)
        self.ntri, self.npair = K * (K + 1) // 2, J1 * (J1 + 1) // 2
        self.acc = np.zeros((self.npair, self.ntri)) if hetero else None

    @property
    def M(self):
        return _assemble_tri(self.acc, self.pos, self.ncols, self.off, self.K)

    def add(self, blk):
        n = len(blk["y"])
        combined = self.combined and self.sumG * 8 <= n
        n_parts = self.n_parts(n, self.npair * self.ntri if self.hetero else self.P)
        X = np.ascontiguousarray(blk["X"])
        cols = np.asarray(self.m.cols, np.int64)
        w = blk.get("w")
        wv = np.empty(0) if w is None else np.ascontiguousarray(w, np.float64)
        wmode = self.wmode if w is not None else 0
        cl = self._clusters_of(blk, n)
        hz = self.hetero
        acc_t = np.zeros((n_parts, self.npair if hz else 0, self.ntri if hz else 0))
        U_t = np.zeros((n_parts, self.sumG if combined else 0, self.P))
        with nb.parallel_chunksize(1):
            if hz or combined:
                _mnl_meat_pass(X, self.theta, cols, self.pos, self.ncols, self.off, blk["y"], wv,
                               wmode, cl if combined else cl[:, :0], self.cloff, hz, acc_t, U_t)
            if not combined:
                for k in range(len(self.names)):
                    _mnl_cluster_pass(X, self.theta, cols, self.pos, self.ncols, self.off,
                                      blk["y"], wv, w is not None,
                                      np.ascontiguousarray(cl[:, k]), self.owner_bounds(k),
                                      self.U[self.cloff[k]:self.cloff[k + 1]])
        if hz:
            self.acc += acc_t.sum(axis=0)
        if combined:
            self.U += U_t.sum(axis=0)
        return True


class OrderedMeat(_MeatPass):
    def __init__(self, driver, theta, weights_type, clusters, hetero):
        super().__init__(driver, theta, weights_type, clusters, hetero)
        m = self.m
        self.K, self.C = m.K, m.C
        self.cuts = np.ascontiguousarray(m.cuts(self.theta))
        K, C = self.K, self.C
        self.ntri = K * (K + 1) // 2
        self.S = np.zeros(self.ntri)
        self.XC, self.CC = np.zeros((C, K)), np.zeros((C, C))

    @property
    def M(self):
        K, C = self.K, self.C
        S = np.zeros((K, K))
        S[np.triu_indices(K)] = self.S
        H = np.zeros((K + C, K + C))
        H[:K, :K] = S + np.triu(S, 1).T
        H[:K, K:] = self.XC.T
        H[K:, :K] = self.XC
        H[K:, K:] = np.triu(self.CC) + np.triu(self.CC, 1).T
        return H

    def add(self, blk):
        n = len(blk["y"])
        K, C = self.K, self.C
        combined = self.combined and self.sumG * 8 <= n
        n_parts = self.n_parts(n, self.ntri + C * K + C * C + self.P)
        w, off = blk.get("w"), blk.get("off")
        hz = self.hetero
        A_t = np.zeros((n_parts, self.ntri if hz else 0))
        XC_t = np.zeros((n_parts, C if hz else 0, K))
        CC_t = np.zeros((n_parts, C if hz else 0, C))
        U_t = np.zeros((n_parts, self.sumG if combined else 0, self.P))
        empty = np.empty(0)
        X = np.ascontiguousarray(blk["X"])
        cols = np.asarray(self.m.cols, np.int64)
        wv = empty if w is None else np.ascontiguousarray(w, np.float64)
        ov = empty if off is None else np.ascontiguousarray(off, np.float64)
        wmode = self.wmode if w is not None else 0
        probit = self.m.link.name == "probit"
        cl = self._clusters_of(blk, n)
        with nb.parallel_chunksize(1):
            if hz or combined:
                _ord_meat_pass(X, self.theta, cols, self.cuts, blk["y"], wv, ov, wmode, probit,
                               cl if combined else cl[:, :0], self.cloff, hz, A_t, XC_t, CC_t,
                               U_t)
            if not combined:
                for k in range(len(self.names)):
                    _ord_cluster_pass(X, self.theta, cols, self.cuts, blk["y"], wv, ov,
                                      w is not None, probit, np.ascontiguousarray(cl[:, k]),
                                      self.owner_bounds(k),
                                      self.U[self.cloff[k]:self.cloff[k + 1]])
        if hz:
            self.S += A_t.sum(axis=0)
            self.XC += XC_t.sum(axis=0)
            self.CC += CC_t.sum(axis=0)
        if combined:
            self.U += U_t.sum(axis=0)
        return True


def make(driver, theta, weights_type, clusters, hetero):
    """The fused variance pass for `driver`'s model, or None when the NumPy code
    should run (a fixed effect, a device, a design the fused passes of the
    Newton steps do not take)."""
    m = driver.model
    if driver.fe is not None or driver.device is not None:
        return None
    if m.kind == "mlogit":
        if m.K > M_.MNL_KERNEL_MAX_K or m.nonzero_form():
            return None
        return MultinomialMeat(driver, theta, weights_type, clusters, hetero)
    if m.kind == "ologit":
        if m.K == 0 or m.K > M_.ORD_KERNEL_MAX_K or m.fix_first:
            return None
        return OrderedMeat(driver, theta, weights_type, clusters, hetero)
    return None


def min_rows():
    """The smallest block the fused pass is used for (that of the Newton passes)."""
    return M_.MNL_KERNEL_MIN_ROWS
