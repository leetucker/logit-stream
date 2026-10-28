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

import numba as nb
import numpy as np
from scipy.special import expit, ndtr, ndtri

# The multinomial information from the rows' nonzeros (`_mnl_info_sparse`)
# rather than dense Gram products, when the design is at most this dense on
# average, and the parameters few enough for a per-thread accumulator each
SPARSE_DENSITY = 0.5
SPARSE_MAX_PARAMS = 1200


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

    def __init__(self, xnames, K, levels, intercept):
        super().__init__(xnames, K)
        self.levels = list(levels)
        self.J = len(levels)
        self.intercept = intercept          # index of the Intercept column, or None

    @property
    def P(self):
        return self.K * (self.J - 1)

    def names(self):
        return [f"{lv}:{self.xnames[c]}" for lv in self.levels[1:] for c in self.cols]

    def penalized(self):
        pen = np.ones((self.J - 1, self.K), bool)
        if self.intercept is not None and self.intercept in self.cols:
            pen[:, int(np.flatnonzero(self.cols == self.intercept)[0])] = False
        return pen.ravel()

    def start(self, counts, fe=False):
        """Zero slopes, and the intercepts that give each level its sample
        share: the null model, so that the first log-likelihood is the null
        log-likelihood."""
        B = np.zeros((self.J - 1, self.K))
        if self.intercept is not None and not fe:
            B[:, self.intercept] = np.log(counts[1:] / counts[0])
        return B.ravel()

    def x_block(self, H):
        return H[:self.K, :self.K], []

    def drop(self, keep):
        idx = np.flatnonzero(np.tile(keep, self.J - 1))
        self.cols = self.cols[keep]
        if self.intercept is not None and self.intercept not in self.cols:
            self.intercept = None
        return idx

    def width(self, scores=False):
        return self.K * (self.J - 1) if scores else self.K + self.J

    def eta(self, theta, X):
        return X @ theta.reshape(self.J - 1, self.K).T

    @staticmethod
    def probs(E):
        """Probabilities of levels 1..J-1 and of the base level, from the
        n x (J-1) linear predictors, stably; and log of the denominator."""
        m = np.maximum(E.max(axis=1), 0.0) if E.shape[1] else np.zeros(len(E))
        ex = np.exp(E - m[:, None])
        e0 = np.exp(-m)
        den = e0 + ex.sum(axis=1)
        return ex, e0, den, m + np.log(den)

    def accumulate(self, theta, X, y, w, starts, hessian, offset=None):
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
        g = (X.T @ R).T.ravel()
        if not hessian:
            return ll, g, None
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

    def accumulate(self, theta, X, y, w, starts, hessian, offset=None):
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
def _cfe_units(X, eta, y, w, starts, hessian, n_threads, ll_u, g_u, out):
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
    and gradient; the information is summed into `out` over `n_threads`
    blocks of units."""
    U, P = len(starts) - 1, X.shape[1]
    acc = np.zeros((n_threads, P, P))
    step = (U + n_threads - 1) // n_threads
    for th in nb.prange(n_threads):
        # (a thread's own accumulator, copied out at the end: the threads'
        # slices of `acc` are too close together to add into unit by unit)
        a_ = np.zeros((P, P))
        for u in range(th * step, min(U, (th + 1) * step)):
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
        acc[th] = a_
    for th in range(n_threads):
        out += acc[th]


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
        H = np.zeros((P, P))
        ww = np.ones(len(y)) if w is None else np.ascontiguousarray(w, dtype=np.float64)
        _cfe_units(np.ascontiguousarray(X), np.ascontiguousarray(eta), y, ww,
                   np.ascontiguousarray(starts, dtype=np.int64), hessian,
                   nb.get_num_threads(), ll_u, g_u, H)
        return ll_u, g_u, H

    def accumulate(self, theta, X, y, w, starts, hessian, offset=None):
        ll_u, g_u, H = self._units(theta, X, y, w, starts, hessian, offset)
        return float(ll_u.sum()), g_u.sum(axis=0), (H if hessian else None)

    def scores(self, theta, X, y, starts, offset=None):
        """Each unit's score, n_units x K (unweighted: the meat weights it)."""
        return self._units(theta, X, y, None, starts, False, offset)[1]
