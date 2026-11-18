"""Adaptive Gauss-Hermite quadrature for the multinomial logit's random
intercepts (``random_method="adaptive"`` with `mlogit_stream`): one random
grouping, a (J - 1)-vector of intercepts per group with an unstructured or
diagonal Sigma, optionally within an absorbed fixed effect that the grouping
nests in. `adaptive.py` is the scalar case (binary and ordered).

**Why.** Laplace's approximation replaces each group's integral over its
effects by a Gaussian around the mode. With few rows per group the
integrand is far from Gaussian, and the approximation's maximum is biased.
In a simulation with J = 5, sds of 0.6 to 0.8 and three known rows per
group, the common levels' variances came out about a fifth too low, the
rarest level's (about 4% of the rows) twice too high, and the correlations
40% too low.

**The approximation.** Group g's likelihood is

    L_g = int f_g(a) phi(a; 0, Sigma) da
        ~ |R_g| sum_k omega_k f_g(a_gk) phi(a_gk; 0, Sigma) / phi(z_k),

with f_g the product of the group's rows' probabilities and a_gk = m_g +
R_g z_k. Here m_g is the group's conditional mode, R_g R_g' is the inverse
of the curvature there (the Laplace approximation's covariance), and (z_k,
omega_k) is the product of K one-dimensional Gauss-Hermite rules for the
standard normal (K^(J-1) points). Centred and scaled this way, few points
per dimension are accurate. In a check with J = 3 and groups of one to
fifteen rows, 5 per dimension put the total log-likelihood within 0.005 of
brute-force integration (the worst group within 4e-5). Quasi-Monte Carlo
importance sampling with 1,024 points was 0.1 off (its error is common to
all groups, since the points are shared). Pruning the product's low-weight
corners lost most of the accuracy.

Groups of LARGE_ROWS rows or more, whose integrands are close to normal,
take the smaller rule of SMALL_NODES points per dimension (81 for J = 5).
One point (Laplace's approximation) would not do: with the points held
during a step it loses the posterior's spread from Sigma's equation, and
biases Sigma down. With a third of the groups of 30 to 90 rows, the
estimates moved by less than 0.05 of their standard errors. The reported
log-likelihood and the groups' moments come from one more read with the
full rule for every group.

**The fit.** It is that of `adaptive.py`, in more dimensions. Newton's
method runs in the coefficients b, the fixed effect's free levels and
Sigma^-1's elements, with the points held where they are during a step.
The value, gradient and information are then exact sums over each group's
rows at each of its points:

    d log L_g = sum_k pi_gk d log w_gk,
    d2 log L_g = sum_k pi_gk d2 log w_gk + Cov_pi(d log w_g.),

with pi_gk the point's share of L_g. Sigma is parametrized by its inverse,
Lambda, whose log prior density, 1/2 log|Lambda| - 1/2 a'Lambda a, has a
second derivative that does not depend on the point. The fixed effect's
block of the information is block-diagonal (a group is in one of its
levels) and is eliminated. Between steps the points move: each read also
gives every group's score and curvature at its mode, from which one Newton
step updates the mode and R_g. Converged when the Newton decrement is
below `tol` and the points have stopped moving.

The start is the joint-mode fit (lme4's ``nAGQ = 0``). The standard errors
come from the inverse of the whole information (b, the fixed effect and
Lambda), carried to Sigma's elements by the delta method. The
coefficients' standard errors include Sigma's uncertainty, as the Laplace
fit's do.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numba as nb
import numpy as np
from hdfe_stream.core.report import warn
from numpy.polynomial.hermite import hermgauss

from ..newton import _design, _Driver

MNL_NODES = 5               # the default points per dimension (random_nodes)
MNL_MIN_NODES = 5           # fewer bias the fit (3 per dimension: a rare level's variance
                            # up by half with ~3 rows per group, as Laplace's)
MAX_POINTS = 20_000         # points per group (random_nodes^(J-1)), at most
LARGE_ROWS = 30             # groups of this many rows or more take SMALL_NODES per dimension
SMALL_NODES = 3
VAR_MIN = 1e-6              # the smallest eigenvalue of Sigma a step may reach
HALF_LOG_2PI = 0.5 * float(np.log(2 * np.pi))


def _rule(k, d):
    """The product of k-point Gauss-Hermite rules for the d-dimensional
    standard normal: points z (n x d), log(omega exp(|z|^2 / 2) (2 pi)^(d/2)),
    and the index of the centre (a point of weight zero is added when k is
    even, for the mode's step)."""
    x, w = hermgauss(k)
    z1 = np.sqrt(2.0) * x
    lw1 = np.log(w / np.sqrt(np.pi))
    Z = np.array(np.meshgrid(*[z1] * d, indexing="ij")).reshape(d, -1).T
    LW = np.array(np.meshgrid(*[lw1] * d, indexing="ij")).reshape(d, -1).T.sum(axis=1)
    LW = LW + 0.5 * (Z * Z).sum(axis=1) + d * HALF_LOG_2PI
    if k % 2 == 0:
        Z, LW = np.vstack([Z, np.zeros((1, d))]), np.r_[LW, -np.inf]
    return Z, LW, int(np.flatnonzero((Z == 0.0).all(axis=1))[0])


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _group_kernel(E0, A, vx, y, w, X, st, kc, logL, pi, Pbar, PP, Ds, Db, score0, curv0):
    """Each group's quadrature (the groups' rows st[g]:st[g+1] of a chunk),
    in parallel over the groups: its log-likelihood and point shares from
    the points' log-weights (vx, all but the rows' log-likelihood); each
    row's pi-mean probabilities and products of probabilities; and each
    point's deviations from the pi-means of the group's probability sums
    (Ds: levels; Db: levels 1.. by the columns of X), which are the
    deviations of the levels' and b's scores negated; the probabilities at
    the centre point kc give the group's score and curvature there."""
    G, Kt, d = A.shape
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
                        e += A[g, k, j - 1]
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
        # the point shares
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
        # the rows' pi-means, and the points' sums
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
                    q = pk * pa
                    for c in range(a, J):
                        PP[i, a, c] += q * p[ii, k, c]
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
        # the centre point: the score and curvature in the random levels
        for a in range(d):
            score0[g, a] = 0.0
            for c in range(d):
                curv0[g, a, c] = 0.0
        for ii in range(ng):
            i = r0 + ii
            wi = w[i]
            for a in range(d):
                pa = p[ii, kc, a + 1]
                score0[g, a] += wi * ((1.0 if y[i] == a + 1 else 0.0) - pa)
                curv0[g, a, a] += wi * pa
                for c in range(d):
                    curv0[g, a, c] -= wi * pa * p[ii, kc, c + 1]


def _sigma_params(d, cov):
    own = [(j, j) for j in range(d)]
    if cov != "diagonal":
        own += [(j, l) for j in range(d) for l in range(j + 1, d)]
    return own


def _E(d, own):
    """The symmetric unit matrices of the parameters (j, l)."""
    out = np.zeros((len(own), d, d))
    for p, (j, l) in enumerate(own):
        out[p, j, l] = out[p, l, j] = 1.0
    return out


class _AdaptiveMNL:
    """The adaptive quadrature fit of a multinomial `_RandomEffects` with
    one random grouping (see the module docstring)."""

    def __init__(self, re, driver, n_nodes, log, logger=None, large_rows=None):
        large_rows = LARGE_ROWS if large_rows is None else large_rows
        if (re.model.kind != "mlogit" or len(re.random) != 1 or re.kind != "tree"
                or re.factors[0] is not re.random[0]):
            raise ValueError("random_method='adaptive' is for mlogit_stream with one random "
                             "grouping")
        if getattr(re.model, "own", None):
            raise ValueError("random_method='adaptive' does not take own_covariates= yet")
        if not getattr(re.p0, "fe_sorted", False):
            raise ValueError(
                "random_method='adaptive' needs each group's rows in one block of the store, "
                "and they could not be grouped (a group larger than a block, or too little "
                "memory to sort them; see the log): raise block_mb or max_memory_gb")
        if int(n_nodes) != n_nodes or n_nodes < 1:
            raise ValueError("random_nodes must be a positive integer")
        self.re, self.driver, self.log, self.logger = re, driver, log, logger
        self.m = re.model
        self.f = re.random[0]
        self.fe = re.fixed_factor
        self.J, self.d = re.J, re.d
        self.own = _sigma_params(self.d, re.cov)
        self.Ep = _E(self.d, self.own)
        self.ns = len(self.own)
        self.K = int(n_nodes)
        if self.K ** self.d > MAX_POINTS:
            raise ValueError(
                f"random_method='adaptive' with {self.J} outcome levels and random_nodes="
                f"{self.K} needs {self.K ** self.d:,} points per group ({self.K}^{self.d}); at "
                f"most {MAX_POINTS:,} are supported: use fewer nodes, or "
                "random_method='laplace'")
        self.z, self.lw, self.kc = _rule(self.K, self.d)
        # groups of LARGE_ROWS rows or more, whose integrands are close to normal,
        # take the smaller rule of SMALL_NODES points per dimension
        size = re.ncnt_u.sum(axis=1)
        self.large = size >= large_rows
        self.rules = [(self.z, self.lw, self.kc, False)]
        if self.K > SMALL_NODES and self.large.any():
            self.rules.append((*_rule(SMALL_NODES, self.d), True))
        else:
            self.large[:] = False

    # ------------------------------------------------------------ parameters
    def lam_of(self, S):
        L = np.linalg.inv(S)
        return np.array([L[j, l] for j, l in self.own])

    def _Lam(self, lam):
        L = np.zeros((self.d, self.d))
        for x, (j, l) in zip(lam, self.own):
            L[j, l] = L[l, j] = x
        return L

    def unpack(self, theta):
        P = self.m.P
        alpha = self.re._theta_of(theta, self.fe).copy() if self.fe is not None else None
        mode = self.re._theta_of(theta, self.f)[:, self.re.lv].copy()
        return theta[:P].copy(), alpha, self.lam_of(self.f.S), mode

    def pack(self, b, alpha, mode):
        re, f = self.re, self.f
        parts = [b]
        A = np.zeros((f.G, re.m))
        A[:, re.lv] = mode
        parts.append(np.where(f.free, A, 0.0).ravel())
        if self.fe is not None:
            parts.append(np.where(self.fe.free, alpha, 0.0).ravel())
        return np.concatenate(parts)

    def feasible(self, lam):
        try:
            S = np.linalg.inv(self._Lam(lam))
            return bool(np.linalg.eigvalsh(S).min() >= VAR_MIN
                        and np.linalg.eigvalsh(self._Lam(lam)).min() > 0)
        except np.linalg.LinAlgError:
            return False

    # ------------------------------------------------------------ the pass
    def _chunks(self, rows_per):
        """Chunks of whole groups, about `rows_per` rows each, of the kept
        rows: (X, y, w, u) per chunk."""
        re, m, driver = self.re, self.m, self.driver
        for blk in re.store:
            u_all = blk[re.unit_key]
            keep = re.keep_u[u_all]
            if not keep.all():
                blk = {k: (v[keep] if (isinstance(v, np.ndarray) and len(v) == len(keep))
                           else v) for k, v in blk.items()}
                u_all = u_all[keep]
            n_all = len(u_all)
            if not n_all:
                continue
            starts = np.r_[0, np.flatnonzero(u_all[1:] != u_all[:-1]) + 1]
            cuts = [0]
            while cuts[-1] < n_all:
                j = int(np.searchsorted(starts, cuts[-1] + rows_per, side="left"))
                cuts.append(int(starts[j]) if j < len(starts) else n_all)
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                sl = slice(c0, c1)
                X = _design({"X": blk["X"][sl]}, m.cols, driver.K0)
                w = blk["w"][sl] if blk.get("w") is not None else None
                yield X, blk["y"][sl], (np.ones(c1 - c0) if w is None else w), u_all[sl]

    def _base(self, b, alpha, X, u):
        """Each row's linear predictors (n x J) without the random effect."""
        n, J = len(u), self.J
        E0 = np.zeros((n, J))
        E0[:, 1:] = X @ b.reshape(J - 1, -1).T
        if alpha is not None:
            E0 += alpha[self.fe.of_unit[u]]
        return E0

    @staticmethod
    def _probs(E):
        """Probabilities and log denominators along the last axis (a level
        at -inf, unobserved in its fixed-effect level, has probability 0)."""
        mx = E.max(axis=-1, keepdims=True)
        ex = np.exp(E - mx)
        den = ex.sum(axis=-1, keepdims=True)
        return ex / den, (mx + np.log(den))[..., 0]

    def centre(self, b, alpha, mode):
        """Each group's score and curvature (information) at `mode`, from
        one read: (G x d, G x d x d)."""
        G, d, lv = self.f.G, self.d, self.re.lv
        score, curv = np.zeros((G, d)), np.zeros((G, d, d))
        self.driver.passes += 1
        for X, y, w, u in self._chunks(200_000):
            E = self._base(b, alpha, X, u)
            E[:, 1:] += mode[u]
            Pr, _ = self._probs(E)
            R = -Pr
            R[np.arange(len(y)), y] += 1.0
            p = Pr[:, lv]
            Wm = -p[:, :, None] * p[:, None, :]
            Wm[:, np.arange(d), np.arange(d)] += p
            score += _bysum(u, (R[:, lv] * w[:, None]), G)
            curv += _bysum(u, (Wm * w[:, None, None]).reshape(len(y), -1), G).reshape(G, d, d)
        return score, curv

    def evaluate(self, b, alpha, lam, mode, Rg, logdetR):
        """One read at (b, alpha, lam) with the points at mode + Rg z: the
        log-likelihood, gradient and information (see the module
        docstring), each group's score and curvature at its mode, and its
        posterior mean and covariance. None if a row's probability is 0
        there."""
        m, driver = self.m, self.driver
        driver.passes += 1
        J, d, P, ns = self.J, self.d, m.P, self.ns
        Kx = P // (J - 1)
        Kt = len(self.z)
        own, Ep = self.own, self.Ep
        Lam = self._Lam(lam)
        S = np.linalg.inv(Lam)
        ldL = float(np.linalg.slogdet(Lam)[1])
        coef = np.array([0.5 if j == l else 1.0 for j, l in own])
        jj = np.array([j for j, _ in own])
        ll_ = np.array([l for _, l in own])
        Sv = np.array([S[j, l] for j, l in own])
        fe = self.fe
        Gf = fe.G if fe is not None else 0
        G = self.f.G
        nE = P + ns
        out = SimpleNamespace(
            ll=0.0, gE=np.zeros(nE), JEE=np.zeros((nE, nE)), ga=np.zeros((Gf, J)),
            Jaa=np.zeros((Gf, J, J)), JaE=np.zeros((Gf, J, nE)), n_groups=0,
            score0=np.zeros((G, d)), curv0=np.zeros((G, d, d)), pmean=np.zeros((G, d)),
            pcov=np.zeros((G, d, d)))
        c = SimpleNamespace(Lam=Lam, ldL=ldL, coef=coef, jj=jj, ll_=ll_, Sv=Sv, nE=nE, mode=mode,
                            Rg=Rg, logdetR=logdetR, b=b, alpha=alpha)
        # rows per chunk: the groups' points x (J + d Kx) arrays of about chunk_bytes
        width = 8 * Kt * (J + d * Kx + 4)
        rows_per = max(64, int(driver.chunk_bytes // width))
        for X, y, w, u in self._chunks(rows_per):
            if len(self.rules) == 1:
                self._accumulate(out, c, X, y, w, u, *self.rules[0][:3])
                continue
            for z, lw, kc, large in self.rules:
                sel = self.large[u] == large
                if sel.any():
                    self._accumulate(out, c, X[sel], y[sel], w[sel], u[sel], z, lw, kc)
        # Lambda's information: the prior's part, the same at every point
        # (-d2 of 1/2 log|Lambda| in lam_p, lam_q: 1/2 tr(S E_p S E_q))
        out.JEE[P:, P:] += out.n_groups * 0.5 * np.einsum("ab,pbc,cd,qda->pq", S, Ep, S, Ep)
        return out

    def _accumulate(self, out, c, X, y, w, u, z, lw, kc):
        """One chunk's groups (whole groups), all with the rule (z, lw, kc),
        into `out`."""
        J, d, P = self.J, self.d, self.m.P
        Kx = P // (J - 1)
        Kt = len(z)
        Lam, ldL, coef, jj, ll_, Sv, nE = c.Lam, c.ldL, c.coef, c.jj, c.ll_, c.Sv, c.nE
        mode, Rg, logdetR, b, alpha = c.mode, c.Rg, c.logdetR, c.b, c.alpha
        fe = self.fe
        Gf = fe.G if fe is not None else 0
        n = len(y)
        st = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
        gu = u[st]
        gc = len(st)
        A = mode[gu][:, None, :] + np.swapaxes(Rg[gu] @ z.T, 1, 2)       # gc x Kt x d
        quad = ((A @ Lam) * A).sum(axis=2)
        vx = lw[None, :] + logdetR[gu][:, None] + 0.5 * ldL - 0.5 * quad - d * HALF_LOG_2PI
        logL, pi = np.empty(gc), np.empty((gc, Kt))
        Pbar, Wbar = np.empty((n, J)), np.empty((n, J, J))
        Ds, Db = np.empty((gc, Kt, J)), np.empty((gc, Kt, J - 1, Kx))
        sc0, cv0 = np.empty((gc, d)), np.empty((gc, d, d))
        _group_kernel(np.ascontiguousarray(self._base(b, alpha, X, u)),
                      np.ascontiguousarray(A), np.ascontiguousarray(vx),
                      y.astype(np.int64), np.ascontiguousarray(w, dtype=np.float64),
                      np.ascontiguousarray(X, dtype=np.float64), np.r_[st, n].astype(np.int64),
                      kc, logL, pi, Pbar, Wbar, Ds, Db, sc0, cv0)
        if not np.all(np.isfinite(logL)):
            return None
        out.ll += float(logL.sum())
        out.n_groups += gc
        Wbar *= -1.0
        Wbar[:, np.arange(J), np.arange(J)] += Pbar
        Rbar = -Pbar
        Rbar[np.arange(n), y] += 1.0
        # b: the weighted rows' gradient and information
        out.gE[:P] += (X.T @ (Rbar[:, 1:] * w[:, None])).T.ravel()
        Ww = Wbar * w[:, None, None]
        for a in range(1, J):
            sa = slice((a - 1) * Kx, a * Kx)
            for c in range(a, J):
                sc = slice((c - 1) * Kx, c * Kx)
                H = (X * Ww[:, a, c][:, None]).T @ X
                out.JEE[sa, sc] += H
                if c != a:
                    out.JEE[sc, sa] += H.T
        # Lambda: each point's score, 1/2 c (S_jl - a_j a_l), averaged
        sl = coef[None, None, :] * (Sv[None, None, :] - A[:, :, jj] * A[:, :, ll_])
        slbar = (pi[:, None, :] @ sl)[:, 0, :]
        out.gE[P:] += slbar.sum(axis=0)
        # the points' deviations from their pi-means: b, Lambda and the levels
        Db = Db.reshape(gc, Kt, P)
        Dl = sl - slbar[:, None, :]
        DE = np.concatenate([Db, Dl], axis=2)                             # gc x Kt x nE
        Wd = (np.sqrt(pi)[:, :, None] * DE).reshape(gc * Kt, nE)
        out.JEE -= Wd.T @ Wd
        if fe is not None:
            cell = fe.of_unit[gu]
            rcell = fe.of_unit[u]
            out.ga += _bysum(rcell, Rbar * w[:, None], Gf)
            out.Jaa += _bysum(rcell, Ww.reshape(n, -1), Gf).reshape(Gf, J, J)
            Bv = (Ww[:, :, 1:, None] * X[:, None, None, :]).reshape(n, J, P)
            out.JaE[:, :, :P] += _bysum(rcell, Bv.reshape(n, -1), Gf).reshape(Gf, J, P)
            Dsp = np.swapaxes(Ds * pi[:, :, None], 1, 2)                   # gc x J x Kt
            out.Jaa -= _bysum(cell, (Dsp @ Ds).reshape(gc, -1), Gf).reshape(Gf, J, J)
            out.JaE -= _bysum(cell, (Dsp @ DE).reshape(gc, -1), Gf).reshape(Gf, J, nE)
        # the mode's score and curvature, and the posterior moments
        out.score0[gu] = sc0
        out.curv0[gu] = cv0
        Ap = np.swapaxes(A * pi[:, :, None], 1, 2)                           # gc x d x Kt
        pm = Ap.sum(axis=2)
        out.pmean[gu] = pm
        out.pcov[gu] = Ap @ A - pm[:, :, None] * pm[:, None, :]

    # ------------------------------------------------------------ the step
    def _system(self, E):
        """The reduced system in (b, lambda) after eliminating the fixed
        effect's free levels: (A, rhs, JaE, Dinv, free)."""
        if self.fe is None:
            return E.JEE, E.gE, None, None, None
        free = self.fe.free
        J = self.J
        Daa = E.Jaa.copy()
        nf = ~free
        Daa[nf[:, :, None] | nf[:, None, :]] = 0.0
        idx = np.arange(J)
        Daa[:, idx, idx] = np.where(nf, 1.0, Daa[:, idx, idx])
        try:
            np.linalg.cholesky(Daa)
        except np.linalg.LinAlgError:
            return None
        Dinv = np.linalg.inv(Daa)
        JaE = np.where(free[:, :, None], E.JaE, 0.0)
        ga = np.where(free, E.ga, 0.0)
        DJ = np.einsum("gab,gbe->gae", Dinv, JaE)
        A = E.JEE - np.einsum("gae,gaf->ef", JaE, DJ)
        rhs = E.gE - np.einsum("gae,ga->e", DJ, ga)
        return A, rhs, JaE, Dinv, free

    def step(self, E):
        sysm = self._system(E)
        if sysm is None:
            return None
        A, rhs, JaE, Dinv, free = sysm
        dE, pd = _Driver.solve(A, rhs)
        if not pd:
            lam = 1e-3 * float(np.max(np.abs(np.diag(A)))) + 1e-8
            for _ in range(30):
                dE, pd = _Driver.solve(A + lam * np.eye(len(A)), rhs)
                if pd:
                    break
                lam *= 10
        if free is not None:
            ga = np.where(free, E.ga, 0.0)
            da = np.einsum("gab,gb->ga", Dinv, ga - np.einsum("gae,e->ga", JaE, dE))
            da = np.where(free, da, 0.0)
        else:
            da, ga = None, None
        dec = float(E.gE @ dE + (np.sum(ga * da) if da is not None else 0.0))
        return dE, da, dec, pd

    def inference(self, E):
        """The coefficients' variance given Sigma, the variance of lambda,
        and d b / d lambda."""
        A = self._system(E)[0]
        P = self.m.P
        Vb_cond = _Driver.inverse(A[:P, :P])
        VE = _Driver.inverse(A)
        return Vb_cond, VE[P:, P:], -Vb_cond @ A[:P, P:]

    # ------------------------------------------------------------ the points
    def _proposal(self, b, alpha, lam, mode, score=None, curv=None):
        """One Newton step for each group's mode, and the new scale (a
        Cholesky factor of the inverse curvature at the mode, prior
        included), from a read's score and curvature at the mode."""
        Lam = self._Lam(lam)
        if score is None:
            score, curv = self.centre(b, alpha, mode)
        C = curv + Lam[None]
        g = score - mode @ Lam
        step = np.linalg.solve(C, g[:, :, None])[:, :, 0]
        V = np.linalg.inv(C)
        Rg = np.linalg.cholesky(0.5 * (V + np.transpose(V, (0, 2, 1))))
        sd = np.sqrt(np.diagonal(V, axis1=1, axis2=2))
        return step, Rg, sd

    # ------------------------------------------------------------ the fit
    def fit(self, theta, tol, maxiter=100, node_tol=1e-4, halvings=30):
        t0 = time.time()
        p_start = self.driver.passes
        b, alpha, lam, mode = self.unpack(theta)
        keep = self.f.keep
        # the start's modes are the joint-mode fit's; two Newton steps of the
        # modes at the start's parameters, then its scales
        for _ in range(2):
            step, Rg, sd = self._proposal(b, alpha, lam, mode)
            mode = np.where(keep[:, None], mode + np.clip(step, -3 * sd, 3 * sd), 0.0)
        _, Rg, sd = self._proposal(b, alpha, lam, mode)
        logdetR = np.log(np.diagonal(Rg, axis1=1, axis2=2)).sum(axis=1)
        converged, dec, it, n_steps = False, np.inf, 0, 0
        path = []
        E = self.evaluate(b, alpha, lam, mode, Rg, logdetR)
        if E is None:
            raise RuntimeError("adaptive quadrature: a probability of 0 at the start")
        for it in range(1, maxiter + 1):
            step_m, Rn, sdn = self._proposal(b, alpha, lam, mode, E.score0, E.curv0)
            moved = float(np.max(np.abs(step_m[keep]) / sd[keep])) if keep.any() else 0.0
            path.append(E.ll)
            st = self.step(E)
            if st is None:
                raise RuntimeError("adaptive quadrature: the fixed effect's information is not "
                                   "positive")
            dE, da, dec, pd = st
            self.log(f"  adaptive round {it}: log-likelihood {E.ll:.12g}, decrement "
                     f"{dec:.3e}, points moved {moved:.2e} (sd)")
            if pd and dec < tol and moved < node_tol:
                converged = True
                break
            if not (pd and dec < tol):
                t = 1.0
                P = self.m.P
                for _ in range(halvings):
                    b_c = b + t * dE[:P]
                    lam_c = lam + t * dE[P:]
                    alpha_c = alpha + t * da if alpha is not None else None
                    if self.feasible(lam_c):
                        Ec = self.evaluate(b_c, alpha_c, lam_c, mode, Rg, logdetR)
                        if Ec is not None and Ec.ll >= E.ll - 1e-10 * abs(E.ll):
                            break
                    t /= 2
                else:
                    warn("adaptive quadrature: the line search failed", self.logger)
                    break
                b, alpha, lam, E = b_c, alpha_c, lam_c, Ec
                n_steps += 1
                step_m, Rn, sdn = self._proposal(b, alpha, lam, mode, E.score0, E.curv0)
            # move the points
            mode = np.where(keep[:, None], mode + np.clip(step_m, -3 * sd, 3 * sd), 0.0)
            Rg, sd = Rn, sdn
            logdetR = np.log(np.diagonal(Rg, axis1=1, axis2=2)).sum(axis=1)
            E = self.evaluate(b, alpha, lam, mode, Rg, logdetR)
            if E is None:
                raise RuntimeError("adaptive quadrature: a probability of 0 after moving the "
                                   "points")
        if not converged:
            warn(f"adaptive quadrature did not converge in {it} rounds (decrement {dec:.2g})",
                 self.logger)
        Vb_cond, V_lam, dbdlam = self.inference(E)
        theta = self.pack(b, alpha, mode)
        ll, moments = E.ll, E
        if len(self.rules) > 1:
            # the value (and the groups' moments) with the full rule for every group:
            # the smaller rule is accurate for the estimates, less so for the value
            rules, self.rules = self.rules, self.rules[:1]
            try:
                Ef = self.evaluate(b, alpha, lam, mode, Rg, logdetR)
            finally:
                self.rules = rules
            if Ef is not None:
                ll, moments = Ef.ll, Ef
        return dict(theta=theta, b=b, alpha=alpha, lam=lam, mode=mode, E=E, moments=moments,
                    Vb_cond=Vb_cond, V_lam=V_lam, dbdlam=dbdlam, ll=ll,
                    info={"adaptive_rounds": it, "adaptive_steps": n_steps,
                          "adaptive_converged": converged, "adaptive_decrement": dec,
                          "adaptive_nodes": self.K, "adaptive_loglik_path": path,
                          "adaptive_passes": self.driver.passes - p_start,
                          "adaptive_seconds": round(time.time() - t0, 3)})


def _bysum(idx, V, G):
    """Sums of the rows of V (n x k) by idx, G x k."""
    out = np.zeros((G, V.shape[1]))
    np.add.at(out, idx, V)
    return out


def _fit_adaptive_mnl(driver, re, fit, re_info, fml, tol, n_nodes, log, logger):
    """`_AdaptiveMNL` from the joint-mode fit `fit`; sets the estimates,
    Sigma, the conditional sds and the coefficients' variance on `re` and
    returns the fit and `re_info` updated as `_fit_one` uses them."""
    from dataclasses import replace

    from .fit import _sigma_summary

    ims = _AdaptiveMNL(re, driver, n_nodes, log, logger)
    log(f"adaptive Gauss-Hermite quadrature, {n_nodes} nodes per dimension "
        f"({len(ims.z) - (n_nodes % 2 == 0):,} points per group)")
    res = ims.fit(fit.theta, tol)
    own, Ep = ims.own, ims.Ep
    S = np.linalg.inv(ims._Lam(res["lam"]))
    S = 0.5 * (S + S.T)
    re.set_covs([S])
    # the conditional sds: the quadrature's (the base level's are 0)
    E = res["E"]
    f = ims.f
    sd = np.zeros((f.G, re.m))
    sd[:, re.lv] = np.sqrt(np.clip(np.diagonal(res["moments"].pcov, axis1=1, axis2=2), 0,
                                   None))
    re.sd_override = [np.where(f.keep[:, None], sd, np.nan)]
    re.adaptive_V = res["Vb_cond"]
    re.method = "adaptive"
    from ..draws import adaptive_mnl_draw_info
    re.draw_info = adaptive_mnl_draw_info(ims, res["E"], res["b"], res["alpha"], res["lam"])
    # from lambda (Sigma^-1's elements) to Sigma's: d Sigma = -Sigma dLambda Sigma
    T = -np.einsum("ab,pbc,cd->pad", S, Ep, S)
    T = np.array([[T[p, j, l] for p in range(len(own))] for j, l in own])     # d Sigma_q / d lam_p
    VS = T @ res["V_lam"] @ T.T
    Ti = np.linalg.inv(T)
    Jb = res["dbdlam"] @ Ti
    by = {f.name: _sigma_summary(S, VS, own)}
    gS = Ti.T @ E.gE[ims.m.P:]
    sigma = {"by_factor": by, "vcov": VS, "J": Jb, "gradient": [float(x) for x in gS]}
    info = res["info"]
    re_info = dict(re_info)
    re_info.update(info)
    re_info["converged"] = info["adaptive_converged"]
    re_info["passes"] += info["adaptive_passes"]
    re_info["newton_steps"] += info["adaptive_steps"]
    re_info["seconds"] = round(re_info["seconds"] + info["adaptive_seconds"], 3)
    re_info["loglik"] = res["ll"]
    re_info["sigma"] = sigma
    fit = replace(fit, theta=res["theta"], converged=info["adaptive_converged"],
                  decrement=info["adaptive_decrement"])
    return fit, re_info
