"""The streamed Newton iteration and the variance matrices.

A step is one read of the store: every block adds its log-likelihood,
gradient and information to the running sums (`_pass`), and the step solves
one dense system of the size of the parameter vector. The log-likelihoods
here are concave, so the Newton direction always points uphill; a step that
lowers the log-likelihood (possible far from the optimum) is halved until it
does not, each halving costing a read of the rows for the log-likelihood
alone (no information).

Convergence is judged by the Newton decrement, g' H^-1 g: twice the increase
in the log-likelihood the next step would bring if the model were quadratic,
and the squared length of that step in standard-error units (H^-1 being the
variance of the estimates). The default tolerance, 1e-8, stops when the next
step would move the estimates by about 1e-4 standard errors or less -- the
quantity Stata's ``nrtolerance()`` measures, whose default is 1e-5. Since
Newton's method converges quadratically, a tighter tolerance usually costs at
most one more step.

The linear system is solved after scaling the information to unit diagonal
(Jacobi equilibration), which makes the Cholesky factorization indifferent to
the scale of the covariates -- what standardizing them would achieve, without
touching the data. Newton's method is invariant to rescaling the covariates
in exact arithmetic, so this changes rounding only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
from hdfe_stream.core.report import warn
from hdfe_stream.core.utils import single_blas as _single_blas

_HALVINGS = 12
COLLIN_TOL = 1e-10


def _floored(D):
    top = float(D.max()) if D.size else 1.0
    return np.maximum(D, 1e-8 * top if top > 0 else 1.0)


@dataclass
class _Fit:
    theta: np.ndarray
    ll: float
    g: np.ndarray
    H: np.ndarray                   # information at theta (penalty included)
    ll_start: float
    iterations: int
    converged: bool
    decrement: float
    halvings: int
    path: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    seconds: float = 0.0
    passes: int = 0


def _chunks(blk, rows):
    """Split a block into sub-chunks of about `rows` rows (whole choice sets
    when the block has set boundaries)."""
    n = len(blk["y"])
    if n <= rows:
        yield blk
        return
    starts = blk.get("starts")
    if starts is None:
        cuts = list(range(0, n, rows)) + [n]
    else:
        cuts = [0]
        while cuts[-1] < n:
            i = int(np.searchsorted(starts, cuts[-1] + rows, side="right")) - 1
            nxt = int(starts[i])
            if nxt <= cuts[-1]:
                nxt = int(starts[min(i + 1, len(starts) - 1)])
            cuts.append(nxt)
    for a, b in zip(cuts[:-1], cuts[1:]):
        sub = {k: (v[a:b] if k != "starts" else None) for k, v in blk.items()}
        if starts is not None:
            i0, i1 = np.searchsorted(starts, [a, b])
            sub["starts"] = starts[i0:i1 + 1] - a
        yield sub


def _design(blk, cols, K):
    X = blk["X"]
    return X if len(cols) == K else X[:, cols]


class _Driver:
    """Streamed Newton over a `BlockStore` for one `_Model` (`model`), optionally
    with absorbed fixed effects (`fe`, see fe.py)."""

    def __init__(self, store, model, *, tol, maxiter, l2, chunk_mb, log, logger=None,
                 fe=None, firth=False, device=None):
        self.store, self.model = store, model
        self.device = device            # the GPU's array namespace (device/), or None
        self._dstore = None
        self.tol, self.maxiter, self.l2 = float(tol), int(maxiter), float(l2)
        self.firth = bool(firth)
        self._firth_A = None            # I^-1 at the last evaluation, for the next
        self.chunk_bytes = float(chunk_mb) * 1e6
        self.log, self.logger = log, logger
        self.K0 = len(model.cols)
        self.fe = fe
        self.passes = 0

    # ------------------------------------------------------------ passes
    def _rows(self, scores=False):
        return max(256, int(self.chunk_bytes // (8 * max(self.model.width(scores), 1))))

    def _pass(self, theta, hessian=True):
        """One read of the store: log-likelihood, gradient and (with
        `hessian`) information at `theta`."""
        if self.fe is not None:
            with _single_blas(hessian and self.fe.numba_pass()):
                return self.fe.pass_(self, theta, hessian)
        with _single_blas(hessian and self.model.sparse(None)):
            return self._plain_pass(theta, hessian)

    def _plain_pass(self, theta, hessian, firth_A=None):
        if self.device is not None:
            return self._device_pass(theta, hessian, firth_A)
        self.passes += 1
        m = self.model
        ll, g, H = 0.0, np.zeros(m.P), (np.zeros((m.P, m.P)) if hessian else None)
        rows = self._rows()
        for blk in self.store:
            for ch in _chunks(blk, rows):
                X = _design(ch, m.cols, self.K0)
                l, gg, hh = m.accumulate(theta, X, ch["y"], ch.get("w"), ch.get("starts"),
                                         hessian, offset=ch.get("off"))
                if not np.isfinite(l):
                    return -np.inf, None, None
                ll += l
                g += gg
                if firth_A is not None:
                    g += m.firth_adjustment(theta, X, ch.get("w"), firth_A)
                if hessian:
                    H += hh
        return ll, g, H

    def _device_pass(self, theta, hessian, firth_A=None):
        """`_plain_pass` on the GPU (device/): the blocks' sums formed
        there, and only they brought back."""
        from . import device
        self.passes += 1
        be, m = self.device, self.model
        xp = be.xp
        if self._dstore is None:
            self._dstore = device.DeviceStore(self.store, be)
        th = xp.asarray(theta, dtype=xp.float64)
        ll = 0.0
        g = xp.zeros(m.P)
        H = xp.zeros((m.P, m.P)) if hessian else None
        cols = None if len(m.cols) == self.K0 else xp.asarray(m.cols)
        A = None if firth_A is None else xp.asarray(firth_A, dtype=xp.float64)
        for blk in self._dstore:
            X = blk["X"] if cols is None else blk["X"][:, cols]
            l, gg, hh = device.accumulate(xp, m, th, X, blk["y"], blk.get("w"),
                                          blk.get("starts"), hessian, blk.get("off"))
            ll = ll + l
            g += gg
            if A is not None:
                g += device.firth_adjustment(xp, m, th, X, blk.get("w"), A)
            if hessian:
                H += hh
        ll = float(ll)
        if not np.isfinite(ll):
            return -np.inf, None, None
        return ll, be.asnumpy(g), (be.asnumpy(H) if hessian else None)

    def _pen_mask(self, n):
        pen = self.model.penalized()
        return np.r_[pen, np.zeros(n - len(pen), bool)]

    def _penalty(self, theta):
        if not self.l2:
            return 0.0, 0.0, None
        pen = self._pen_mask(len(theta))
        return (0.5 * self.l2 * float(theta[pen] @ theta[pen]),
                np.where(pen, self.l2 * theta, 0.0), pen)

    def objective(self, theta, hessian=True):
        """Penalized log-likelihood, its gradient and information."""
        if self.firth:
            return self._firth_objective(theta)
        ll, g, H = self._pass(theta, hessian)
        if not np.isfinite(ll) or not self.l2:
            return ll, g, H
        p, gp, pen = self._penalty(theta)
        ll -= p
        g = g - gp
        if H is not None:
            if self.fe is not None:
                H = self.fe.penalize(H, self.l2 * pen[:self.model.P])
            else:
                H = H.copy()
                H[np.diag_indices_from(H)] += np.where(pen, self.l2, 0.0)
        return ll, g, H

    def _firth_objective(self, theta):
        """Firth's penalized log-likelihood l + log|I| / 2 (Jeffreys
        prior), its gradient and the information I (the Newton matrix and,
        inverted, the variance, as brglm2 reports). The gradient's
        adjustment needs I^-1 in every row, so it uses the previous
        evaluation's (one read per step; equal at convergence, where the
        steps vanish); the first evaluation reads twice, to have one. The
        value is exact at `theta`, from the I the same read forms."""
        with _single_blas(self.model.sparse(None)):
            if self._firth_A is None:
                _, _, H0 = self._plain_pass(theta, True)
                if H0 is None:
                    return -np.inf, None, None
                self._firth_A = self.inverse(H0)
            ll, g, H = self._plain_pass(theta, True, self._firth_A)
        if not np.isfinite(ll):
            return -np.inf, None, None
        sign, logdet = np.linalg.slogdet(H)
        if sign <= 0:
            return -np.inf, None, None
        self._firth_A = self.inverse(H)
        return ll + 0.5 * logdet, g, H

    def _solve(self, H, g):
        return self.fe.solve(H, g) if self.fe is not None else self.solve(H, g)

    def _feasible(self, theta):
        return self.model.feasible(theta[:self.model.P])

    # ------------------------------------------------------------ algebra
    @staticmethod
    def solve(H, g):
        """H^-1 g, equilibrated; and whether H was positive definite."""
        d = np.sqrt(np.abs(np.diag(H)))
        d[d == 0] = 1.0
        Hs = H / d[:, None] / d[None, :]
        try:
            c = sla.cho_factor(Hs, lower=True, check_finite=False)
            return sla.cho_solve(c, g / d, check_finite=False) / d, True
        except np.linalg.LinAlgError:
            return (np.linalg.lstsq(Hs, g / d, rcond=None)[0]) / d, False

    @staticmethod
    def inverse(H):
        """H^-1 (equilibrated); the pseudoinverse if H is singular."""
        d = np.sqrt(np.abs(np.diag(H)))
        d[d == 0] = 1.0
        Hs = H / d[:, None] / d[None, :]
        try:
            c = sla.cho_factor(Hs, lower=True, check_finite=False)
            inv = sla.cho_solve(c, np.eye(len(H)), check_finite=False)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(Hs, hermitian=True)
        inv = inv / d[:, None] / d[None, :]
        return (inv + inv.T) / 2

    @staticmethod
    def collinear(A, n_protected=0, tol=COLLIN_TOL, ref=None):
        """Columns of the Gram matrix A (after its first `n_protected`, which
        are always kept) that are linear combinations of the columns before
        them: an incremental Cholesky factorization of A scaled to unit
        diagonal, dropping a column whose residual variance is below `tol`
        times its variance -- or, with `ref`, times ``ref[i]``: the column's
        variance before the fixed effects were partialled out of A, so that
        a covariate that does not vary within the fixed effect's levels
        (whose variance in A is rounding) is dropped. Returns a keep mask
        over the columns after the protected ones."""
        n = A.shape[0]
        diag = np.diag(A).copy()
        ref = diag if ref is None else np.asarray(ref, dtype=float)
        keep = np.zeros(n, bool)
        L = np.zeros((n, n))
        kept = []
        for i in range(n):
            if not (diag[i] > tol * ref[i]) or not np.isfinite(diag[i]) or ref[i] <= 0:
                if i < n_protected:
                    raise ValueError("a parameter that cannot be dropped has no information")
                continue
            if kept:
                a = A[kept, i] / np.sqrt(diag[kept] * diag[i])
                Lk = L[:len(kept), :len(kept)]
                l = sla.solve_triangular(Lk, a, lower=True, check_finite=False)
                r = 1.0 - l @ l
            else:
                l, r = np.zeros(0), 1.0
            if r * diag[i] > tol * ref[i] or i < n_protected:
                j = len(kept)
                L[j, :j] = l
                L[j, j] = np.sqrt(max(r, tol))
                kept.append(i)
                keep[i] = True
        return keep[n_protected:]

    # ------------------------------------------------------------ the iteration
    def run(self, theta, fml="", final_step=True):
        """Newton's method from `theta`, to a decrement below `tol`. With
        `final_step`, the step that check computed is taken (it costs no
        read, and puts the estimates quadratically closer to the optimum:
        ~1e-4 standard errors away at the default tolerance before it, ~1e-8
        after), while the log-likelihood, gradient and information returned
        stay those of the point where it was computed -- R's glm convention:
        its variance uses the last iteration's weights. The log-likelihood
        at the new point is higher by about half the decrement (< 5e-9)."""
        t0 = time.time()
        m = self.model
        ll, g, H = self.objective(theta)
        if not np.isfinite(ll):
            raise FloatingPointError(f"{fml}: the starting values give a non-finite "
                                     "log-likelihood")
        # collinearity, from the first step's information (net of the fixed
        # effects, if any)
        A, prot = m.x_block(H if self.fe is None else self.fe.reduced(H))
        ref = None if self.fe is None else np.diag(m.x_block(H.tt)[0])
        keep = self.collinear(A, len(prot), ref=ref)
        dropped = []
        if not keep.all():
            dropped = [m.xnames[c] for c, k in zip(m.cols, keep) if not k]
            if not keep.any():
                raise ValueError(f"{fml}: every covariate is collinear"
                                 + (" with the fixed effects" if self.fe is not None else ""))
            warn(f"{fml}: {len(dropped)} variables dropped as collinear: {dropped}",
                 self.logger)
            idx = m.drop(keep)
            if self.fe is not None:
                theta, g, H = self.fe.subset(theta, g, H, idx)
            else:
                theta, g, H = theta[idx], g[idx], H[np.ix_(idx, idx)]
                if self.firth:
                    self._firth_A = self.inverse(H)
        ll_start = ll
        path, converged, dec = [ll], False, np.inf
        refreshed = False
        self.halvings = 0
        it = 0
        prev_dec = np.inf
        for it in range(1, self.maxiter + 1):
            step, pd = self._solve(H, g)
            dec = float(g @ step)
            if self.firth and not refreshed and dec > 0.25 * prev_dec:
                # the decrement is not falling fast: the gradient's lagged I^-1
                # is holding the iteration back (near separation the
                # information changes quickly); recompute it here (one read)
                self._firth_A = self.inverse(H)
                ll, g, H = self.objective(theta)
                refreshed = True
                step, pd = self._solve(H, g)
                dec = float(g @ step)
            prev_dec = dec
            self.log(f"  Newton {it}: log-likelihood {ll:.12g}, decrement {dec:.3e}")
            if pd and 0 <= dec < self.tol:
                converged = True
                if final_step:
                    theta = theta + step
                else:
                    it -= 1                 # this pass checked convergence; no step taken
                break
            trial = theta + step
            ll_new, g_new, H_new = (self.objective(trial) if self._feasible(trial)
                                    else (-np.inf, None, None))
            if not self._better(ll_new, ll):
                trial = self._backtrack(theta, ll, g, H, step)
                if trial is None and self.firth and not refreshed:
                    # Firth's gradient used the previous point's I^-1, which near
                    # separation can differ enough to point nowhere uphill: once,
                    # recompute it with this point's (one read) and try again
                    self._firth_A = self.inverse(H)
                    ll, g, H = self.objective(theta)
                    refreshed = True
                    continue
                if trial is None:
                    warn(f"{fml}: no step along the Newton direction or a damped one "
                         f"increases the log-likelihood ({ll:.12g}); stopping. This "
                         "usually means the estimates are at the optimum to rounding, or "
                         "that the data are (quasi-)separated; see diagnostics['newton']",
                         self.logger)
                    break
                ll_new, g_new, H_new = self.objective(trial)
            theta, ll, g, H = trial, ll_new, g_new, H_new
            refreshed = False
            path.append(ll)
        else:
            step, pd = self._solve(H, g)
            dec = float(g @ step)
            converged = pd and 0 <= dec < self.tol
            if not converged:
                warn(f"{fml}: Newton's method did not converge in maxiter={self.maxiter} "
                     f"steps (decrement {dec:.3g})", self.logger)
        return _Fit(theta=theta, ll=ll, g=g, H=H, ll_start=ll_start, iterations=it,
                    converged=converged, decrement=dec, halvings=self.halvings, path=path,
                    dropped=dropped, seconds=round(time.time() - t0, 3), passes=self.passes)

    def _backtrack(self, theta, ll, g, H, step):
        """A point that increases the log-likelihood when the full Newton
        step does not: the step halved, up to `_HALVINGS` times; then, if
        the information is so ill-conditioned that the Newton direction
        itself is useless (as when the data separate some coefficient, whose
        estimate is then heading for infinity), Levenberg-Marquardt steps,
        solving (H + lambda diag(H)) d = g with lambda growing tenfold, which
        bend the direction toward (scaled) gradient ascent. Each trial costs
        a read of the rows for the log-likelihood alone. None if nothing
        works."""
        t = 1.0
        for _ in range(_HALVINGS):
            t /= 2
            self.halvings += 1
            trial = theta + t * step
            if self._feasible(trial):
                ll_new, _, _ = self.objective(trial, hessian=False)
                if self._better(ll_new, ll):
                    return trial
        for lam in 10.0 ** np.arange(-6, 9):
            d, _ = self._solve(self._damp(H, lam), g)
            for t in (1.0, 0.5, 0.25):
                self.halvings += 1
                trial = theta + t * d
                if self._feasible(trial):
                    ll_new, _, _ = self.objective(trial, hessian=False)
                    if np.isfinite(ll_new) and ll_new > ll:
                        return trial
        return None

    def _damp(self, H, lam):
        """H + lam diag(H), the diagonal floored at 1e-8 of its largest
        entry: a coefficient the data separate can have almost no
        information left, and damping in proportion to that would not
        shorten its step."""
        if self.fe is not None:
            return self.fe.damp(H, lam)
        D = _floored(np.abs(np.diag(H)))
        return H + lam * np.diag(D)

    @staticmethod
    def _better(new, old):
        """A log-likelihood no lower than the current one (to rounding: at the
        optimum a full step can lower it by a hair)."""
        return np.isfinite(new) and (new >= old or abs(new - old) <= 1e-12 * (1 + abs(old)))

    # ------------------------------------------------------------ inference
    def meat(self, theta, H, weights_type, clusters, hetero):
        """One read of the store for the robust variances' middle matrices:
        sum s s' over the units (rows, or choice sets) for "hetero", and for
        each clustering sum over clusters of (sum of s within it)(...)'. The
        clusterings are {name: (column of the store's `cl`, number of
        clusters)}. On the device too (`device.plain_meat`) with device=."""
        if self.fe is not None:
            return self.fe.meat(self, theta, H, weights_type, clusters, hetero)
        if self.device is not None:
            from . import device
            return device.plain_meat(self.device.xp, self, theta, weights_type, clusters,
                                     hetero)
        m = self.model
        P = len(theta)
        M = np.zeros((P, P)) if hetero else None
        U = {name: np.zeros((G, P)) for name, (_, G) in clusters.items()}
        rows = self._rows(scores=True)
        for blk in self.store:
            for ch in _chunks(blk, rows):
                X = _design(ch, m.cols, self.K0)
                S = m.scores(theta, X, ch["y"], ch.get("starts"), offset=ch.get("off"))
                w, cl = ch.get("w"), ch.get("cl")
                if m.unit == "cases":
                    st = ch["starts"][:-1]
                    w = None if w is None else w[st]
                    cl = None if cl is None else cl[st]
                self._add_meat(S, w, cl, weights_type, M, U, clusters)
        return M, U

    def cluster_information(self, theta, col, G):
        """Each cluster's information at `theta`, G x P x P, for the CRV3
        jackknife: one read, each chunk's rows grouped by the cluster codes
        in column `col` of the store's `cl` (rows are the units: not for the
        conditional logit)."""
        m = self.model
        P = len(theta)
        Hg = np.zeros((G, P, P))
        rows = self._rows()
        with _single_blas(self.model.sparse(None)):
            for blk in self.store:
                for ch in _chunks(blk, rows):
                    X = _design(ch, m.cols, self.K0)
                    codes = ch["cl"][:, col]
                    order = np.argsort(codes, kind="stable")
                    cs = codes[order]
                    st = np.r_[np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]), len(cs)]
                    w = ch.get("w")
                    for a, b in zip(st[:-1], st[1:]):
                        idx = order[a:b]
                        off = ch.get("off")
                        _, _, h = m.accumulate(theta, X[idx], ch["y"][idx],
                                               None if w is None else w[idx], None, True,
                                               offset=None if off is None else off[idx])
                        Hg[cs[a]] += h
        self.passes += 1
        return Hg

    @staticmethod
    def _add_meat(S, w, cl, weights_type, M, U, clusters):
        if M is not None:
            if w is None:
                M += S.T @ S
            elif weights_type == "fweights":
                Ss = S * np.sqrt(w)[:, None]
                M += Ss.T @ Ss
            else:
                Ss = S * w[:, None]
                M += Ss.T @ Ss
        if U:
            Sw = S if w is None else S * w[:, None]
            for name, (col, _) in clusters.items():
                codes = cl[:, col]
                order = np.argsort(codes, kind="stable")
                cs = codes[order]
                st = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
                U[name][cs[st]] += np.add.reduceat(Sw[order], st, axis=0)
