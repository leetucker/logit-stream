"""The optimizer of the likelihoods that are not concave (the mixed and the
nested logit): streamed Newton and quasi-Newton steps from an analytic
gradient, with a line search.

A subclass gives `evaluate(thetas, n_bhhh)`: one read of the data for the
log-likelihood and gradient at each of several parameter vectors, and the
units' score outer products at the first `n_bhhh`; and sets `P`, `tol`,
`maxiter`, `switch`, `log` and `passes`.

The log-likelihood is not concave, so a Newton
step is taken only where the Hessian is negative definite; elsewhere the
step is BHHH's (the information estimated by the sum of the units' score
outer products, always positive semi-definite). The Hessian is computed by
central differences of the analytic gradient: 2P + 1 gradients in one read
of the data (the block is in memory; the arithmetic is about 2P gradient
passes). That is expensive when the arithmetic dominates, as it does here,
so after a Newton step the information is updated by BFGS from the change
in the gradient (quasi-Newton steps, one gradient each), and the Hessian is
recomputed only when the quasi-Newton decrement meets the tolerance, to
confirm convergence with the Newton decrement itself, or when a
quasi-Newton step fails. So the final Hessian is always computed at the
estimates: it gives the model-based variance; the robust one is the
sandwich with the units' score outer products. Each step
is followed by a line search: the full step, then shorter ones, three per
read. With ``switch=`` finite, BHHH steps are taken first until their
decrement falls below it; BHHH converges slowly far from the optimum (it
overstates the curvature there), so the default starts with Newton.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .newton import _Driver

# line search: the full step alone (usually taken), then shorter ones a read
# at a time, three per read
_STEPS = np.array([0.5, 0.25, 0.1])
_SHRINK = 0.05                               # further candidates: _STEPS * _SHRINK^k


@dataclass
class _GradientFit:
    theta: np.ndarray
    ll: float
    g: np.ndarray
    H: np.ndarray                   # information (minus the Hessian) at theta
    B: np.ndarray                   # sum of the choosers' score outer products at theta
    iterations: int
    bhhh_steps: int
    newton_steps: int
    converged: bool
    decrement: float
    path: list = field(default_factory=list)
    passes: int = 0
    seconds: float = 0.0
    quasi_steps: int = 0


class _GradientDriver:
    """The optimizer (module docstring); a subclass gives `evaluate`."""

    def evaluate(self, thetas, n_bhhh=0):
        raise NotImplementedError

    def hessian_info(self, theta):
        """The information (minus the Hessian of the simulated
        log-likelihood) by central differences of the analytic gradient,
        with the log-likelihood, gradient and score outer products at
        theta: one read."""
        P = self.P
        h = 6e-6 * np.maximum(1.0, np.abs(theta))
        E = np.diag(h)
        ll, G, B = self.evaluate(np.vstack([theta, theta + E, theta - E]), n_bhhh=1)
        dG = (G[1:P + 1] - G[P + 1:]) / (2 * h[:, None])          # row j: dg / d theta_j
        info = -(dG + dG.T) / 2
        return ll[0], G[0], B[0], info

    def _direction(self, A, g):
        d, ok = _Driver.solve(A, g)
        return d, ok and np.isfinite(d).all()

    def _line_search(self, theta, ll, d):
        """The full step along d if it raises the log-likelihood, else the
        best of shorter candidates that does, tried a read at a time; None
        if none does."""
        lls, G, B = self.evaluate(theta + d, n_bhhh=1)
        if np.isfinite(lls[0]) and lls[0] > ll:
            return theta + d, lls[0], G[0], B[0], 1.0
        steps = _STEPS
        for _ in range(4):
            lls, G, B = self.evaluate(theta + steps[:, None] * d[None, :], n_bhhh=len(steps))
            lls = np.where(np.isfinite(lls), lls, -np.inf)
            i = int(np.argmax(lls))
            if lls[i] > ll:
                return theta + steps[i] * d, lls[i], G[i], B[i], steps[i]
            steps = steps * _SHRINK
        return None

    def run(self, theta):
        t0 = time.time()
        theta = np.asarray(theta, dtype=float).copy()
        ll, G, B = self.evaluate(theta, n_bhhh=1)
        ll, g, B = float(ll[0]), G[0], B[0]
        path = [ll]
        phase = "bhhh" if np.isfinite(self.switch) else "newton"
        n_bhhh = n_newton = n_quasi = 0
        converged = False
        dec = np.inf
        info = None             # the last finite-difference information
        A = None                # its BFGS update, between finite differences
        for _ in range(self.maxiter):
            if phase == "newton":
                ll, g, B, info = self.hessian_info(theta)
                d, ok = self._direction(info, g)
                if ok and g @ d > 0:
                    dec, A = float(g @ d), info.copy()
                else:
                    d, _ = self._direction(B, g)
                    dec, A = np.inf, None
                self.log(f"  Newton {n_newton + 1}: simulated log-likelihood {ll:.6f}, "
                         f"decrement {dec:.3e}")
                if dec < self.tol:
                    converged = True
                    break
            elif phase == "quasi":
                d, ok = self._direction(A, g)
                dec = float(g @ d) if ok else -1.0
                if dec < 0:
                    phase = "newton"
                    continue
                self.log(f"  quasi-Newton {n_quasi + 1}: simulated log-likelihood {ll:.6f}, "
                         f"decrement {dec:.3e}")
                if dec < self.tol:
                    phase = "newton"            # confirm with the Hessian itself
                    continue
            else:
                d, ok = self._direction(B, g)
                dec = float(g @ d)
                self.log(f"  BHHH {n_bhhh + 1}: simulated log-likelihood {ll:.6f}, "
                         f"decrement {dec:.3e}")
                if dec < self.switch:
                    phase = "newton"
                    continue
            step = self._line_search(theta, ll, d)
            if step is None:
                if phase != "newton":
                    phase = "newton"
                    continue
                # no ascent along the Newton direction: at the optimum up to rounding
                converged = dec < np.sqrt(self.tol)
                break
            new, ll, g_new, B, _ = step
            path.append(float(ll))
            if phase == "bhhh":
                n_bhhh += 1
            elif phase == "newton":
                n_newton += 1
            else:
                n_quasi += 1
            if A is not None and phase != "bhhh":
                # BFGS update of the information: s the step, y the fall in
                # the gradient; skipped unless the curvature is positive
                s, yv = new - theta, g - g_new
                sy = float(s @ yv)
                As = A @ s
                if sy > 1e-10 * np.sqrt((s @ s) * (yv @ yv)):
                    A = A - np.outer(As, As) / float(s @ As) + np.outer(yv, yv) / sy
                phase = "quasi"
            theta, g = new, g_new
        if info is None or not converged:
            ll, g, B, info = self.hessian_info(theta)
        return _GradientFit(theta=theta, ll=float(ll), g=g, H=info, B=B,
                         iterations=n_bhhh + n_newton + n_quasi, bhhh_steps=n_bhhh,
                         newton_steps=n_newton, converged=converged, decrement=float(dec),
                         path=path, passes=self.passes, seconds=round(time.time() - t0, 2),
                         quasi_steps=n_quasi)
