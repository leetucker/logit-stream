"""Sigma's estimation (`_SigmaMixin`, part of `_RandomEffects`): its updates
between reads -- the fixed point of the Laplace approximation's gradient,
the local optimum (joint mode), quasi-Newton steps -- and its variance
(the approximation's curvature)."""

from __future__ import annotations

import numpy as np

from ..fe import _sym
from .linalg import _sum_by

VAR_FLOOR = 1e-6            # Sigma's eigenvalues at least this (an sd of 0.001): a variance
                            # whose estimate is 0 is approached by EM only geometrically, so it
                            # is held here (a boundary, "singular" fit; see `singular`)




class _SigmaMixin:
    """Sigma's updates and variance, on the passes `_RandomEffects` caches."""

    # ------------------------------------------------------------ Sigma's iterations
    def laplace(self, ll_pen):
        """The Laplace approximation from the penalized log-likelihood and
        the current blocks, with the effects moved to their mode at the
        current b by one Newton step of the data's quadratic model (and the
        log-determinant to first order along it, when the pass has the exact
        terms). At the joint estimates the step is zero; away from them this
        is the approximation as a function of b (and the fixed effect),
        whose gradient is the one the exact Newton steps solve, so it is the
        line search's measure: at the effects as they are, the
        log-determinant's own gradient in them is not zero at the solution,
        and a step towards it can lower that value."""
        D = self._cache
        Lams = [f.Lam for f in self.random]
        levels, F = self._random_factor(D.Haa, D.ga, D, Lams)
        ld = sum(float(x.sum()) for x in F.logdet)
        out = ll_pen - 0.5 * ld
        for f in self.random:
            out -= 0.5 * f.n_eff * np.linalg.slogdet(f.S)[1]
        rhs = [(lev.g - self._theta_of(D.theta, f) @ Lam) * f.free
               for lev, f, Lam in zip(levels, self.random, Lams)]
        d, _, _ = self._apply(F, rhs)
        d = [x * f.free for x, f in zip(d, self.random)]
        out += 0.5 * sum(float(np.sum(x * r)) for x, r in zip(d, rhs))
        if D.T is not None and D.Vu is not None:
            lv = self.lv
            Cfull = np.zeros((self.U, self.m))
            Cfull[:, lv] = np.einsum("ulj,ujlk->uk", D.Vu[:, lv][:, :, lv], D.T) \
                * self.keep_u[:, None]
            out -= 0.5 * sum(float(np.sum(_sum_by(f.of_unit, Cfull, f.G) * x))
                             for f, x in zip(self.random, d))
        return float(out)

    def _local(self):
        D = self._cache
        a0 = [self._theta_of(D.theta, f) for f in self.random]
        return D, a0

    def _sigma_map(self, Ss, laplace, D, a0):
        """One fixed-point step for every Sigma on the data's quadratic model
        at the cached estimates, b and the fixed effect held: the effects'
        modes under Ss by one Newton step of the model, their covariances,
        and the update (with the exact approximation's term if `laplace`)."""
        Lams = [self._embed(S) for S in Ss]
        levels, F = self._random_factor(D.Haa, D.ga, D, Lams)
        rhs = [lev.g - a @ Lam for lev, a, Lam in zip(levels, a0, Lams)]
        d, _, _ = self._apply(F, rhs)
        a = [x + dx * f.free for x, dx, f in zip(a0, d, self.random)]
        Haa = D.Haa
        if laplace and D.T is not None:
            deta = np.zeros((self.U, self.m))
            for f, x, x0 in zip(self.random, a, a0):
                deta += (x - x0)[f.of_unit]
            dH = np.einsum("ujlk,uk->ujl", D.T, deta[:, self.lv])
            Haa = Haa.copy()
            Haa[np.ix_(np.arange(self.U), self.lv, self.lv)] += dH
            levels, F = self._random_factor(Haa, D.ga, D, Lams)
        own, Vu = self._cov(F)
        extra = [0.0] * len(self.random)
        if laplace and D.T is not None:
            lv = self.lv
            Cu = np.einsum("ulj,ujlk->uk", Vu[:, lv][:, :, lv], D.T)
            Cfull = np.zeros((self.U, self.m))
            Cfull[:, lv] = Cu * self.keep_u[:, None]
            C_lev = [_sum_by(f.of_unit, Cfull, f.G) * f.free for f in self.random]
            u, _, _ = self._apply(F, C_lev)
            extra = [_sym(x[f.keep][:, lv].T @ uu[f.keep][:, lv])
                     for f, x, uu in zip(self.random, a, u)]
        out = []
        for f, x, cv, ex in zip(self.random, a, own, extra):
            k, lv = f.keep, self.lv
            S = _sym((x[k][:, lv].T @ x[k][:, lv] + cv[k][:, lv][:, :, lv].sum(axis=0) - ex)
                     / f.n_eff)
            out.append(self._floor(np.diag(np.diag(S)) if self.cov == "diagonal" else S))
        return out

    @staticmethod
    def _floor(S):
        """S with its eigenvalues at least `VAR_FLOOR`."""
        w, V = np.linalg.eigh(S)
        if w.min() >= VAR_FLOOR:
            return S
        return _sym((V * np.maximum(w, VAR_FLOOR)) @ V.T)

    def singular(self, Ss=None):
        """The random groupings whose Sigma (or the one in `Ss`) is at the
        floor in some direction (an estimated variance of zero)."""
        Ss = self.covs if Ss is None else Ss
        return [f.name for f, S in zip(self.random, Ss)
                if np.linalg.eigvalsh(S).min() <= VAR_FLOOR * 1.001]

    def em_local(self, tol, max_steps=2_000, laplace=False):
        """The Sigmas at `_sigma_map`'s fixed point, accelerated by SQUAREM
        (Varadhan and Roland, 2008), with the plain step when an
        extrapolation is not positive definite. Returns (Sigmas, steps)."""
        D, a0 = self._local()
        nr = len(self.random)

        def F(v):
            return self._sigma_map(v, laplace, D, a0)

        def pd(v):
            return all(np.all(np.isfinite(S)) and np.all(np.linalg.eigvalsh(S) > 0) for S in v)

        S0, steps = self.covs, 0
        while steps < max_steps:
            S1 = F(S0)
            S2 = F(S1)
            steps += 2
            r = [b - a for a, b in zip(S0, S1)]
            v = [c - 2 * b + a for a, b, c in zip(S0, S1, S2)]
            nr_, nv = np.sqrt(sum(np.sum(x ** 2) for x in r)), np.sqrt(sum(np.sum(x ** 2)
                                                                          for x in v))
            alpha = min(-nr_ / nv if nv > 0 else -1.0, -1.0)
            S3 = [a - 2 * alpha * x + alpha ** 2 * y for a, x, y in zip(S0, r, v)]
            S_new = F(S3) if pd(S3) else S2
            steps += 1
            if not pd(S_new):
                S_new = S2
            done = all(np.max(np.abs(a - b)) < tol * max(np.max(np.abs(b)), 1e-12)
                       for a, b in zip(S_new, S0))
            S0 = S_new
            if done or nr == 0:
                break
        return S0, steps

    def _local_objective(self, Ss, D, a0):
        """The Laplace approximation (at the joint mode) on the data's
        quadratic model at the cached estimates, b and the fixed effect
        held, with the information held there, and its gradient in each
        Sigma. The modes under Ss are one Newton step of the model,
        a = a0 + M0^-1 (g - Lam a0), M0 = I + Lam, so its log-likelihood
        there is l0 + g' da - da' (g - Lam a) / 2 without multiplying by I.
        (With the information updated to first order in the modes, as the
        exact approximation's updates use, a maximizer can push its
        eigenvalues toward zero and exploit -log|M| / 2: `em_local` is used
        for those instead.)"""
        Lams = [self._embed(S) for S in Ss]
        levels, F0 = self._random_factor(D.Haa, D.ga, D, Lams)
        rhs = [lev.g - x @ Lam for lev, x, Lam in zip(levels, a0, Lams)]
        dd, _, _ = self._apply(F0, rhs)
        da = [x * f.free for x, f in zip(dd, self.random)]
        a = [x + y for x, y in zip(a0, da)]
        val = 0.0
        for f, lev, x, dx, Lam in zip(self.random, levels, a, da, Lams):
            k = f.keep
            # g' da - da' (g - Lam a) / 2 - a' Lam a / 2 (the data's g, not the penalized)
            g = lev.g[k]
            val += float(np.sum(g * dx[k]) - 0.5 * np.sum(dx[k] * (g - x[k] @ Lam))
                         - 0.5 * np.sum((x[k] @ Lam) * x[k]))
        F1 = F0
        if not F0.pd:
            return -np.inf, None
        val -= 0.5 * sum(float(x.sum()) for x in F1.logdet)
        own, Vu = self._cov(F1)
        u = None
        grads = []
        for i, (f, x, cv, S) in enumerate(zip(self.random, a, own, Ss)):
            k, lv = f.keep, self.lv
            xa = x[k][:, lv]
            val -= 0.5 * f.n_eff * np.linalg.slogdet(S)[1]
            dLam = 0.5 * (-xa.T @ xa + f.n_eff * S - cv[k][:, lv][:, :, lv].sum(axis=0))
            if u is not None:
                dLam += 0.5 * _sym(xa.T @ u[i][k][:, lv])
            Li = np.linalg.inv(S)
            grads.append(-Li @ dLam @ Li)
        return val, grads

    def optimize_local(self, max_iter=500):
        """The Sigmas maximizing `_local_objective` within a trust region, by
        L-BFGS-B over each one's Cholesky factor with its diagonal bounded at
        the floor's square root. EM's fixed point is this optimum's, but EM
        approaches a variance near zero sublinearly; a quasi-Newton method in
        the Cholesky factor does not (the approach lme4 takes). The local
        model (b held, the data's quadratic model, the information updated to
        first order) holds only near the estimates it was built at, so each
        entry of a factor moves by at most half the grouping's largest sd (at
        least 0.05) per round; the rounds re-centre it. Returns (Sigmas,
        evaluations, the objective's gain)."""
        from scipy.optimize import minimize
        D, a0 = self._local()
        d, diag = self.d, self.cov == "diagonal"
        tri = [(j, j) for j in range(d)] if diag else [(j, l) for j in range(d)
                                                       for l in range(j + 1)]
        nf = len(tri)

        def unpack(v):
            Ls = []
            for i in range(len(self.random)):
                L = np.zeros((d, d))
                for (j, l), x in zip(tri, v[i * nf:(i + 1) * nf]):
                    L[j, l] = x
                Ls.append(L)
            return Ls

        f0 = [None]

        def fun(v):
            Ls = unpack(v)
            Ss = [L @ L.T for L in Ls]
            try:
                # (the bounds hold each factor's diagonal at the floor's root, not
                # each Sigma's eigenvalues at the floor: with large off-diagonal
                # entries, an eigenvalue far below it makes the penalty too large
                # for the model's value to keep any precision)
                if min(np.linalg.eigvalsh(S).min() for S in Ss) < 0.999 * VAR_FLOOR:
                    raise np.linalg.LinAlgError
                val, gS = self._local_objective(Ss, D, a0)
            except np.linalg.LinAlgError:
                val, gS = -np.inf, None
            if not np.isfinite(val) or gS is None:
                # a step too far (the effects' information near singular): a poor
                # value, so that the line search shortens it
                return (abs(f0[0]) + 1.0) * 10 + 1e6, np.zeros(len(v))
            if f0[0] is None:
                f0[0] = -val
            gv = []
            for L, G in zip(Ls, gS):
                GL = 2 * G @ L                       # d / dL of f(L L'), G symmetric
                gv += [GL[j, l] for j, l in tri]
            return -val, -np.array(gv)

        v0, bounds = [], []
        lo = np.sqrt(VAR_FLOOR)
        for f in self.random:
            L = np.linalg.cholesky(self._floor(f.S))
            r = 0.5 * max(float(np.sqrt(np.diag(f.S)).max()), 0.05)
            for j, l in tri:
                x = L[j, l]
                v0.append(x)
                # sds of 20 on the logit scale are far beyond any useful model
                if j == l:
                    bounds.append((max(lo, x - r), min(20.0, x + r)))
                else:
                    bounds.append((max(-20.0, x - r), min(20.0, x + r)))
        res = minimize(fun, np.array(v0), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-9,
                                "maxls": 50})
        f_start = fun(np.array(v0))[0]
        x = res.x if res.fun <= f_start else np.array(v0)
        gain = max(f_start - min(res.fun, f_start), 0.0)
        return [self._floor(L @ L.T) for L in unpack(x)], int(res.nfev), float(gain)

    # ------------------------------------------------------------ Sigma's variance
    def _sigma_params(self):
        d = self.d
        own = [(j, j) for j in range(d)] + ([] if self.cov == "diagonal" else
                                            [(j, l) for j in range(d) for l in range(j + 1, d)])
        return [(fi, j, l) for fi in range(len(self.random)) for j, l in own]

    def sigma_vcov(self):
        """The variance of all the Sigmas' elements: minus the inverse of the
        Laplace approximation's curvature in them (`_sigma_curvature`).
        Returns (params, vcov, gradient at the estimates, J), J the
        derivative of b's estimates in the elements."""
        params, Hs, g0, Jb = self._sigma_curvature()
        try:
            V = np.linalg.inv(-Hs)
        except np.linalg.LinAlgError:
            V = np.full_like(Hs, np.nan)
        return params, V, g0, Jb

    def sigma_quasi_newton(self, state=None):
        """The Sigmas one quasi-Newton step on from the current ones, on the
        exact Laplace approximation with b and the effects re-solved for
        each Sigma, and the state for the next call. At converged estimates
        the approximation's gradient in the Sigmas is exact; its curvature
        on the data's quadratic model (`_sigma_curvature`) is only roughly
        right when the groups are tiny (their effects, and so the units'
        information, move with the Sigmas beyond first order). So the first
        step uses that curvature (the coupling between b and the Sigmas,
        which makes the fixed-point iteration crawl there, is in it) and
        each later one updates it by BFGS from the gradients at successive
        estimates (`state`: the last call's curvature, Sigmas and
        gradient). Returns (None, None) when the curvature is not negative
        definite or the step leaves Sigmas with an eigenvalue below the
        floor (a variance at zero: the fixed-point iteration handles the
        boundary)."""
        params = self._sigma_params()
        v = np.array([self.random[fi].S[j, l] for fi, j, l in params])
        if state is None:
            _, Hs, g, _ = self._sigma_curvature()
            B = -Hs
        else:
            B, v0, g0 = state
            g = self._sigma_curvature(curvature=False)[2]
            sv, yv = v - v0, g0 - g
            sy = float(sv @ yv)
            if sy > 1e-12 * np.linalg.norm(sv) * np.linalg.norm(yv):
                Bs = B @ sv
                B = B - np.outer(Bs, Bs) / float(sv @ Bs) + np.outer(yv, yv) / sy
        try:
            np.linalg.cholesky(B)
            dv = np.linalg.solve(B, g)
        except np.linalg.LinAlgError:
            return None, None
        if not np.all(np.isfinite(dv)):
            return None, None
        Ss = [S.copy() for S in self.covs]
        for (fi, j, l), x in zip(params, dv):
            Ss[fi][j, l] += x
            if j != l:
                Ss[fi][l, j] += x
        if any(np.linalg.eigvalsh(S).min() < VAR_FLOOR for S in Ss):
            return None, None
        return Ss, (B, v, g)

    def _sigma_curvature(self, curvature=True):
        """The Laplace approximation's gradient and curvature in all the
        Sigmas' elements, the curvature by central differences of the
        analytic gradient; each gradient re-solves b and the effects on the
        data's quadratic model and updates the units' information to first
        order (their derivatives T, from the cached pass, which must have
        them). Returns (params, curvature, gradient at the estimates, J), J
        the derivative of b's estimates in the elements; without
        `curvature`, (params, None, gradient, None)."""
        D = self._cache
        params = self._sigma_params()
        lv, d, M = self.lv, self.d, self.m
        nr = len(self.random)
        a0 = [self._theta_of(D.theta, f) for f in self.random]
        steps = []

        def to_mats(v):
            Ss = [np.zeros((d, d)) for _ in range(nr)]
            for (fi, j, l), x in zip(params, v):
                Ss[fi][j, l] = Ss[fi][l, j] = x
            return Ss

        def grad(v):
            Ss = to_mats(v)
            Lams = [self._embed(S) for S in Ss]
            _, g, H = self._assemble(D, Lams, corrections=D.extra)
            step, _ = self.solve(H, g)
            steps.append(step[:self.model.P])
            th = D.theta + step
            a = [self._theta_of(th, f) for f in self.random]
            deta = np.zeros((self.U, M))
            for f, x, x0 in zip(self.random, a, a0):
                deta += (x - x0)[f.of_unit]
            Haa = D.Haa.copy()
            if D.T is not None:
                Haa[np.ix_(np.arange(self.U), lv, lv)] += np.einsum("ujlk,uk->ujl", D.T,
                                                                    deta[:, lv])
            _, F = self._random_factor(Haa, D.ga, D, Lams)
            own, Vu = self._cov(F)
            u = [np.zeros((f.G, M)) for f in self.random]
            if D.T is not None:
                Cu = np.einsum("ulj,ujlk->uk", Vu[:, lv][:, :, lv], D.T)
                Cfull = np.zeros((self.U, M))
                Cfull[:, lv] = Cu * self.keep_u[:, None]
                u, _, _ = self._apply(F, [_sum_by(f.of_unit, Cfull, f.G) * f.free
                                          for f in self.random])
            out = []
            for f, x, cv, uu, S in zip(self.random, a, own, u, Ss):
                k = f.keep
                xa, ua = x[k][:, lv], uu[k][:, lv]
                dLam = 0.5 * (-xa.T @ xa + f.n_eff * S - cv[k][:, lv][:, :, lv].sum(axis=0)
                              + _sym(xa.T @ ua))
                Li = np.linalg.inv(S)
                dS = -Li @ dLam @ Li
                out.append(dS)
            return np.array([out[fi][j, l] * (1.0 if j == l else 2.0) for fi, j, l in params])

        v0 = np.array([self.random[fi].S[j, l] for fi, j, l in params])
        if not curvature:
            return params, None, grad(v0), None
        h = np.array([1e-4 * np.sqrt(self.random[fi].S[j, j] * self.random[fi].S[l, l])
                      for fi, j, l in params])
        n = len(params)
        Hs, Jb = np.empty((n, n)), np.empty((self.model.P, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = h[i]
            Hs[:, i] = (grad(v0 + e) - grad(v0 - e)) / (2 * h[i])
            Jb[:, i] = (steps[-2] - steps[-1]) / (2 * h[i])
        return params, _sym(Hs), grad(v0), Jb
