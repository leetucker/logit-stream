"""Adaptive Gauss-Hermite quadrature for one scalar random intercept
(``random_method="adaptive"``): the binary logit (`logit_stream`) and the
ordered logit, with one random grouping, optionally within an absorbed fixed
effect it nests in.

**Why.** Laplace's approximation (``"laplace"``, lme4's ``nAGQ = 1``)
replaces each group's integral over its effect by a Gaussian around the
mode. With few rows per group and a binary or ordinal outcome the integrand
is far from Gaussian, and the approximation's maximum is biased: the
variance most (downwards, by a fifth with about three rows per group and an
sd near 0.8), the coefficients and the fixed effect less. Quadrature with
enough nodes removes the bias (lme4's ``nAGQ > 1``, available there for a
single scalar random intercept too).

**The approximation.** Group g's likelihood is

    L_g = int prod_{i in g} p_i(y_i | eta_i + a) phi(a; 0, tau^2) da
        ~ sum_k s_g omega_k exp(z_k^2 / 2) sqrt(2 pi) f_g(m_g + s_g z_k),

with f_g the integrand, (z_k, omega_k) the K-point Gauss-Hermite rule for
the standard normal, and the nodes centred at the group's conditional mode
m_g and scaled by its conditional sd s_g = (curvature at the mode)^-1/2, as
lme4's are. One node (K = 1) is Laplace's approximation.

**The fit.** Newton's method in the coefficients b (slopes, cutpoints), the
fixed effect's levels and nu = log tau, with the nodes held where they are
during a step: the value, gradient and information of sum_g log L_g are
then exact sums over each group's rows at each of its nodes,

    d log L_g = sum_k pi_gk d log f_gk,
    d2 log L_g = sum_k pi_gk d2 log f_gk + Cov_pi(d log f_g.),

pi_gk the node's share of L_g. The covariance term needs a group's every
row before its weights are known, so the store holds the rows grouped by
the random grouping (each group's rows consecutive in one block), and one
read gives everything. The fixed effect's block of the information is
diagonal (a group is in one of its levels) and is eliminated. Between
steps the nodes move: each read also gives every group's score and
curvature at its centre node (the mode's), from which one Newton step
updates the mode and the scale. Converged when the Newton decrement is
below `tol` and the nodes have stopped moving. With the nodes held during a
step the fixed point is the maximum of the quadrature's value only to the
rule's accuracy; with the default number of nodes that is far below the
estimates' standard errors (checked against brute-force integration in the
tests, and by increasing the nodes).

The start is the joint-mode fit (lme4's ``nAGQ = 0``). The standard errors
are from the inverse of the whole information (b, the fixed effect and nu),
so the coefficients' include tau's uncertainty, as the Laplace fit's do.
"""

from __future__ import annotations

import time

import numpy as np
from hdfe_stream.core.report import warn
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp

from ..newton import _design, _Driver

ADAPTIVE_NODES = 9                  # the default number of nodes (random_nodes)
MIN_NODES = 5                       # fewer: the held nodes' fixed point is biased (1 node
                                    # collapses tau to 0; 3 give ~85% of it with tiny groups)
LOG_FLOOR = float(np.log(1e-3))     # the variance floor of the Laplace fit (an sd of 0.001)
HALF_LOG_2PI = 0.5 * float(np.log(2 * np.pi))


def _rule(K):
    """The K-point Gauss-Hermite rule for the standard normal: nodes z and
    log(omega exp(z^2 / 2) sqrt(2 pi)), and the index of the centre node
    (one with weight zero is added when K is even, for the mode's step)."""
    x, w = hermgauss(K)
    z = np.sqrt(2.0) * x
    lw = np.log(w / np.sqrt(np.pi)) + 0.5 * z * z + HALF_LOG_2PI
    if K % 2 == 0:
        z, lw = np.r_[z, 0.0], np.r_[lw, -np.inf]
    return z, lw, int(np.flatnonzero(z == 0.0)[0])


class _Adaptive:
    """The adaptive quadrature fit of a `_RandomEffects` with one scalar
    random grouping (see the module docstring)."""

    def __init__(self, re, driver, n_nodes, log, logger=None):
        if re.model.kind != "ologit" or re.d != 1 or len(re.random) != 1 or re.kind != "tree":
            raise ValueError("random_method='adaptive' is for one random grouping with a scalar "
                             "intercept: logit_stream or ologit_stream with one random= grouping")
        if not getattr(re.p0, "fe_sorted", False):
            raise ValueError(
                "random_method='adaptive' needs each group's rows in one block of the store, and "
                "they could not be grouped (a group larger than a block, or too little memory "
                "to sort them; see the log): raise block_mb or max_memory_gb")
        if int(n_nodes) != n_nodes or n_nodes < 1:
            raise ValueError("random_nodes must be a positive integer")
        # (the estimator refuses fewer than MIN_NODES; the class allows them for tests)
        self.re, self.driver, self.log, self.logger = re, driver, log, logger
        self.m = re.model
        self.f = re.random[0]
        self.fe = re.fixed_factor
        self.K = int(n_nodes)
        self.z, self.lw, self.k0 = _rule(self.K)
        self._check_grouped()

    # ------------------------------------------------------------ layout
    def _check_grouped(self):
        """Every group's rows are one run of one block."""
        seen = np.zeros(self.re.U, bool)
        for blk in self.re.store.iterate([self.re.unit_key]):
            u = blk[self.re.unit_key]
            if not len(u):
                continue
            st = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
            gu = u[st]
            if np.any(seen[gu]) or len(np.unique(gu)) < len(gu):
                raise RuntimeError("the store's rows are not grouped by the random grouping")
            seen[gu] = True

    def _alpha_free(self):
        return np.flatnonzero(self.fe.free[:, 0]) if self.fe is not None else np.zeros(0, int)

    def unpack(self, theta):
        """(b, the fixed effect's levels (all, G_fe), nu) from the
        `_RandomEffects` parameter vector and Sigma."""
        P = self.m.P
        alpha = self.re._theta_of(theta, self.fe)[:, 0].copy() if self.fe is not None else None
        nu = 0.5 * float(np.log(self.f.S[0, 0]))
        return theta[:P].copy(), alpha, nu

    def pack(self, b, alpha, mode):
        """The `_RandomEffects` parameter vector with these values (the
        random slots hold the modes)."""
        re = self.re
        parts = [b]
        for f in re.factors:
            if f is self.f:
                parts.append(np.where(f.free[:, 0], mode, 0.0))
            else:
                parts.append(np.where(f.free[:, 0], alpha, f.fixed[:, 0]))
        return np.concatenate(parts)

    # ------------------------------------------------------------ the pass
    def _row_scores(self, r, X, y):
        """Each row's gradient in b (slopes, then the free cutpoints)."""
        m = self.m
        C = m.C
        n = len(y)
        cut = np.zeros((n, C))
        up, lo = np.flatnonzero(y < C), np.flatnonzero(y > 0)
        cut[up, y[up]] += r["ga"][up]
        cut[lo, y[lo] - 1] += r["gb"][lo]
        if m.fix_first:
            cut = cut[:, 1:]
        return np.column_stack([r["s"][:, None] * X, cut])

    def evaluate(self, b, alpha, nu, mode, scale):
        """One read at (b, alpha, nu) with the nodes at mode + scale z: the
        log-likelihood, gradient and information (as `SimpleNamespace`
        fields), and each group's score and curvature at its centre node.
        None if a row's probability is 0 there."""
        re, m, driver = self.re, self.m, self.driver
        driver.passes += 1
        P = m.P
        tau2 = float(np.exp(2 * nu))
        z, lw, k0 = self.z, self.lw, self.k0
        Kt = len(z)
        fe = self.fe
        Gf = fe.G if fe is not None else 0
        a_unit = alpha[fe.of_unit] if fe is not None else None
        out = dict(ll=0.0, gb=np.zeros(P), Jbb=np.zeros((P, P)), gnu=0.0, Jnn=0.0,
                   Jbn=np.zeros(P), ga=np.zeros(Gf), Jaa=np.zeros(Gf), Jba=np.zeros((Gf, P)),
                   Jan=np.zeros(Gf), score0=np.zeros(re.U), curv0=np.zeros(re.U),
                   pmean=np.zeros(re.U), psd=np.zeros(re.U))
        width = 8 * (P * (Kt + 2) + Kt * (12 + m.C))
        target = max(256, int(driver.chunk_bytes // width))
        fields = ("s", "ga", "gb", "h", "haa", "hbb", "hab")
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
            # chunks of whole groups, about `target` rows each
            cuts = [0]
            while cuts[-1] < n_all:
                j = int(np.searchsorted(starts, cuts[-1] + target, side="left"))
                cuts.append(int(starts[j]) if j < len(starts) else n_all)
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                sl = slice(c0, c1)
                u = u_all[sl]
                X = _design({"X": blk["X"][sl]}, m.cols, driver.K0)
                y = blk["y"][sl]
                w = blk["w"][sl] if blk.get("w") is not None else None
                wt = np.ones(len(y)) if w is None else w
                base = np.zeros(len(y)) if blk.get("off") is None else blk["off"][sl].copy()
                if a_unit is not None:
                    base += a_unit[u]
                st = np.r_[0, np.flatnonzero(u[1:] != u[:-1]) + 1]
                gu = u[st]
                gc = len(st)
                lg = np.repeat(np.arange(gc), np.diff(np.r_[st, len(u)]))
                A = mode[gu][:, None] + scale[gu][:, None] * z[None, :]       # gc x Kt
                F = np.empty((gc, Kt))
                S = np.empty((gc, Kt))
                Gb = np.empty((gc, Kt, P))
                rs = []
                for k in range(Kt):
                    r = m.rows(b, X, y, w, offset=base + A[lg, k], hessian=True)
                    if "h" not in r:                    # a probability of 0
                        return None
                    F[:, k] = np.add.reduceat(r["lp"] * wt, st)
                    S[:, k] = np.add.reduceat(r["s"], st)
                    Gb[:, k, :] = np.add.reduceat(self._row_scores(r, X, y), st, axis=0)
                    rs.append(r)
                # the node weights
                v = (np.log(scale[gu])[:, None] + lw[None, :] + F
                     - A * A / (2 * tau2) - nu - HALF_LOG_2PI)
                logL = logsumexp(v, axis=1)
                if not np.all(np.isfinite(logL)):
                    return None
                pi = np.exp(v - logL[:, None])
                out["ll"] += float(logL.sum())
                # b: the weighted rows' information, less the nodes' covariance
                pr = pi[lg]
                rbar = {fd: sum(pr[:, k] * rs[k][fd] for k in range(Kt)) for fd in fields}
                rbar["Q"] = sum(pr[:, k, None] * rs[k]["Q"] for k in range(Kt))
                rbar["lp"] = np.zeros(len(y))
                _, g_c, H_c = m._assemble(rbar, X, y, None, True)
                gbar = np.einsum("gk,gkp->gp", pi, Gb)
                D = Gb - gbar[:, None, :]
                W = (np.sqrt(pi)[:, :, None] * D).reshape(gc * Kt, P)
                out["gb"] += g_c
                out["Jbb"] += H_c - W.T @ W
                # nu
                t = A * A / tau2 - 1.0
                tbar = (pi * t).sum(axis=1)
                dt = t - tbar[:, None]
                out["gnu"] += float(tbar.sum())
                out["Jnn"] += float((pi * 2 * A * A / tau2).sum() - (pi * dt * dt).sum())
                out["Jbn"] -= np.einsum("gk,gkp,gk->p", pi, D, dt)
                # the fixed effect: diagonal in its levels
                if fe is not None:
                    cell = fe.of_unit[gu]
                    rcell = fe.of_unit[u]
                    Sbar = (pi * S).sum(axis=1)
                    dS = S - Sbar[:, None]
                    out["ga"] += np.bincount(cell, Sbar, Gf)
                    out["Jaa"] += (np.bincount(rcell, rbar["h"], Gf)
                                   - np.bincount(cell, (pi * dS * dS).sum(axis=1), Gf))
                    Q = rbar["Q"][:, 1:] if m.fix_first else rbar["Q"]
                    Bv = np.column_stack([X * rbar["h"][:, None], Q])
                    np.add.at(out["Jba"], rcell, Bv)
                    np.add.at(out["Jba"], cell, -np.einsum("gk,gkp,gk->gp", pi, D, dS))
                    out["Jan"] -= np.bincount(cell, (pi * dS * dt).sum(axis=1), Gf)
                # the centre node: the mode's score and curvature; the posterior moments
                out["score0"][gu] = S[:, k0]
                out["curv0"][gu] = np.add.reduceat(rs[k0]["h"], st)
                pm = (pi * A).sum(axis=1)
                out["pmean"][gu] = pm
                out["psd"][gu] = np.sqrt(np.maximum((pi * A * A).sum(axis=1) - pm * pm, 0.0))
        from types import SimpleNamespace
        return SimpleNamespace(**out)

    # ------------------------------------------------------------ the step
    def _system(self, E, nu_free):
        """The reduced system in (b, nu) after eliminating the fixed effect's
        free levels: (A, rhs, Jea / Daa, gfree / Daa, idx), A the reduced
        information. With `nu_free` False, nu is held (its rows dropped)."""
        P = self.m.P
        JEE = np.zeros((P + 1, P + 1))
        JEE[:P, :P] = E.Jbb
        JEE[:P, P] = JEE[P, :P] = E.Jbn
        JEE[P, P] = E.Jnn
        gE = np.r_[E.gb, E.gnu]
        keep = np.arange(P + 1) if nu_free else np.arange(P)
        JEE, gE = JEE[np.ix_(keep, keep)], gE[keep]
        idx = self._alpha_free()
        if len(idx):
            Daa = E.Jaa[idx]
            Jea = np.vstack([E.Jba[idx].T, E.Jan[idx][None, :]])[keep]     # (P+1) x n_alpha
            if np.any(Daa <= 0):
                return None
            JEa_D = Jea / Daa[None, :]
            A = JEE - JEa_D @ Jea.T
            rhs = gE - JEa_D @ E.ga[idx]
            return A, rhs, Jea, Daa, idx
        return JEE, gE, None, None, idx

    def step(self, E, nu_free):
        """The Newton step (in b, the free fixed-effect levels, nu) and the
        decrement, or None when the information is not positive definite
        (then a damped step)."""
        sysm = self._system(E, nu_free)
        P = self.m.P
        if sysm is None:
            return None
        A, rhs, Jea, Daa, idx = sysm
        dE, pd = _Driver.solve(A, rhs)
        if not pd:
            lam = 1e-3 * float(np.max(np.abs(np.diag(A)))) + 1e-8
            for _ in range(30):
                dE, pd = _Driver.solve(A + lam * np.eye(len(A)), rhs)
                if pd:
                    break
                lam *= 10
        if len(idx):
            da = (E.ga[idx] - Jea.T @ dE) / Daa
        else:
            da = np.zeros(0)
        db = dE[:P]
        dnu = float(dE[P]) if nu_free else 0.0
        dec = float(np.r_[E.gb, E.gnu if nu_free else 0.0] @ np.r_[db, dnu]
                    + (E.ga[idx] @ da if len(idx) else 0.0))
        return db, da, dnu, dec, pd

    def inference(self, E, nu_free):
        """The coefficients' variance (all of the information inverted), the
        variance of nu, and d b / d nu (for `_fit_one`'s V(b | Sigma) +
        J V(Sigma) J')."""
        A, _, _, _, _ = self._system(E, nu_free)
        P = self.m.P
        Vb_cond = _Driver.inverse(A[:P, :P])
        if not nu_free:
            return Vb_cond, Vb_cond, np.nan, np.zeros(P)
        VE = _Driver.inverse(A)
        dbdnu = -Vb_cond @ A[:P, P]
        return VE[:P, :P], Vb_cond, float(VE[P, P]), dbdnu

    # ------------------------------------------------------------ the fit
    def fit(self, theta, tol, maxiter=100, node_tol=1e-6, halvings=30):
        t0 = time.time()
        p_start = self.driver.passes
        b, alpha, nu = self.unpack(theta)
        f = self.f
        mode = self.re._theta_of(theta, f)[:, 0].copy()
        sd0 = self.re.conditional_sd(theta)[0][:, 0]
        if np.any(~np.isfinite(sd0[f.keep])):
            # (no cache at theta: start the scale at the prior's)
            sd0 = np.full(f.G, float(np.sqrt(f.S[0, 0])))
        scale = np.where(f.keep, np.maximum(sd0, 1e-8), 1.0)
        nu_free = nu > LOG_FLOOR + 1e-9
        idx = self._alpha_free()
        converged, dec, it, n_steps = False, np.inf, 0, 0
        path = []
        E = self.evaluate(b, alpha, nu, mode, scale)
        if E is None:
            raise RuntimeError("adaptive quadrature: a probability of 0 at the start")
        for it in range(1, maxiter + 1):
            # the nodes at the current estimates
            tau2 = float(np.exp(2 * nu))
            curv = E.curv0 + 1.0 / tau2
            step_m = (E.score0 - mode / tau2) / curv
            new_scale = 1.0 / np.sqrt(curv)
            moved = float(np.max(np.abs(step_m[f.keep]) / scale[f.keep])) if f.keep.any() \
                else 0.0
            path.append(E.ll)
            if not nu_free and E.gnu > 0:
                nu_free = True                  # at the floor, but the value rises off it
            st = self.step(E, nu_free)
            if st is None:
                raise RuntimeError("adaptive quadrature: the fixed effect's information is not "
                                   "positive")
            db, da, dnu, dec, pd = st
            self.log(f"  adaptive round {it}: log-likelihood {E.ll:.12g}, decrement {dec:.3e}, "
                     f"nodes moved {moved:.2e} (sd)")
            if pd and dec < tol and moved < node_tol:
                converged = True
                break
            if not (pd and dec < tol):
                # a Newton step with the nodes held, halved until the value does not fall
                t = 1.0
                for _ in range(halvings):
                    nu_c = nu + t * dnu
                    if nu_free and nu_c < LOG_FLOOR:
                        nu_c = LOG_FLOOR
                    alpha_c = alpha.copy() if alpha is not None else None
                    if len(idx):
                        alpha_c[idx] += t * da
                    b_c = b + t * db
                    if self.m.feasible(b_c):
                        Ec = self.evaluate(b_c, alpha_c, nu_c, mode, scale)
                        if Ec is not None and Ec.ll >= E.ll - 1e-10 * abs(E.ll):
                            break
                    t /= 2
                else:
                    warn("adaptive quadrature: the line search failed", self.logger)
                    break
                b, alpha, nu = b_c, alpha_c, nu_c
                if nu_free and nu <= LOG_FLOOR:
                    nu_free = False
                n_steps += 1
                E = Ec
                tau2 = float(np.exp(2 * nu))
                curv = E.curv0 + 1.0 / tau2
                step_m = (E.score0 - mode / tau2) / curv
                new_scale = 1.0 / np.sqrt(curv)
            # move the nodes (one Newton step for each mode, from the last read)
            lim = 3.0 * scale
            mode = np.where(f.keep, mode + np.clip(step_m, -lim, lim), 0.0)
            scale = np.where(f.keep, new_scale, 1.0)
            E = self.evaluate(b, alpha, nu, mode, scale)
            if E is None:
                raise RuntimeError("adaptive quadrature: a probability of 0 after moving the "
                                   "nodes")
        if not converged:
            warn(f"adaptive quadrature did not converge in {it} rounds (decrement {dec:.2g})",
                 self.logger)
        Vb, Vb_cond, var_nu, dbdnu = self.inference(E, nu_free)
        theta = self.pack(b, alpha if alpha is not None else np.zeros(0), mode)
        return dict(theta=theta, b=b, alpha=alpha, nu=nu, mode=mode, scale=scale,
                    pmean=E.pmean, psd=E.psd, E=E, Vb=Vb, Vb_cond=Vb_cond, var_nu=var_nu,
                    dbdnu=dbdnu, nu_free=nu_free, ll=E.ll,
                    info={"adaptive_rounds": it, "adaptive_steps": n_steps,
                          "adaptive_converged": converged, "adaptive_decrement": dec,
                          "adaptive_nodes": self.K, "adaptive_loglik_path": path,
                          "adaptive_passes": self.driver.passes - p_start,
                          "adaptive_seconds": round(time.time() - t0, 3)})


def _fit_adaptive(driver, re, fit, re_info, fml, tol, n_nodes, log, logger):
    """`_Adaptive` from the joint-mode fit `fit`; sets the estimates, Sigma,
    the conditional sds and the coefficients' variance on `re` and returns
    the fit and `re_info` updated as `_fit_one` uses them."""
    from dataclasses import replace

    log(f"adaptive Gauss-Hermite quadrature, {n_nodes} nodes")
    re.method = "joint-mode"
    re.pass_(driver, fit.theta, True)            # the cache at the start, for its sds
    ad = _Adaptive(re, driver, n_nodes, log, logger)
    res = ad.fit(fit.theta, tol)
    nu = res["nu"]
    tau, tau2 = float(np.exp(nu)), float(np.exp(2 * nu))
    re.set_covs([np.array([[max(tau2, 1e-6)]])])
    re.sd_override = [np.where(ad.f.keep, res["scale"], np.nan)[:, None]]
    re.adaptive_V = res["Vb_cond"]
    re.method = "adaptive"
    from ..draws import adaptive_draw_info
    re.draw_info = adaptive_draw_info(ad, res["E"], res["b"], res["alpha"], nu, res["nu_free"])
    se_nu = float(np.sqrt(res["var_nu"])) if np.isfinite(res["var_nu"]) else np.nan
    z = 1.959963984540054
    dS = 2 * tau2                                  # d Sigma / d nu
    by = {ad.f.name: {
        "Sigma_se": np.array([[dS * se_nu]]), "sd": np.array([tau]),
        "sd_se": np.array([tau * se_nu]),
        "sd_ci": np.array([[tau * np.exp(-z * se_nu), tau * np.exp(z * se_nu)]]),
        "corr": np.array([[1.0]]), "corr_se": np.array([[np.nan]]),
        "corr_ci": np.array([[[np.nan, np.nan]]]),
        "vcov_Sigma": np.array([[(dS * se_nu) ** 2]]), "Sigma_params": [[0, 0]]}}
    free = res["nu_free"] and np.isfinite(se_nu)
    J = (res["dbdnu"] / dS)[:, None] if free else np.zeros((len(res["b"]), 1))
    V = np.array([[(dS * se_nu) ** 2]]) if free else np.zeros((1, 1))
    E = res["E"]
    sigma = {"by_factor": by, "vcov": V, "J": J, "gradient": [float(E.gnu / dS)]}
    info = res["info"]
    re_info = dict(re_info)
    re_info.update(info)
    re_info["converged"] = info["adaptive_converged"]    # (the start's does not matter)
    re_info["passes"] += info["adaptive_passes"]
    re_info["newton_steps"] += info["adaptive_steps"]
    re_info["seconds"] = round(re_info["seconds"] + info["adaptive_seconds"], 3)
    re_info["loglik"] = res["ll"]
    re_info["sigma"] = sigma
    fit = replace(fit, theta=res["theta"], converged=info["adaptive_converged"],
                  decrement=info["adaptive_decrement"])
    return fit, re_info
