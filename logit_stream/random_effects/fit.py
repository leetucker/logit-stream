"""The random intercepts' fit: rounds of Newton's method for b and the
effects at the current Sigmas and of Sigma's updates between reads, at
the joint mode and then at the exact Laplace approximation's stationary
point; and Sigma's standard errors."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
from hdfe_stream.core.report import warn

SIGMA_STEP = 0.25   # the exact-Laplace rounds' largest relative change in Sigma
GAIN_TOL = 1e-5     # with a variance at its floor, converged when re-optimizing Sigma
                    # gains less log-likelihood than this


def _fit_random(driver, re, theta, fml, tol, log, logger, max_outer=200):
    """Random intercepts at their joint mode: Newton's method for b and the
    effects at the current Sigmas, then the Sigmas maximizing the Laplace
    approximation on the data's quadratic model at the estimates, b held
    (`_RandomEffects.optimize_local`, no read); then Newton's method resumes
    from the estimates with the new Sigmas (its first pass reused from the
    cache), so reads are needed only as the estimates move. Converged when
    an update changes every Sigma by less than `tol` relative to its largest
    element, or, with a variance at its floor, gains less than `GAIN_TOL`
    in the approximation (which is nearly flat along a variance near zero,
    where the updates shrink slowly). Returns the last fit and a summary."""
    t0 = time.time()
    p_start = driver.passes
    steps, path, inner_total = 0, [], 0
    converged = False
    change = gain = np.inf
    it = 0
    for it in range(1, max_outer + 1):
        # (no final step: the next round's first pass is this one's, from the cache)
        fit = driver.run(theta, fml, final_step=False)
        steps += fit.iterations
        theta = fit.theta
        lap = re.laplace(fit.ll)
        path.append(lap)
        S_new, k, gain = re.optimize_local()
        inner_total += k
        change = re.cov_change(S_new, re.covs)
        log(f"  Sigma round {it}: Laplace log-likelihood {lap:.12g}, relative change in "
            f"Sigma {change:.2e}, gain {gain:.2e}")
        if fit.converged and (change < tol or (gain < GAIN_TOL and re.singular(S_new))):
            converged = True
            break
        re.set_covs(S_new)
    if not converged:
        warn(f"{fml}: the random intercepts' covariance did not converge in {it} rounds "
             f"(relative change {change:.2g}, gain {gain:.2g})", logger)
    return fit, {"em_rounds": it, "sigma_evaluations": inner_total, "converged": converged,
                 "sigma_change": change, "sigma_gain": gain, "loglik": path[-1],
                 "loglik_path": path, "newton_steps": steps, "passes": driver.passes - p_start,
                 "seconds": round(time.time() - t0, 3), "tol": tol}


def _stalled(changes, tol, rounds=3):
    """The Sigmas' relative change has stopped shrinking within a few
    hundred times the tolerance (the inner fits' precision floor)."""
    if len(changes) <= rounds or changes[-1] > 300 * tol:
        return False
    return min(changes[-rounds:]) >= 0.5 * min(changes[:-rounds])


def _refine_laplace(driver, re, fit, fml, tol, sigma_tol, log, logger, max_outer=100,
                    max_newton=50, halvings=12):
    """From the joint-mode fit to the exact Laplace approximation's
    stationary point, in rounds like the joint mode's: Newton's method for b
    and the effects at the current Sigmas with the approximation's gradient
    (its log-determinant terms, from the effects' covariances at the step
    before), each step one read and halved until the approximation does not
    fall; then the Sigmas at the fixed point of the approximation's
    gradient on the data's quadratic model there (b held; no read;
    `em_local`, with the exact term). The joint-mode fit has brought the
    Sigmas near their optimum, so these updates are small; the fixed-point
    iteration, unlike maximizing the local model's value, is safe with the
    information updated to first order (whose value a maximizer can
    exploit). It converges linearly, at the rate of the missing
    information, which with tiny groups is near 1: once an update shrinks
    by less than half from the round before, or overshoots, the rounds take
    quasi-Newton steps in the Sigmas instead, b and the effects re-solved
    for each (`_RandomEffects.sigma_quasi_newton`: the coupling between b
    and the Sigmas is in their curvature), falling back to the fixed point
    when that curvature is not negative definite or a variance is at its
    floor. Converged
    when a round's first Newton decrement is below `tol` and the update
    changes the Sigmas by less than `sigma_tol` or has stopped shrinking
    within a few hundred times it; or, with a variance at its floor, when
    the round's steps converged and the approximation rose by less than
    `GAIN_TOL` over it. Each round moves the Sigmas a fraction omega of the
    way to the fixed point, and by at most `SIGMA_STEP` relative to their
    largest elements; an update after which the approximation is lower is
    taken half back and omega halved (it grows by a quarter, up to 1, after
    each update that raises it); once an update has been taken back, also
    converged when the approximation rose by less than `GAIN_TOL` over five
    rounds. Returns the fit and a summary."""
    t0 = time.time()
    p_start = driver.passes
    theta = fit.theta
    re.method = "laplace"
    converged, dec, change, inner, path, n_steps = False, np.inf, np.inf, 0, [], 0
    changes, best, omega, overshot, n_newton, qn = [], None, 1.0, False, 0, None
    slow = False
    rnd = 0
    for rnd in range(1, max_outer + 1):
        first = True
        settled = solved = False
        for _ in range(max_newton):
            re.prepare_laplace()
            ll, g, H = re.pass_(driver, theta, True)
            lap = re.laplace(ll)
            step, pd = re.solve(H, g)
            dec = float(g @ step)
            if pd and dec < tol:                    # (below zero only by rounding)
                settled, solved = first, True
                break
            first = False
            t = 1.0
            for _ in range(halvings):
                cand = theta + t * step
                ll_c, g_c, H_c = re.pass_(driver, cand, True)
                if np.isfinite(ll_c):
                    if re.laplace(ll_c) >= lap - 1e-10 * abs(lap):
                        break
                    # or the decrement halved: the value's own error (its effects a
                    # Newton step from their mode, exact to second order) can exceed
                    # its change after a Sigma update with tiny groups
                    st_c, pd_c = re.solve(H_c, g_c)
                    if pd_c and float(g_c @ st_c) < 0.5 * dec:
                        break
                t /= 2
            theta = cand
            n_steps += 1
        if solved and best is not None and lap < best[0] - 1e-9 * abs(best[0]):
            # the last update lowered the approximation (the fixed-point iteration
            # need not raise it; with tiny groups it can overshoot, alternating):
            # back half way towards the best Sigmas, and damp the updates more;
            # when even a small step lowers it, the best is the stationary point
            omega /= 2
            overshot = True
            S_back = [B + 0.5 * (S - B) for B, S in zip(best[1], re.covs)]
            change = re.cov_change(S_back, best[1])
            log(f"  Laplace round {rnd}: log-likelihood {lap:.12g}, below the best; Sigma "
                f"back half way (relative change {change:.2e})")
            if change < sigma_tol:
                re.set_covs(best[1])
                theta = best[2]
                re.prepare_laplace()
                ll, g, H = re.pass_(driver, theta, True)
                step, pd = re.solve(H, g)
                dec = float(g @ step)
                converged = True
                break
            re.set_covs(S_back)
            continue
        best = (lap, list(re.covs), theta.copy())
        path.append(lap)
        # the fixed point of the approximation's gradient, until it is slow (the
        # update shrinking by less than half a round, or overshooting): from then
        # on quasi-Newton steps in the Sigmas, b and the effects re-solved with
        # them, unless their curvature is not negative definite or a variance is
        # at the floor
        slow = slow or overshot or (len(changes) >= 2 and changes[-1] > 0.5 * changes[-2])
        S_fix = None
        if slow:
            S_fix, qn = re.sigma_quasi_newton(qn)
            how = "quasi-Newton"
            n_newton += S_fix is not None
        if S_fix is None:
            S_fix, k = re.em_local(1e-3 * sigma_tol, max_steps=300, laplace=True)
            inner += k
            how = "fixed point"
        change = re.cov_change(S_fix, re.covs)        # the update's size
        # move part way: by the damping, and at most SIGMA_STEP (the local model
        # holds near the estimates it was built at, and with tiny groups its fixed
        # point can lie far outside that); a convex combination, positive definite
        t = min(omega, SIGMA_STEP / change) if change > 0 else 1.0
        S_new = [S0 + t * (S1 - S0) for S0, S1 in zip(re.covs, S_fix)]
        changes.append(change)
        log(f"  Laplace round {rnd}: log-likelihood {lap:.12g}, decrement {dec:.3e}, relative "
            f"change in Sigma {change:.2e} ({how})"
            + (f" (taken {t:.2g} of the way)" if t < 1 else ""))
        # at a variance's floor, the first step after an update is never small (the
        # penalty's inverse is large along it), so flatness asks only that the round's
        # steps converged
        flat = solved and len(path) > 1 and abs(path[-1] - path[-2]) < GAIN_TOL \
            and bool(re.singular())
        # once an update has overshot, the approximation is flat along the
        # direction it alternates in, where the local model's fixed point is only
        # roughly the stationary point: converged when it has stopped rising
        flat = flat or (overshot and len(path) > 5 and path[-1] - min(path[-6:]) < GAIN_TOL)
        if flat or (settled and (change < sigma_tol or _stalled(changes, sigma_tol))):
            converged = True
            break
        re.set_covs(S_new)
        omega = min(1.0, 1.25 * omega)
    if not converged:
        warn(f"{fml}: the exact Laplace approximation did not converge in {rnd} rounds "
             f"(decrement {dec:.2g}, relative change in Sigma {change:.2g})", logger)
    fit = replace(fit, theta=theta, ll=ll, g=g, H=H, converged=converged, decrement=dec)
    return fit, {"laplace_rounds": rnd, "laplace_steps": n_steps, "laplace_converged": converged,
                 "laplace_sigma_evaluations": inner, "laplace_sigma_quasi_newton": n_newton,
                 "laplace_loglik_path": path,
                 "loglik": path[-1], "sigma_change": change,
                 "laplace_passes": driver.passes - p_start,
                 "laplace_seconds": round(time.time() - t0, 3)}


def _sigma_inference(driver, re, fit):
    """The Sigmas' standard errors (`_RandomEffects.sigma_vcov`), and their
    standard deviations' and correlations' (delta method), with Wald
    intervals on the scales that respect their bounds (log sd, Fisher's z).
    Needs a pass at the estimates with the units' information derivatives:
    the exact Laplace refinement's last, or one more read."""
    c = re._cache
    if c is None or c.T is None or not np.array_equal(c.theta, fit.theta):
        re.prepare_laplace()
        re._cache = re._data_pass(driver, fit.theta, extra=True)
    params, V, g0, Jb = re.sigma_vcov()
    z = 1.959963984540054
    by = {}
    for fi, f in enumerate(re.random):
        idx = [i for i, p in enumerate(params) if p[0] == fi]
        own = [(j, l) for _, j, l in (params[i] for i in idx)]
        Vf = V[np.ix_(idx, idx)]
        S, d = f.S, re.d
        se = np.sqrt(np.clip(np.diag(Vf), 0, None))
        S_se = np.full((d, d), np.nan)
        for (j, l), x in zip(own, se):
            S_se[j, l] = S_se[l, j] = x
        sd = np.sqrt(np.diag(S))
        sd_se = np.array([se[own.index((j, j))] / (2 * sd[j]) for j in range(d)])
        corr = S / np.outer(sd, sd)
        corr_se = np.full((d, d), np.nan)
        pos = {p: i for i, p in enumerate(own)}
        for (j, l) in own:
            if j == l:
                continue
            gv = np.zeros(len(own))
            gv[pos[(j, l)]] = 1 / (sd[j] * sd[l])
            gv[pos[(j, j)]] = -corr[j, l] / (2 * S[j, j])
            gv[pos[(l, l)]] = -corr[j, l] / (2 * S[l, l])
            corr_se[j, l] = corr_se[l, j] = float(np.sqrt(max(gv @ Vf @ gv, 0.0)))
        # (at the floor, the interval for log sd reaches to infinity)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sd_ci = np.column_stack([sd * np.exp(-z * sd_se / sd), sd * np.exp(z * sd_se / sd)])
            zc = np.arctanh(np.clip(corr, -1 + 1e-15, 1 - 1e-15))
            zse = corr_se / (1 - corr ** 2)
            corr_ci = np.stack([np.tanh(zc - z * zse), np.tanh(zc + z * zse)], axis=-1)
        by[f.name] = {"Sigma_se": S_se, "sd": sd, "sd_se": sd_se, "sd_ci": sd_ci, "corr": corr,
                      "corr_se": corr_se, "corr_ci": corr_ci, "vcov_Sigma": Vf,
                      "Sigma_params": [list(p) for p in own]}
    return {"by_factor": by, "vcov": V, "J": Jb, "gradient": [float(x) for x in g0]}
