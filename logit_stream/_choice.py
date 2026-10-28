"""What the fits on choice data with `optimize._GradientDriver` (the mixed
and the nested logit) share: the one formula, the conditional logit as the
starting point, and the diagnostics of the fit."""

from __future__ import annotations

import time

import numpy as np


def plan_one(fml, lf, engine, log, model):
    """The formula's one model, without fixed effects (`model` names the
    estimator in the errors)."""
    from ._frontend import _plan
    log(f"planning {fml!r}")
    specs = _plan(fml, lf, intercept=False, collect=lambda q, what: engine.collect(q, what))
    if len(specs) != 1:
        raise ValueError(f"multiple estimation is not available for the {model}")
    spec = specs[0]
    if spec.fe:
        raise ValueError(f"{spec.fml}: fixed effects are not available for the {model}")
    return spec


def clogit_start(p0, xnames, spec, logger, log):
    """The conditional logit on the store: the starting point (it refuses
    collinear covariates, which these fits do not drop)."""
    from .models import _Conditional
    from .newton import _Driver
    log("starting values: the conditional logit")
    K = len(xnames)
    cdrv = _Driver(p0.store, _Conditional(xnames, K), tol=1e-8, maxiter=50, l2=0.0,
                   chunk_mb=64.0, log=lambda m: None, logger=logger)
    cfit = cdrv.run(np.zeros(K), spec.fml)
    if cfit.dropped:
        raise ValueError(f"{spec.fml}: collinear covariates {cfit.dropped}; drop them (they "
                         "are not dropped for this model)")
    return cfit


def fit_diagnostics(fit, cfit, p0, *, storage, dtype, tol, t0, device=None, drv=None,
                    **more):
    """The diagnostics of a `_GradientFit` started from `cfit`."""
    out = {
        "optimizer": {"iterations": fit.iterations, "bhhh_steps": fit.bhhh_steps,
                      "newton_steps": fit.newton_steps, "quasi_steps": fit.quasi_steps,
                      "converged": fit.converged, "decrement": fit.decrement,
                      "loglik": fit.path, "passes": fit.passes, "seconds": fit.seconds,
                      "tol": tol},
        "start": {"loglik_clogit": float(cfit.ll), "passes": cfit.passes},
        "gradient": [float(v) for v in fit.g],
        "pass0_seconds": p0.seconds, "n_rows": p0.n_obs, "n_raw": p0.n_raw,
        "store": {"storage": storage, "dtype": dtype, "blocks": len(p0.store.blocks),
                  "bytes": int(p0.store.bytes)},
        "disk_peak_gb": round((p0.store.bytes if storage == "disk" else 0) / 1e9, 4),
        "seconds_total": round(time.time() - t0, 2),
        **more,
    }
    if device is not None:
        ds = drv._dstore
        out["device"] = {"name": device.name, "resident": bool(ds and ds.resident),
                         "bytes": int(ds.bytes) if ds else 0}
    return out
