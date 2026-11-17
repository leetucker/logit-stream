"""Internals for the adaptive-quadrature tests: a random-intercept ordered
(or binary, as the two-level ordered) logit set up as `_fit_one` sets it up,
with the store grouped by the random grouping, so that the quadrature's
pass can be evaluated at chosen parameters."""

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from hdfe_stream.core.store import PolarsEngine
from hdfe_stream.core.utils import as_lazy

from logit_stream._frontend import _plan
from logit_stream.models import _Ordered
from logit_stream.newton import _Driver
from logit_stream.random_effects import (
    _fit_random,
    _RandomEffects,
    _refine_laplace,
    _sigma_inference,
)
from logit_stream.random_effects.adaptive import _Adaptive
from logit_stream.store import _build


def setup(fml, df, random, *, laplace=True, offset=None, tol=1e-10, random_tol=1e-8):
    """(re, driver, fit): the Laplace fit (or the joint mode's), with the
    rows grouped by `random` and the cache at the estimates."""
    engine = PolarsEngine("cpu")
    lf = as_lazy(df)
    (spec,) = _plan(fml, lf, intercept=False, collect=lambda q, what: engine.collect(q, what))
    outer = list(spec.fe)
    spec = replace(spec, fe=[random, *outer])
    run = Path(tempfile.mkdtemp())
    p0 = _build(lf, spec, "ologit", run, engine, weights=None, clusters={}, case=None,
                levels=None, in_memory=True, sort_fe=random, dtype=np.float64, block_mb=64,
                log=lambda m: None,
                offset=None if offset is None else __import__("polars").col(offset))
    xnames = [nm for nm, _ in spec.x]
    model = _Ordered(xnames, len(xnames), p0.levels)
    keys = {nm: ("fe" if i == 0 else f"fe{i + 1}") for i, nm in enumerate(spec.fe)}
    re = _RandomEffects(model, p0, [random], outer[0] if outer else None, keys,
                        log=lambda m: None)
    driver = _Driver(p0.store, model, tol=tol, maxiter=50, l2=0.0, chunk_mb=64,
                     log=lambda m: None, fe=re)
    fit, _ = _fit_random(driver, re, re.start(), fml, 1e-5 if laplace else random_tol,
                         lambda m: None, None)
    if laplace:
        fit, info = _refine_laplace(driver, re, fit, fml, tol, random_tol, lambda m: None, None)
        fit.laplace_loglik = info["loglik"]
        _sigma_inference(driver, re, fit)
    else:
        re.pass_(driver, fit.theta, True)
    return re, driver, fit


def adaptive(re, driver, n_nodes):
    return _Adaptive(re, driver, n_nodes, lambda m: None)
