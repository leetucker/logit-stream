"""Fitting: from a formula and a source to a result.

For each model of the formula: pass 0 (store.py) writes the sample as
blocks; the Newton driver (newton.py) iterates over them, one read per step;
a final read collects the scores if a robust or clustered variance was asked
for; and the result is assembled (results.py). The run directory holds the
blocks while the model is fitted and is deleted afterwards (results keep no
files).
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from hdfe_stream.core.report import warn
from hdfe_stream.core.resources import MemoryBudget
from hdfe_stream.core.results_base import vcov_key
from hdfe_stream.core.store import PolarsEngine
from hdfe_stream.core.workspace import Run

from ._frontend import _plan
from ._limits import limited
from .models import _CFELogit, _Conditional, _Multinomial, _Ordered
from .newton import _Driver
from .random_effects import _fit_random, _RandomEffects, _refine_laplace, _sigma_inference
from .results import (
    ConditionalFELogitResult,
    ConditionalLogitResult,
    LogitMulti,
    MultinomialLogitResult,
    OrderedLogitResult,
    OrderedProbitResult,
    _ic,
)
from .store import _build

WORKDIR_ENV = "LOGIT_STREAM_WORKDIR"

_KINDS = {"mlogit": "multinomial logit", "ologit": "ordered logit",
          "clogit": "conditional logit", "cfelogit": "conditional fixed-effects logit"}


def _log(verbose, msg, logger=None, level=None):
    """Progress: to `logger` if given, else printed with a timestamp."""
    if not verbose:
        return
    if logger is not None:
        logger.log(level if level is not None else logging.INFO, msg)
    else:
        print(f"[logit {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _base_dir(workdir):
    return Path(workdir or os.environ.get(WORKDIR_ENV) or os.environ.get("HDFE_STREAM_WORKDIR")
                or tempfile.gettempdir())


def _clusterings(vcov, cluster):
    """The vcov to report and every clustering to compute: (key of the
    reported vcov, {"CRV1:var": [columns]} of the one-way clusterings to
    compute, whether "hetero" is needed, {requested key: [(one-way key, sign,
    whether it is one of the request's own terms)]} for the clustered
    requests, CRV3 included). Multi-way "CRV1:a+b" is inclusion-exclusion
    over the intersections of its terms -- a, b and a^b -- each a one-way
    clustering of its own."""
    from itertools import combinations
    key = vcov_key(vcov)
    extra = [cluster] if isinstance(cluster, str) else list(cluster or [])
    keys = [key] + [vcov_key(c if ":" in str(c) or isinstance(c, dict) else {"CRV1": c})
                    for c in extra]
    clusters, hetero, requests = {}, False, {}
    for k in keys:
        if k == "hetero":
            hetero = True
            continue
        if k == "iid":
            continue
        kind, var = k.split(":", 1)
        terms = var.split("+")          # (vcov_key refuses multi-way CRV3)
        parts = []
        for r in range(1, len(terms) + 1):
            for combo in combinations(terms, r):
                cols = list(dict.fromkeys(c for t in combo for c in t.split("^")))
                one = "CRV1:" + "^".join(cols)
                clusters.setdefault(one, cols)
                parts.append((one, 1 if r % 2 else -1, r == 1))
        requests[k] = parts
    return key, clusters, hetero, requests


@limited
def _fit(kind: str, fml: str, data: Any, *, workdir=None, vcov="iid",
         cluster: Sequence[str] | str = (), weights: str | None = None,
         weights_type: str = "aweights", levels: Sequence | None = None,
         case: str | Sequence[str] | None = None, tol: float = 1e-8, maxiter: int = 50,
         l2: float = 0.0, storage: str = "disk", dtype: str = "float64",
         block_mb: float = 64.0, chunk_mb: float = 64.0, polars_engine: str = "cpu",
         keep_intermediates: bool = False, verbose: bool = True, logger=None,
         log_level=None, random: str | None = None, random_cov: str = "unstructured",
         random_tol: float = 1e-6, random_method: str = "laplace", firth: bool = False,
         offset=None, start=None, device: str = "cpu", link: str = "logit",
         altwise: bool = False, poisson_options: dict | None = None,
         profile: bool = False, n_threads: int | None = None,
         max_memory_gb: float | None = None, memory_reserve: float | None = None):
    if poisson_options is not None and kind != "mlogit":
        raise ValueError("poisson_options= is for mlogit_stream with several fixed effects")
    if weights_type not in ("aweights", "fweights"):
        raise ValueError("weights_type must be 'aweights' or 'fweights'")
    if storage not in ("disk", "memory"):
        raise ValueError("storage must be 'disk' or 'memory'")
    if dtype not in ("float64", "float32"):
        raise ValueError("dtype must be 'float64' or 'float32'")
    if not tol > 0:
        raise ValueError("tol must be positive")
    if int(maxiter) < 1:
        raise ValueError("maxiter must be at least 1")
    if l2 < 0:
        raise ValueError("l2 must be nonnegative")
    if link == "probit" and random is not None:
        raise ValueError("random= is for the ordered logit, not the probit (its exact Laplace "
                         "terms are the logistic's)")
    from .device import resolve
    device = resolve(device)                # fails now, without CuPy or a GPU
    if offset is not None:
        if kind == "mlogit":
            raise ValueError("offset= is for the ordered and conditional logit: the multinomial "
                             "logit would need one per outcome level")
        offset = (pl.col(offset) if isinstance(offset, str) else offset).cast(pl.Float64)
    if firth:
        if kind == "ologit":
            raise ValueError(
                "firth= is for the multinomial logit (and, as its two-level case, the binary "
                "logit): the ordered logit is not a canonical-link model, for which "
                "Firth's penalty log|I| / 2 gives finite estimates but does not remove the "
                "O(1/n) bias (Kosmidis 2014 derives the adjustment that does)")
        if kind != "mlogit":
            raise ValueError("firth= (Firth's penalized likelihood) is for mlogit_stream")
        if l2:
            raise ValueError("firth= and l2= are two penalties; use one")
        if random is not None:
            raise ValueError("firth= and random= cannot be combined")
    if kind == "cfelogit":
        if random is not None or firth:
            raise ValueError("cfelogit_stream conditions the unit effects out; random= and "
                             "firth= do not apply")
    if kind == "clogit":
        if case is None:
            raise ValueError("clogit_stream needs case=, the column(s) identifying each "
                             "choice set")
        case = [case] if isinstance(case, str) else list(case)
    elif case is not None:
        raise ValueError("case= is for clogit_stream")
    if altwise and kind != "clogit":
        raise ValueError("altwise= is for clogit_stream")
    if random is not None:
        if kind not in ("mlogit", "ologit"):
            raise ValueError("random= (random intercepts) is for mlogit_stream and ologit_stream")
        random = [random] if isinstance(random, str) else list(random)
        if not random or len(set(random)) < len(random) or any(
                not isinstance(r, str) or "+" in r for r in random):
            raise ValueError("random= is a grouping column (or interaction a^b), or a list of "
                             "distinct ones")
        if random_cov not in ("unstructured", "diagonal"):
            raise ValueError("random_cov must be 'unstructured' or 'diagonal'")
        if random_method not in ("laplace", "joint-mode"):
            raise ValueError("random_method must be 'laplace' or 'joint-mode'")
        if l2:
            raise ValueError("l2= and random= cannot be combined")
    vkey, clusters, hetero, requests = _clusterings(vcov, cluster)
    budget = MemoryBudget(max_memory_gb, memory_reserve)

    def log(msg):
        _log(verbose, msg, logger, log_level)

    from hdfe_stream.core.utils import as_lazy
    lf = as_lazy(data)
    engine = PolarsEngine(polars_engine)
    log(f"planning {fml!r}")
    specs = _plan(fml, lf, intercept=(kind == "mlogit"),
                  collect=lambda q, what: engine.collect(q, what))
    results = []
    for spec in specs:
        if kind == "cfelogit":
            # the formula's one fixed effect is the unit, conditioned out
            if len(spec.fe) != 1:
                raise ValueError(f"{spec.fml}: cfelogit_stream needs the unit as the one fixed "
                                 "effect of the formula, as in 'y ~ x | person_id'")
            case = spec.fe[0].split("^")
            spec = replace(spec, fe=())
        if any(r.startswith("CRV3:") for r in requests) and (
                kind in ("clogit", "cfelogit") or spec.fe or random is not None):
            raise ValueError(f"{spec.fml}: CRV3 is available for the multinomial and ordered "
                             "logit without fixed effects or random intercepts; use CRV1")
        if random is not None:
            if len(spec.fe) > 1:
                raise ValueError(f"{spec.fml}: with random intercepts, one absorbed fixed "
                                 f"effect is available (got {list(spec.fe)})")
            both = [r for r in random if r in spec.fe]
            if both:
                raise ValueError(f"{spec.fml}: {both} cannot be both a fixed and a random "
                                 "effect")
            # the groupings' codes are the store's fe, fe2, ... in this order
            spec = replace(spec, fe=[*random, *spec.fe])
        elif spec.fe:
            if kind == "clogit":
                raise ValueError(f"{spec.fml}: fixed effects are not available for the "
                                 "conditional logit (a characteristic of the choice set "
                                 "cancels; interact it with the alternatives instead)")
            if firth and spec.fe:
                raise ValueError(f"{spec.fml}: firth= is not available with an absorbed "
                                 "fixed effect")
            if len(spec.fe) > 1 and kind == "mlogit":
                if profile:
                    from .profile import profile_multi_fe
                    results.append(profile_multi_fe(
                        spec, lf, data, weights=weights, weights_type=weights_type,
                        clusters=clusters, requests=requests, hetero=hetero, vkey=vkey,
                        levels=levels, options=poisson_options, budget=budget,
                        n_threads=n_threads, tol=tol, maxiter=maxiter))
                    continue
                # several: as a Poisson regression on the data expanded by level
                bad = [nm for nm, v in (("l2=", l2), ("start=", start),
                                        ("storage='memory'", storage == "memory"))
                       if v]
                if bad:
                    raise ValueError(f"{spec.fml}: with several absorbed fixed effects (fitted "
                                     f"by fepois_stream), {', '.join(bad)} are not available")
                from .multi_fe import fit_poisson_route
                # device="cuda" is fepois_stream's: its fixed effects' solve on the GPU
                opts = {"device": "cpu" if device is None else "cuda",
                        "polars_engine": polars_engine, **(poisson_options or {})}
                results.append(fit_poisson_route(
                    spec, lf, engine, vkey=vkey, clusters=clusters, requests=requests,
                    hetero=hetero, weights=weights, weights_type=weights_type, levels=levels,
                    tol=tol, maxiter=maxiter, workdir=workdir,
                    keep_intermediates=keep_intermediates, log=log, logger=logger,
                    options=opts))
                continue
            if len(spec.fe) > 1:
                raise ValueError(f"{spec.fml}: one absorbed fixed effect is available for "
                                 f"{_KINDS[kind]} (got {list(spec.fe)}); put small "
                                 "dimensions in as C(var)")
        if profile:
            from .profile import profile_model
            results.append(profile_model(
                kind, spec, lf, data, weights=weights, clusters=clusters, requests=requests,
                case=case, levels=levels, storage=storage, dtype=dtype, block_mb=block_mb,
                chunk_mb=chunk_mb, offset=offset, random=random, firth=firth,
                n_threads=n_threads, budget=budget))
            continue
        results.append(_fit_one(kind, spec, lf, engine, vkey=vkey, clusters=clusters,
                                requests=requests,
                                hetero=hetero, weights=weights, weights_type=weights_type,
                                levels=levels, case=case, tol=tol, maxiter=maxiter, l2=l2,
                                storage=storage, dtype=dtype, block_mb=block_mb,
                                chunk_mb=chunk_mb, workdir=workdir,
                                keep_intermediates=keep_intermediates, log=log,
                                logger=logger, random=random, random_cov=random_cov,
                                random_tol=random_tol, random_method=random_method,
                                firth=firth, offset=offset, start=start, device=device,
                                link=link, altwise=altwise, budget=budget))
    if profile:
        from .profile import combine_profiles
        return combine_profiles(results)
    return results[0] if len(results) == 1 else LogitMulti(results)


def _fit_one(kind, spec, lf, engine, *, vkey, clusters, requests, hetero, weights, weights_type,
             levels, case, tol, maxiter, l2, storage, dtype, block_mb, chunk_mb, workdir,
             keep_intermediates, log, logger, random=None, random_cov="unstructured",
             random_tol=1e-6, random_method="laplace", firth=False, offset=None, start=None,
             device=None, link="logit", altwise=False, budget=None):
    t0 = time.time()
    log(f"{_KINDS[kind]}: {spec.fml}")
    run = Run(_base_dir(workdir), auto_cleanup=not keep_intermediates)
    disk_peak = 0
    try:
        p0 = _build(lf, spec, kind, run.path, engine, weights=weights,
                    clusters=clusters, case=case, levels=levels,
                    in_memory=(storage == "memory"),
                    dtype=np.float32 if dtype == "float32" else np.float64,
                    block_mb=block_mb, log=log, offset=offset, weights_type=weights_type,
                    altwise=altwise)
        disk_peak = p0.store.bytes if storage == "disk" else 0
        xnames = [nm for nm, _ in spec.x]
        K = len(xnames)
        if kind == "mlogit":
            ic = xnames.index("Intercept") if "Intercept" in xnames else None
            model = _Multinomial(xnames, K, p0.levels, ic)
        elif kind == "ologit":
            model = _Ordered(xnames, K, p0.levels, link=link)
        elif kind == "cfelogit":
            if K == 0:
                raise ValueError(f"{spec.fml}: the conditional fixed-effects logit needs "
                                 "covariates")
            model = _CFELogit(xnames, K)
        else:
            if K == 0:
                raise ValueError(f"{spec.fml}: the conditional logit needs covariates")
            model = _Conditional(xnames, K)
        if K:
            model.density = float(np.mean(p0.xnonzero))
        fe = None
        outer = spec.fe[len(random)] if random is not None and len(spec.fe) > len(random) \
            else None
        if random is not None:
            keys = {nm: ("fe" if i == 0 else f"fe{i + 1}") for i, nm in enumerate(spec.fe)}
            fe = _RandomEffects(model, p0, random, outer, keys, log=log, logger=logger, budget=budget,
                                cov=random_cov)
            if clusters:
                cl = {k: (i, 0) for i, k in enumerate(clusters)}
                bad = [k for k, ok in fe.nested(cl).items() if not ok]
                if bad:
                    raise ValueError(
                        f"{spec.fml}: with random intercepts, a clustering must contain whole "
                        f"groups of the coarsest random grouping ({fe.random[-1].name}), and "
                        f"crossed groupings allow none; {bad} do not")
        elif spec.fe:
            from .fe import _AbsorbedFE
            fe = _AbsorbedFE(model, p0, spec.fe[0], log=log, logger=logger, budget=budget)
        driver = _Driver(p0.store, model, tol=tol, maxiter=maxiter, l2=l2, firth=firth,
                         chunk_mb=chunk_mb, device=device,
                         log=log, logger=logger, fe=fe)
        theta0 = fe.start() if fe is not None else model.start(p0.counts)
        if start is not None:
            theta0 = _warm_start(theta0, start, model, spec.fml)
        re_info = None
        if random is not None:
            # the joint-mode fit only starts the exact approximation's, which
            # carries Sigma on to random_tol
            tol_a = max(random_tol, 1e-5) if random_method == "laplace" else random_tol
            fit, re_info = _fit_random(driver, fe, theta0, spec.fml, tol_a, log, logger)
            if random_method == "laplace":
                fit, info = _refine_laplace(driver, fe, fit, spec.fml, tol, random_tol, log,
                                            logger)
                re_info.update(info)
                re_info["converged"] = re_info["converged"] and info["laplace_converged"]
                re_info["passes"] += info["laplace_passes"]
                re_info["newton_steps"] += info["laplace_steps"]
                re_info["seconds"] = round(re_info["seconds"] + info["laplace_seconds"], 3)
            re_info["sigma"] = _sigma_inference(driver, fe, fit)
            sing = fe.singular()
            re_info["singular"] = sing
            if sing:
                warn(f"{spec.fml}: the random intercepts by {sing} have an estimated variance "
                     "of zero in some direction (held at the floor, an sd of 0.001; a "
                     "boundary or 'singular' fit): those groups differ no more than chance "
                     "allows, beyond the other terms; consider dropping the grouping. Its "
                     "standard errors are not meaningful at the boundary.", logger)
        else:
            fit = driver.run(theta0, spec.fml)
        names = model.names()
        P = len(names)
        N = p0.wsum if weights_type == "fweights" else (
            p0.n_cases if kind == "clogit" else p0.n_obs)
        if kind == "clogit" and weights_type == "fweights" and weights:
            N = p0.wcases
        if kind == "cfelogit":              # the units with variation are the observations
            N = p0.units["wunits"] if weights_type == "fweights" and weights \
                else p0.units["units"]
        Vi = driver.inverse(fit.H)[:P, :P] if fe is None else fe.vcov_beta(fit.H)
        if re_info is not None:
            # with Sigma's uncertainty: V(b | Sigma) + J V(Sigma) J', J = d b / d Sigma
            # (the coefficients' block of the inverse of the whole information)
            sg = re_info["sigma"]
            Vi = Vi + sg["J"] @ sg["vcov"] @ sg["J"].T
        all_vcovs = {"iid": (Vi, int(N - P))}
        n_clusters = {}
        if hetero or clusters:
            log("final pass: scores for the robust variance")
            cl_index = {k: (i, int(p0.cluster_tables[k].height))
                        for i, k in enumerate(clusters)}
            M, U = driver.meat(fit.theta, fit.H, weights_type, cl_index, hetero)
            Hinv = driver.inverse(fit.H) if fe is None else None
            if hetero:
                all_vcovs["hetero"] = (_sandwich(fe, fit, Hinv, M) * N / max(N - 1, 1),
                                       int(N - P))
            # each request: inclusion-exclusion over its one-way terms, every term
            # scaled by G/(G - 1) with the smallest one-way G, df G - 1 (hdfe-stream's
            # and pyfixest's convention; one way, Stata's)
            raw = {k: _sandwich(fe, fit, Hinv, U[k].T @ U[k]) for k in cl_index}
            for req, parts in requests.items():
                if req.startswith("CRV3:"):
                    all_vcovs[req] = _crv3(driver, fit, U, cl_index, parts[0][0], spec, logger,
                                           budget)
                    n_clusters[req.split(":", 1)[1]] = cl_index[parts[0][0]][1]
                    continue
                Vc, df, Gs = _clustered(raw, parts, cl_index, req, spec.fml, logger)
                all_vcovs[req] = (Vc, df)
                n_clusters[req.split(":", 1)[1]] = Gs[0] if len(Gs) == 1 else Gs
        V, df_t = all_vcovs[vkey]
        beta = fit.theta[:P]
        ll = fit.ll + (driver._penalty(fit.theta)[0] if l2 else 0.0)   # unpenalized
        if firth:
            ll = fit.ll - 0.5 * float(np.linalg.slogdet(fit.H)[1])
        if re_info is not None:
            ll = re_info["loglik"]                      # the Laplace approximation
        if kind == "clogit":
            ll0 = _clogit_null(p0)
        elif kind == "cfelogit":
            ll0 = p0.units["loglik_null"]
        else:
            c = p0.counts
            ll0 = float(np.sum(c * np.log(c / c.sum())))
        k_fe = fe.n_params if fe is not None else 0
        aic, bic = _ic(ll, P + k_fe, N)
        # the null with only the fixed effect: free, from its outcome counts
        ll_fe0 = fe.loglik_fe_only() if fe is not None and random is None else np.nan
        diagnostics = {
            "newton": {"iterations": fit.iterations, "converged": fit.converged,
                       "decrement": fit.decrement, "step_halvings": fit.halvings,
                       "loglik": [float(v) for v in fit.path], "passes": fit.passes,
                       "seconds": fit.seconds, "tol": tol},
            "pass0_seconds": p0.seconds,
            "n_rows": p0.n_obs,
            "n_raw": p0.n_raw,
            "store": {"storage": storage, "dtype": dtype, "blocks": len(p0.store.blocks),
                      "bytes": int(p0.store.bytes)},
            "disk_peak_gb": round(disk_peak / 1e9, 4),
            "seconds_total": round(time.time() - t0, 2),
        }
        if engine.fallbacks:
            diagnostics["polars_gpu_fallbacks"] = list(engine.fallbacks)
        if device is not None:
            ds = driver._dstore
            diagnostics["device"] = {"name": device.name, "resident": bool(ds and ds.resident),
                                     "bytes": int(ds.bytes) if ds else 0}
        if re_info is not None:
            diagnostics["newton"].update(iterations=re_info["newton_steps"],
                                         passes=re_info["passes"],
                                         seconds=re_info["seconds"])
            diagnostics["random_effects"] = {k: v for k, v in re_info.items()
                                             if k not in ("loglik", "sigma")}
            diagnostics["random_effects"]["sigma_gradient"] = re_info["sigma"]["gradient"]
        if fe is not None and (random is None or outer is not None):
            diagnostics["separation"] = fe.separation
        _separation_warning(spec, model, beta, np.sqrt(np.diag(all_vcovs["iid"][0])), fit, p0,
                            logger)
        common = dict(fml=spec.fml, depvar=spec.depvar, coefnames=names, beta=beta, vcov=V,
                      vcov_type=vkey, df_t=df_t, n_obs=int(N) if float(N).is_integer() else N,
                      all_vcovs=all_vcovs, n_clusters=n_clusters, collin_vars=fit.dropped,
                      diagnostics=diagnostics, weights=weights, weights_type=weights_type,
                      model=("ordered probit" if link == "probit" else _KINDS[kind]),
                      loglik=ll, loglik_null=ll0,
                      pseudo_r2=float(1 - ll / ll0) if ll0 else np.nan, aic=aic, bic=bic,
                      loglik_fe_null=ll_fe0,
                      pseudo_r2_within=float(1 - ll / ll_fe0) if ll_fe0 else np.nan,
                      firth=bool(firth),
                      n_params=P + k_fe,
                      converged=fit.converged and (re_info is None or re_info["converged"]),
                      fe_names=(list(spec.fe) if random is None
                                else [outer] if outer is not None else []),
                      l2=float(l2),
                      n_levels=({spec.fe[0]: fe.G} if fe is not None and random is None
                                else {outer: fe.fixed_factor.G} if outer is not None else {}),
                      random_effects=({} if random is None else {
                          f.name: {"group": f.name, "n_groups": int(f.n_eff), "Sigma": f.S.copy(),
                                   "cov": random_cov, "method": random_method,
                                   "within": outer, "structure": fe.kind,
                                   "levels": (list(p0.levels[1:]) if kind == "mlogit"
                                              else ["location"]),
                                   **re_info["sigma"]["by_factor"][f.name]}
                          for f in fe.random}))
        xexprs = dict(spec.x)
        rspec = {"source": lf, "xexprs": xexprs,
                 "xcols": [xnames[c] for c in model.cols], "usable": p0.usable,
                 "yexpr": spec.y, "n_raw": p0.n_raw,
                 "fe_cols": {d: d.split("^") for d in spec.fe}, "offset": offset,
                 "incomplete": p0.incomplete}
        if fe is not None:
            rspec.update(fe.result_spec(fit.theta))
        if kind == "mlogit":
            res = MultinomialLogitResult(**common, levels=list(p0.levels), base=p0.levels[0],
                                         _spec=rspec)
        elif kind == "ologit":
            cls = OrderedProbitResult if link == "probit" else OrderedLogitResult
            res = cls(**common, levels=list(p0.levels), _spec=rspec)
        elif kind == "cfelogit":
            res = ConditionalFELogitResult(**common, unit=list(case),
                                           n_units=p0.units["units"],
                                           n_units_dropped=p0.units["dropped"],
                                           n_rows=p0.units["rows"], _spec=rspec)
        else:
            res = ConditionalLogitResult(**common, case=list(case), n_cases=int(p0.n_cases),
                                         _spec=rspec)
        p0.store.delete()
        return res
    finally:
        if not keep_intermediates:
            run.cleanup()


def _warm_start(theta0, start, model, fml):
    """`start=` over the starting coefficients: a mapping {name: value} for
    some of them (the rest keep the null model's), a sequence of all of them
    in the order of the result's `coefnames`, or a result (its coefficients,
    by name). The fixed or random effects keep their own starting values."""
    names = model.names()
    P = len(names)
    if hasattr(start, "coefnames") and hasattr(start, "beta"):
        start = dict(zip(start.coefnames, np.asarray(start.beta)))
    th = np.array(theta0, dtype=float)
    if isinstance(start, dict):
        unknown = [k for k in start if k not in names]
        if unknown:
            raise ValueError(f"{fml}: start= names coefficients the model does not have: "
                             f"{unknown[:5]}")
        for k, v in start.items():
            th[names.index(k)] = float(v)
    else:
        v = np.asarray(start, dtype=float).ravel()
        if len(v) != P:
            raise ValueError(f"{fml}: start= has {len(v)} values for {P} coefficients")
        th[:P] = v
    if not model.feasible(th[:P]):
        raise ValueError(f"{fml}: start= is not a feasible point (non-finite values, or "
                         "cutpoints that do not increase)")
    return th


def _crv3(driver, fit, U, cl_index, key, spec, logger, budget=None):
    """The cluster jackknife (CV3; MacKinnon, Nielsen and Webb 2023):
    (G - 1) / G sum_g d_g d_g', with d_g the change in the estimates from
    dropping cluster g, by one Newton step from them -- the information
    without the cluster's, H - H_g, and its score s_g, d_g = -(H - H_g)^-1
    s_g (exact for a linear model, a close approximation for these: refits
    would cost G full fits). Needs each cluster's information, one more
    read, G x P^2 in memory. Returns (vcov, df)."""
    col, G = cl_index[key]
    P = len(fit.theta)
    need = G * P * P * 8 / 1e9
    budget = budget if budget is not None else MemoryBudget()
    if not budget.fits(need):
        raise MemoryError(f"{spec.fml}: CRV3 by {key.split(':', 1)[1]!r} needs {need:.1f} GB "
                          f"for each cluster's information ({G:,} clusters x {P} parameters "
                          f"squared) and {budget.headroom_gb():.1f} GB is free ({budget}); use "
                          "CRV1, or raise max_memory_gb")
    Hg = driver.cluster_information(fit.theta, col, G)
    H, S = fit.H, U[key]
    D = np.empty((G, P))
    for g in range(G):
        Hm = H - Hg[g]
        try:
            D[g] = -np.linalg.solve(Hm, S[g])
        except np.linalg.LinAlgError:
            D[g] = -np.linalg.lstsq(Hm, S[g], rcond=None)[0]
    if not np.all(np.isfinite(D)):
        warn(f"{spec.fml}: CRV3: some cluster carries all the information on a parameter; "
             "its variance is not defined", logger)
    V = (G - 1) / G * D.T @ D
    return (V + V.T) / 2, int(G - 1)


def _clustered(raw, parts, cl_index, req, fml, logger):
    """A clustered request's vcov from the one-way sandwiches `raw`:
    inclusion-exclusion over its terms, scaled by G/(G - 1) with the smallest
    one-way G, df G - 1 (hdfe-stream's and pyfixest's convention; one way,
    Stata's). Returns (vcov, df, the one-way Gs)."""
    Gs = [cl_index[k][1] for k, _, single in parts if single]
    Gm = min(Gs)
    adj = Gm / (Gm - 1) if Gm > 1 else np.nan
    Vc = sum(sign * raw[k] for k, sign, _ in parts) * adj
    if len(parts) > 1 and np.any(np.diag(Vc) < 0):
        warn(f"{fml}: the {req} vcov has negative variances (possible with multi-way "
             "clustering)", logger)
    return Vc, int(Gm - 1), Gs


def _sandwich(fe, fit, Hinv, M):
    if fe is not None:
        return fe.sandwich(fit.H, M)
    P = len(fit.theta)
    B = Hinv[:P, :P]
    V = B @ M @ B
    return (V + V.T) / 2


def _clogit_null(p0):
    """Equal choice probabilities within each set: -sum over sets of
    w log(set size)."""
    total = 0.0
    for blk in p0.store:
        st = blk["starts"]
        ls = np.log(np.diff(st))
        total -= float(ls.sum() if "w" not in blk else ls @ blk["w"][st[:-1]])
    return total


def _separation_warning(spec, model, beta, se, fit, p0, logger):
    """Warn about estimates the data probably separate.

    A separated direction has an infinite maximum-likelihood estimate, which
    Newton's method approaches linearly (the decrement falls by a constant
    factor per step) rather than quadratically, with the estimate growing
    and its standard error growing faster. So: a coefficient that moves the
    linear predictor by more than 25 per standard deviation of its covariate
    (a factor of e^25 in the odds), or by more than 5 with a standard error
    over 100 per standard deviation -- for an indicator, per unit (a
    coefficient of 25 on an indicator multiplies the odds by e^25); or more
    than 15 Newton steps."""
    if not len(beta):
        return
    sd = np.where(p0.xbinary, 1.0, p0.xsd)
    names = model.names()
    cols = list(model.cols)
    K = len(cols)
    flagged = []
    for i, nm in enumerate(names):
        if model.kind == "mlogit":
            c = cols[i % K]
        elif i < K:
            c = cols[i]
        else:
            continue
        s = sd[c] if sd[c] > 0 else 1.0
        b, e = abs(beta[i]) * s, se[i] * s
        if b > 25 or (b > 5 and e > 100):
            flagged.append(nm)
    if flagged:
        warn(f"{spec.fml}: very large coefficients and standard errors on {flagged[:5]}"
             + (" ..." if len(flagged) > 5 else "")
             + ": the outcome is probably (quasi-)separated by these covariates, so their "
             "estimates and standard errors are not meaningful. Drop or merge the "
             "covariates, or fit with l2= > 0"
             + (" or firth=True" if model.kind == "mlogit" and not spec.fe else "") + ".",
             logger)
    elif fit.converged and fit.iterations > 15:
        warn(f"{spec.fml}: Newton's method took {fit.iterations} steps, which usually means "
             "that some coefficient is (quasi-)separated and drifting toward infinity; "
             "check for very large estimates and standard errors.", logger)
