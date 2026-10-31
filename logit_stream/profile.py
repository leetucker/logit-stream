"""`profile=True`: the memory and disk a fit needs, from one scan.

The size of everything a Newton fit holds is set by P, the number of
parameters, and P depends on the model and on the number of outcome levels J,
which only the data can say (a multinomial logit has K x (J - 1) parameters, an
ordered one K + J - 1, a conditional one K). So the scan counts the outcome's
distinct values (and the choice sets, fixed-effect levels and clusters), the
formula gives K, and the arrays are those the drivers allocate: the dense
P x P information (and a per-thread copy of it in the numba kernels), the
absorbed fixed effects' cross blocks (G x J x P), the clusters' score sums, and
the block store.

Not modeled, and said so in the profile's warnings: random intercepts (their
EM holds group-level arrays in memory) and the mixed and nested logit's
per-evaluation workspaces beyond the per-thread P x P accumulators.
"""

from __future__ import annotations

import numba
import numpy as np
import polars as pl
from hdfe_stream.core.profile import (
    GB, Component, Profile, as_components, baseline_gb, combine_profiles, scan_source,
    source_row_group_rows,
)
from hdfe_stream.core.resources import MemoryBudget

__all__ = ["profile_model", "profile_multi_fe", "combine_profiles"]


def _roots(expr):
    try:
        return list(expr.meta.root_names())
    except Exception:                                        # noqa: BLE001
        return []


def profile_model(kind, spec, lf, source, *, weights=None, clusters=None, requests=(),
                  case=None, panel=None, levels=None, storage="disk", dtype="float64",
                  block_mb=64.0, chunk_mb=64.0, offset=None, random=None, n_lambda=0,
                  n_random=0, draws=0, draws_cache_mb=0.0, n_alt=0, firth=False,
                  n_threads=None, budget=None) -> Profile:
    """The `Profile` of one model of `kind` (``"mlogit"``, ``"ologit"``,
    ``"clogit"``, ``"cfelogit"``, ``"nlogit"``, ``"mixlogit"``)."""
    budget = budget if budget is not None else MemoryBudget()
    clusters = clusters or {}
    case = list(case or [])
    panel = list(panel or [])
    fe = list(spec.fe)
    fe_cols = {d: d.split("^") for d in fe}
    K = len(spec.x)
    nt = n_threads or numba.get_num_threads()
    pl_threads = pl.thread_pool_size()
    isz = 4 if dtype == "float32" else 8
    warns: list[str] = []
    if random:
        warns.append("random intercepts: the EM's group-level arrays are not modeled")

    # ------------------------------------------------------------- scan
    raw = list(dict.fromkeys([c for cs in clusters.values() for c in cs]
                             + [c for cs in fe_cols.values() for c in cs] + case + panel))
    present = [spec.y, *[ex for _, ex in spec.x]] + [pl.col(c) for c in raw]
    if weights:
        present.append(pl.col(weights).cast(pl.Float64))
    if offset is not None:
        present.append(offset)
    distinct = {}
    ycols = _roots(spec.y)
    if kind in ("mlogit", "ologit") and levels is None and ycols:
        distinct["__y"] = ycols[:1]
    for name, cols in clusters.items():
        distinct[f"cl:{name}"] = cols
    for d, cols in fe_cols.items():
        distinct[f"fe:{d}"] = cols
    if case:
        distinct["__case"] = case
    if panel:
        distinct["__panel"] = panel
    if kind == "nlogit" and n_alt:
        pass
    stats = scan_source(lf, present=present, distinct=distinct)
    N = max(stats.n_rows, 1)

    # --------------------------------------------------- model dimensions
    J = len(levels) if levels is not None else max(stats.distinct.get("__y", 2), 2)
    if kind == "mlogit":
        P = K * (J - 1)
        m = J
    elif kind == "ologit":
        P = K + J - 1
        m = 1
    elif kind == "nlogit":
        P = K + n_lambda
        m = 1
    elif kind == "mixlogit":
        P = K + n_random
        m = 1
    else:                                                   # clogit, cfelogit
        P = K
        m = 1
    n_cases = stats.distinct.get("__case", 0)
    src = len(set(_roots(spec.y)) | {c for _, e in spec.x for c in _roots(e)} | set(raw)
              | ({weights} if weights else set()))
    ncodes = len(clusters) + len(fe) + (1 if case else 0)
    width_store = K * isz + 8 * (1 + bool(weights) + (offset is not None)) + 4 * ncodes + 8
    comps: list[Component] = []
    add = comps.append

    # -------------------------------------------------------- pass 0 / store
    rows_blk = max(int(block_mb * 1e6 / max(8 * K, 1)), 1)
    add(Component("kernels and library state (first fit)", 0.6e9, 1.0e9,
                  formula="the floor of a small fit: numba kernels, Polars, scipy"))
    add(Component("pass 0: Polars sink buffers", 0.25 * pl_threads * 500_000 * src * 8,
                  pl_threads * 500_000 * src * 8, phases=("pass0",),
                  formula=f"{pl_threads} threads x ~one row group x {src} source columns x 8"))
    add(Component("pass 0: design batches", min(rows_blk, N) * (K + 4) * 8 * 2,
                  min(rows_blk, N) * (K + 4) * 8 * 4, phases=("pass0",),
                  formula=f"min(block_mb rows={rows_blk:,}, N) x (K={K} + 4) x 8 x 2-4"))
    add(Component("pass 0: moments and collection", N * 8 * 2, N * 8 * 6, phases=("pass0",),
                  formula="outcome / choice-set / group bookkeeping, 2-6 x N x 8"))
    add(Component("sample file (narrow source columns)", 0.7 * N * src * 8, N * src * 8,
                  where="disk", formula=f"N x {src} source columns x 8, compressed"))
    add(Component("block store", N * width_store, N * width_store,
                  where="ram" if storage == "memory" else "disk",
                  formula=f"N x ({K} x {isz} + ...) = {width_store} bytes per row "
                          f"(storage={storage!r})"))
    if case:
        add(Component("choice-set index", n_cases * 16, n_cases * 24, phases=("pass0", "newton"),
                      formula="cases x 16-24 bytes"))

    # --------------------------------------------------------- the fit
    add(Component("block in flight", block_mb * 1e6 * 1.5, block_mb * 1e6 * 3,
                  phases=("newton",), formula="1.5-3 x block_mb"))
    add(Component("chunk scratch (scores, linear predictors)", chunk_mb * 1e6 * 2,
                  chunk_mb * 1e6 * 4, phases=("newton",), formula="2-4 x chunk_mb"))
    dense = P * P * 8
    per_thread = kind in ("mlogit", "cfelogit", "clogit", "mixlogit", "nlogit")
    add(Component("information matrix, factors, steps", 4 * dense, 6 * dense,
                  phases=("newton",), formula=f"4-6 x P={P:,}^2 x 8"))
    if per_thread and (kind != "mlogit" or P <= 1200):
        add(Component("per-thread information accumulators", nt * dense * (1 if kind == "mlogit"
                                                                          else 1),
                      nt * dense * 2, phases=("newton",),
                      formula=f"nthreads={nt} x P^2 x 8 (sparse / unit-wise kernels)"))
    if kind in ("mixlogit", "nlogit"):
        nb_ = P + 2
        add(Component("outer-product (BHHH) accumulators", nt * nb_ * dense, nt * nb_ * dense * 2,
                      phases=("newton",), formula=f"nthreads x ~(P+2) x P^2 x 8"))
    if kind == "mixlogit" and draws:
        cache = min(draws_cache_mb * 1e6, (stats.distinct.get("__panel", n_cases) or 1)
                    * draws * max(n_random, 1) * 8)
        add(Component("mixed logit: cached draws", cache, draws_cache_mb * 1e6, phases=("newton",),
                      formula="min(draws_cache_mb, panels x draws x random x 8)"))

    # ------------------------------------------------- absorbed fixed effect
    for d in fe:
        if random:
            continue
        G = max(stats.distinct.get(f"fe:{d}", 1), 1)
        cross = G * m * P * 8
        add(Component(f"fixed effect {d}: level counts", 3 * G * m * 8, 3 * G * m * 8,
                      phases=("newton",), formula=f"3 x G={G:,} x {m} x 8"))
        add(Component(f"fixed effect {d}: cross blocks", cross, 2 * cross, phases=("newton",),
                      formula=f"G x {m} x P={P:,} x 8 (x2 in the check)"))
        if not budget.fits(2 * cross / GB):
            warns.append(f"fixed effect {d}: the cross blocks need {2 * cross / GB:.1f} GB and "
                         f"{budget.headroom_gb():.1f} GB is free; the fit will raise "
                         "MemoryError (use fewer parameters, a coarser effect, or a larger "
                         "max_memory_gb)")

    # --------------------------------------------------------------- clusters
    for name, cols in clusters.items():
        Gc = max(stats.distinct.get(f"cl:{name}", 1), 1)
        add(Component(f"cluster scores: {name}", Gc * P * 8, Gc * P * 8, phases=("newton",),
                      formula=f"G={Gc:,} x P x 8"))
        for r in requests:
            if r.startswith("CRV3:") and r.split(":", 1)[1] == name.split(":", 1)[-1]:
                need = Gc * P * P * 8
                add(Component(f"CRV3 cluster information: {name}", need, need,
                              phases=("newton",), formula=f"G x P^2 x 8"))
                if not budget.fits(need / GB):
                    warns.append(f"CRV3 by {name}: {need / GB:.1f} GB and "
                                 f"{budget.headroom_gb():.1f} GB is free; the fit raises "
                                 "MemoryError (use CRV1, or a larger max_memory_gb)")

    return Profile(
        model=f"{kind}: {spec.fml}", components=as_components(comps), scan=stats,
        threads=int(nt), polars_threads=int(pl_threads), baseline_gb=baseline_gb(),
        memory_limit_gb=budget.limit_gb, memory_reserve=budget.reserve, warnings=warns,
        notes={"P": P, "K": K, "J": J if kind in ("mlogit", "ologit") else None,
               "rows": N, "cases": n_cases or None})


def profile_multi_fe(spec, lf, source, *, weights=None, weights_type="aweights", clusters=None,
                     requests=(), hetero=False, vkey="iid", levels=None, options=None,
                     budget=None, n_threads=None, tol=1e-8, maxiter=50) -> Profile:
    """The profile of the multinomial logit with several absorbed fixed effects,
    which is fitted as a Poisson regression (`fepois_stream`) on the data
    expanded J-fold (see `multi_fe`): one row per observation and outcome level,
    a fixed effect per observation, each fixed effect interacted with the level,
    each covariate with the non-base levels.

    That regression is hdfe-stream's, so this is hdfe-stream's profile of it,
    from counts worked out here: N x J rows, N groups of J rows (the observation
    fixed effect), G x J levels for each fixed effect, K x (J - 1) covariates.
    The scan reads the *original* data once; nothing is expanded."""
    import hdfe_stream as hs

    from .multi_fe import _CNT, _LVL, _OBS, _W

    budget = budget if budget is not None else MemoryBudget()
    clusters = clusters or {}
    xs = [(nm, ex) for nm, ex in spec.x if nm != "Intercept"]
    if not xs:
        raise ValueError(f"{spec.fml}: the multinomial logit needs covariates")
    K = len(xs)
    fe_cols = {d: d.split("^") for d in spec.fe}
    cl_cols = sorted({c for cols in clusters.values() for c in cols})
    fe_src = list(dict.fromkeys(c for cols in fe_cols.values() for c in cols))

    # ---- one scan of the original data: N, the outcome's levels, groups, clusters
    present = [spec.y, *[ex for _, ex in xs]] + [pl.col(c) for c in dict.fromkeys(fe_src + cl_cols)]
    if weights:
        present.append(pl.col(weights).cast(pl.Float64))
    distinct = {}
    ycols = _roots(spec.y)
    if levels is None and ycols:
        distinct["__y"] = ycols[:1]
    for d, cols in fe_cols.items():
        distinct[f"fe:{d}"] = cols
    for cols in clusters.values():
        distinct["^".join(cols)] = cols
    stats = scan_source(lf, present=present, distinct=distinct)
    N = max(stats.n_rows, 1)
    J = len(levels) if levels is not None else max(stats.distinct.get("__y", 2), 2)

    # ---- the expanded problem's counts
    fe_names = [_OBS, *[f"{d}^{_LVL}" for d in spec.fe]]
    xcols = [f"__ls_x{k}_{j}" for j in range(1, J) for k in range(K)]
    exp_distinct = {_OBS: N, "__cells": N * J}
    for d in spec.fe:
        exp_distinct[f"{d}^{_LVL}"] = max(stats.distinct.get(f"fe:{d}", 1), 1) * J
    for cols in clusters.values():
        key = "^".join(cols)
        exp_distinct[key] = stats.distinct.get(key, N)
    from hdfe_stream.core.profile import ScanStats
    exp = ScanStats(n_raw=stats.n_raw * J, n_rows=N * J, distinct=exp_distinct,
                    seconds=stats.seconds)
    n_src = 1 + len(fe_src) + len(cl_cols) + 1 + len(xcols) + 1 + (1 if weights else 0)

    hkey = {"iid": "iid", "hetero": f"CRV1:{_OBS}"}
    to_h = {k: hkey.get(k, k) for k in ["iid", *(["hetero"] if hetero else []), *requests]}
    vcov_h = to_h[vkey]
    extra = [v.split(":", 1)[1] for k, v in to_h.items() if v != vcov_h and k != "iid"]
    opts = {k: v for k, v in (options or {}).items() if k not in ("device", "polars_engine")}
    est = hs.StreamingGLM(
        y=[(_CNT, pl.col(_CNT))], x=[(c, pl.col(c)) for c in xcols], fe=fe_names,
        family="poisson", weights=_W if weights else None, weights_type=weights_type,
        keep=[_LVL, *([_W] if weights else [])], n_threads=n_threads,
        max_memory_gb=budget.limit_gb, memory_reserve=budget.reserve, **opts)
    prof = est._profile_from_stats(exp, source, vcov_h, extra, None, n_src=n_src)
    prof.model = f"mlogit (several fixed effects, as a Poisson regression): {spec.fml}"
    prof.warnings.insert(0, f"fitted by fepois_stream on the data expanded J={J}-fold "
                            f"({N * J:,} rows, {len(xcols)} covariates, one fixed effect per "
                            "observation): this is hdfe-stream's profile of that regression, "
                            "from counts worked out from one scan of the original data")
    prof.notes.update({"route": "poisson", "J": J, "K": K, "rows_original": N,
                       "expanded_rows": N * J, "scan_original": stats.to_dict()})
    return prof
