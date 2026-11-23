"""The binary logit and probit (`logit_stream`, `probit_stream`): fitted by
hdfe-stream's `feglm_stream`, and reported as this package's
`BinaryLogitResult` (or `BinaryProbitResult`), with its predictions, draws
and multiple imputation, for any number of absorbed fixed effects.

With no fixed effect or one, hdfe-stream fits the model by Newton's method on
compact stored rows (`hdfe_stream.glm_fast`), the one fixed effect profiled
out exactly; with more, by IRLS with the fixed effects absorbed. Either way
the result's conventions are hdfe-stream's: the iid vcov has the factor
(N - 1) / (N - K), the coefficients have ``Intercept`` first without a fixed
effect, and a singleton group (whose outcome cannot vary) is dropped as a
separated level.

`draw_parameters`: without a fixed effect the coefficients are drawn; with
one, its levels jointly with them, from the information at the estimates;
with two or more, the coefficients, the fixed effects held at their
estimates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from hdfe_stream.core.formula import numeric_expr
from hdfe_stream.core.utils import as_lazy

# logit-stream's names for hdfe-stream's options
_RENAME = {"tol": "iwls_tol", "maxiter": "iwls_maxiter"}
# options only the fast path (no fixed effect or one) takes
_FAST_ONLY = ("storage", "dtype", "block_mb", "chunk_mb")


def binary_fit(family: str, fml: str, data: Any, workdir, vcov, cluster, options: dict):
    """`feglm_stream`'s fit of a binary model, as this package's result."""
    from hdfe_stream import feglm_stream, glm_fast

    opts = {_RENAME.get(k, k): v for k, v in options.items()}
    # an option passed at its default is as not given
    for k, default in (("bias_correction", False), ("bootcluster", None)):
        if k in opts and opts[k] == default:
            del opts[k]
    offset = opts.get("offset")
    fast = glm_fast.eligible(
        fml, data, vcov=vcov, cluster=cluster, fe_dof=opts.get("fe_dof", "exact"),
        separation_check=opts.get("separation_check", True),
        profile=opts.get("profile", False),
        options={k: v for k, v in opts.items()
                 if k not in ("offset", "iwls_tol", "iwls_maxiter", "separation_check",
                              "fe_dof", "profile")}
        | ({"workdir": workdir} if workdir is not None else {}))
    if not fast:
        for k in _FAST_ONLY:
            opts.pop(k, None)
    res = feglm_stream(fml, data, family, workdir=workdir, vcov=vcov, cluster=cluster, **opts)
    if opts.get("profile"):
        return res
    if hasattr(res, "all_fitted_models"):           # several models (sw(), ...)
        from .results import LogitMulti
        return LogitMulti([wrap(r, family, data, offset)
                           for r in res.all_fitted_models.values()])
    return wrap(res, family, data, offset)


def wrap(res, family: str, data: Any, offset=None):
    """An hdfe-stream binary fit (`HDFEResult`) as a `BinaryLogitResult` or
    `BinaryProbitResult`."""
    from .results import BinaryLogitResult, BinaryProbitResult

    lf = as_lazy(data)
    schema = lf.collect_schema()
    est = res._estimator
    sample = res._sample or {}
    fe_names = list(res.fe_names)
    fe_cols = dict(sample.get("fe_cols") or {d: d.split("^") for d in fe_names})
    xcols = [c for c in res.coefnames if c != "Intercept"]
    xexprs = {c: est.var_exprs[c] for c in xcols}
    depvar = res.depvar
    yexpr = pl.col(depvar) if depvar in schema else numeric_expr(depvar)
    off = None
    if offset is not None:
        off = (pl.col(offset) if isinstance(offset, str) else offset).cast(pl.Float64)
    spec = {"source": lf, "xexprs": xexprs, "xcols": xcols, "usable": sample.get("usable"),
            "yexpr": yexpr, "n_raw": sample.get("n_raw"), "fe_cols": fe_cols, "offset": off,
            "incomplete": None}
    if fe_names:
        spec["fixef"] = {d: res.fixef(d).collect() for d in fe_names}
        spec["effects"] = [(d, [f"fe_{d}"], "fixed") for d in fe_names]
        dropped = {}
        for kind, tables in (sample.get("tables") or {}).items():
            for d, table in tables.items():
                if table is not None and table.height:
                    dropped[d] = (table if d not in dropped
                                  else pl.concat([dropped[d], table]).unique())
        if dropped:
            spec["dropped"] = dropped
    spec["draw"] = _draw_info(res, fe_names)
    k_fe = int(res.k_fe)
    diagnostics = dict(res.diagnostics)
    if "irls" not in diagnostics and "newton" in diagnostics:
        # this package's binary results report the iteration as "irls"
        nt = diagnostics["newton"]
        diagnostics["irls"] = {k: nt.get(k) for k in ("iterations", "converged", "decrement",
                                                      "step_halvings")}
    converged = bool((diagnostics.get("newton") or diagnostics.get("irls") or {})
                     .get("converged", True))
    cls = BinaryProbitResult if family == "probit" else BinaryLogitResult
    return cls(
        fml=res.fml, depvar=depvar, coefnames=list(res.coefnames),
        beta=np.asarray(res.beta, float), vcov=np.asarray(res.vcov, float),
        vcov_type=res.vcov_type, df_t=res.df_t, n_obs=res.n_obs,
        all_vcovs=dict(res.all_vcovs),
        # this package's convention: a one-way clustering's count as a number
        n_clusters={k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
                    for k, v in res.n_clusters.items()},
        collin_vars=list(res.collin_vars), diagnostics=diagnostics, weights=res.weights,
        weights_type=res.weights_type, model=f"binary {family}", loglik=float(res.loglik),
        loglik_null=float(res.loglik_null), pseudo_r2=float(res.pseudo_r2),
        aic=float(res.aic), bic=float(res.bic), loglik_fe_null=float(res.loglik_fe_null),
        pseudo_r2_within=float(res.pseudo_r2_within), firth=False,
        n_params=len(res.coefnames) + k_fe, converged=converged, fe_names=fe_names, l2=0.0,
        n_levels=dict(res.n_levels), random_effects={}, levels=[0, 1], family=family,
        deviance=float(res.deviance), k_fe=k_fe, _spec=spec)


def _draw_info(res, fe_names):
    """What `draw_parameters` needs (see draws.py)."""
    if not fe_names:
        return {"kind": "beta"}
    if len(fe_names) > 1:
        return {"kind": "beta_fixed"}
    info = getattr(res, "_newton_information", None)
    if info is None:
        return {"kind": "beta_fixed"}
    from .draws import DRAW_INFO_GB
    D, cross, Vred = info["D"], info["cross"], info["Vred"]
    G, P = cross.shape
    if G * (P + 3) * 8 / 1e9 > DRAW_INFO_GB:
        return {"kind": "beta_fixed"}
    A = res.fixef(fe_names[0]).collect()[f"fe_{fe_names[0]}"].to_numpy()
    with np.errstate(divide="ignore"):
        Dinv = np.where(D > 0, 1.0 / D, 0.0)
    return {"kind": "fe", "A": A[:, None].astype(float), "free": np.ones((G, 1), bool),
            "Vred": np.asarray(Vred, float), "fixed": None, "keep": np.ones(G, bool),
            "S": np.sqrt(Dinv)[:, None, None], "BD": (cross * Dinv[:, None])[:, None, :],
            "P": P}
