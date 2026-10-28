"""The multinomial logit with several absorbed fixed effects, as a Poisson
regression (hdfe-stream's `fepois_stream`) on the data expanded J-fold.

With one row per observation i and outcome level j, the count 1[y_i = j],
a fixed effect per observation, each fixed effect interacted with the level
and each covariate with the non-base levels, the Poisson log-likelihood
concentrated over the observations' effects is the multinomial logit's
(Baker 1994; the "Poisson trick" of Guimaraes, Figueiredo and Woodward 2003
for the conditional logit): the observation's effect makes each row's mean
its probability, sum_j mu_ij = 1. The coefficients are the multinomial
logit's, its log-likelihood is the Poisson's plus the number (or weighted
number) of observations, and the information for the coefficients, net of
the effects, is the same, so the variances are too once the small-sample
factors are logit-stream's (inverse information; robust = clustered by
observation, N / (N - 1); clustered, G / (G - 1)) rather than pyfixest's.
"""

from __future__ import annotations

import re
import time
import warnings

import numpy as np
import polars as pl
from hdfe_stream.core.report import warn
from hdfe_stream.core.store import not_missing

_OBS, _LVL, _CNT, _W = "__ls_obs", "__ls_lvl", "__ls_count", "__ls_w"


def fit_poisson_route(spec, lf, engine, *, vkey, clusters, requests, hetero, weights,
                      weights_type, levels, tol, maxiter, workdir, keep_intermediates, log,
                      logger, options=None):
    """`mlogit_stream` for a formula with several absorbed fixed effects:
    fitted by `fepois_stream` on the expanded data (module docstring), the
    result a `MultinomialLogitResult`. `options` go to `fepois_stream`
    (`device`, `polars_engine`, and the caller's `poisson_options`)."""
    import hdfe_stream as hs

    from .results import MultinomialLogitResult

    t0 = time.time()
    log(f"multinomial logit with {len(spec.fe)} fixed effects, as a Poisson regression "
        "on the data expanded by outcome level")
    xs = [(nm, ex) for nm, ex in spec.x if nm != "Intercept"]
    if not xs:
        raise ValueError(f"{spec.fml}: the multinomial logit needs covariates")
    fe_cols = {d: d.split("^") for d in spec.fe}
    cl_cols = sorted({c for cols in clusters.values() for c in cols})
    schema = lf.collect_schema()
    roots = set()
    for _, ex in [(None, spec.y), *xs]:
        roots |= set(ex.meta.root_names())
    roots |= {c for cols in fe_cols.values() for c in cols}
    roots |= set(cl_cols) | ({weights} if weights else set())
    usable = pl.all_horizontal(not_missing(schema, sorted(roots)))
    for _, ex in xs:
        usable = usable & ex.cast(pl.Float64).is_finite()
    Y = "__ls_y"
    base = lf.with_row_index(_OBS).filter(usable)
    if levels is None:
        lv = engine.collect(base.select(spec.y.alias(Y)).unique().sort(Y), "outcome levels")[Y]
    else:
        lv = pl.Series(Y, list(levels))
        seen = set(engine.collect(base.select(spec.y.alias(Y)).unique(), "outcome levels")[Y])
        if seen - set(lv.to_list()) or set(lv.to_list()) - seen:
            raise ValueError(f"{spec.fml}: levels= must be the outcome's values in the "
                             f"sample, got {lv.to_list()} for {sorted(seen)}")
    J = len(lv)
    if J < 2:
        raise ValueError(f"{spec.fml}: the outcome takes only one value in the sample")
    K = len(xs)
    xcols = [f"__ls_x{k}_{j}" for j in range(1, J) for k in range(K)]
    names = [f"{_name(lv[j])}:{nm}" for j in range(1, J) for nm, _ in xs]
    table = pl.DataFrame({Y: lv, _LVL: pl.Series(range(J), dtype=pl.Int32)})
    keep = [_OBS, *sorted({c for cols in fe_cols.values() for c in cols} | set(cl_cols))]
    wide = base.select(*[pl.col(c) for c in keep], spec.y.alias(Y),
                       *[ex.cast(pl.Float64).alias(f"__ls_x{k}") for k, (_, ex) in enumerate(xs)],
                       *([pl.col(weights).cast(pl.Float64).alias(_W)] if weights else []))
    long = (wide.join(table.lazy().rename({Y: "__ls_lv"}), how="cross")
            .with_columns((pl.col(Y) == pl.col("__ls_lv")).cast(pl.Float64).alias(_CNT),
                          *[pl.when(pl.col(_LVL) == j).then(pl.col(f"__ls_x{k}"))
                            .otherwise(0.0).alias(f"__ls_x{k}_{j}")
                            for j in range(1, J) for k in range(K)])
            .drop([Y, "__ls_lv", *[f"__ls_x{k}" for k in range(K)]]))
    fml = (f"{_CNT} ~ {' + '.join(xcols)} | {_OBS} + "
           + " + ".join(f"{d}^{_LVL}" for d in spec.fe))
    if hetero and weights and weights_type == "fweights":
        raise ValueError(f"{spec.fml}: with several absorbed fixed effects (fitted as a "
                         "Poisson regression clustered by observation), the robust variance "
                         "is not available with frequency weights (the clusters' scores "
                         "would carry each weight squared); use a clustered or the iid one")
    # the robust variance is clustered by observation, the clustered ones as asked
    hkey = {"iid": "iid", "hetero": f"CRV1:{_OBS}"}
    to_h = {k: hkey.get(k, k) for k in ["iid", *(["hetero"] if hetero else []), *requests]}
    vcov_h = to_h[vkey]
    extra = [v.split(":", 1)[1] for k, v in to_h.items() if v != vcov_h and k != "iid"]
    opts = dict(options or {})
    taken = [k for k in ("vcov", "cluster", "weights", "weights_type", "keep", "workdir")
             if k in opts]
    if taken:
        raise ValueError(f"poisson_options= cannot set {taken}: they are the multinomial "
                         "logit's own (or the route's)")
    opts.setdefault("iwls_tol", min(tol, 1e-8))
    opts.setdefault("iwls_maxiter", max(maxiter, 25))
    # the Poisson regression's separation warning is about its expanded rows:
    # say it in the multinomial logit's terms
    plog = _Translating(logger, spec.fml) if logger is not None else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = hs.fepois_stream(fml, long, workdir=workdir, vcov=vcov_h, cluster=extra,
                             weights=_W if weights else None, weights_type=weights_type,
                             keep=[_LVL, *([_W] if weights else [])], verbose=False,
                             logger=plog, **opts)
    for w in caught:
        msg = _translate(str(w.message), spec.fml)
        if msg is None:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        else:
            warn(msg, logger)
    irls = p.diagnostics.get("irls", {})
    dropped = [nm for nm, c in zip(names, xcols) if c not in p.coefnames]
    if dropped:
        p.cleanup()
        raise ValueError(f"{spec.fml}: {dropped} are collinear with the fixed effects; "
                         "drop them from the formula")
    sel = [p.coefnames.index(c) for c in xcols]

    # each observation in the sample has one row with a count of 1: its fitted
    # mean is its probability, and its weight is in the row
    w = pl.col(_W) if weights else pl.col(_CNT) * 0.0 + 1.0
    obs = engine.collect(
        p.resid().filter(pl.col(_CNT) == 1.0).group_by(_LVL)
        .agg(pl.len().alias("n"), w.sum().alias("w"),
             (w * pl.col("fitted").log()).sum().alias("ll")).sort(_LVL),
        "the sample's log-likelihood")
    rows_w = engine.collect(p.resid().select(w.sum().alias("w"), pl.len().alias("n")),
                            "the sample's weights")
    n_obs, w_obs = int(obs["n"].sum()), float(obs["w"].sum())
    # analytic weights are scaled to average 1 over the observations (hdfe-stream
    # scales them over its rows, the observations' J rows less those dropped)
    if weights and weights_type != "fweights":
        s_ls, s_h = n_obs / w_obs, float(rows_w["n"][0]) / float(rows_w["w"][0])
    else:
        s_ls = s_h = 1.0
    N = w_obs if weights and weights_type == "fweights" else n_obs
    ll = float(obs["ll"].sum()) * s_ls
    cnt = obs["w"].to_numpy() * s_ls
    ll0 = float(np.sum(cnt * np.log(cnt / cnt.sum())))
    P = len(names)
    all_vcovs, n_clusters = {}, {}
    factors = p.diagnostics["vcov_factors"]
    for k, kh in to_h.items():
        Vh, _ = p.all_vcovs[kh]
        V = Vh[np.ix_(sel, sel)] / factors[kh]
        if k == "iid":
            all_vcovs[k] = (V * s_h / s_ls, int(N - P))
        elif k == "hetero":
            all_vcovs[k] = (V * N / max(N - 1, 1), int(N - P))
        else:
            Gs = p.n_clusters[k.split(":", 1)[1]]
            Gm = min(Gs)
            all_vcovs[k] = (V * Gm / (Gm - 1) if Gm > 1 else V * np.nan, int(Gm - 1))
            n_clusters[k.split(":", 1)[1]] = Gs[0] if len(Gs) == 1 else Gs
    V, df_t = all_vcovs[vkey]
    beta = np.asarray(p.beta)[sel]

    # each fixed effect's table, a column per level, the base level's 0 where the
    # group has it (each group's effects are identified up to a constant)
    fixef, n_levels, k_fe = {}, {}, 0
    for d, cols in fe_cols.items():
        t = engine.collect(p.fixef(f"{d}^{_LVL}").select(*cols, _LVL, f"fe_{d}^{_LVL}", "n_obs"),
                           f"fixed effects {d}")
        cells = t.height
        wide_t = t.pivot(on=_LVL, index=cols, values=f"fe_{d}^{_LVL}")
        n = t.group_by(cols).agg(pl.col("n_obs").first())
        b0 = pl.col("0") if "0" in wide_t.columns else pl.lit(None, dtype=pl.Float64)
        vals = []
        for j in range(J):
            c = str(j)
            e = (pl.col(c) - pl.when(b0.is_not_null()).then(b0).otherwise(0.0)
                 if c in wide_t.columns else pl.lit(None, dtype=pl.Float64))
            vals.append(e.alias(f"fe_{d}:{_name(lv[j])}"))
        fixef[d] = wide_t.select(*cols, *vals).join(n, on=cols, how="left")
        n_levels[d] = fixef[d].height
        k_fe += cells - fixef[d].height
    # (the dimensions share one constant per non-base level, connected)
    k_fe -= (J - 1) * (len(spec.fe) - 1)
    aic = -2 * ll + 2 * (P + k_fe)
    bic = -2 * ll + np.log(N) * (P + k_fe)
    rspec = {"source": lf, "xexprs": dict(xs), "xcols": [nm for nm, _ in xs], "usable": usable,
             "yexpr": spec.y, "n_raw": engine.collect(lf.select(pl.len()), "rows").item(),
             "fe_cols": fe_cols, "offset": None, "incomplete": None, "fixef": fixef,
             "effects": [(d, [f"fe_{d}:{_name(x)}" for x in lv], "fixed") for d in fe_cols]}
    diagnostics = {"route": "poisson",
                   "poisson": {"formula": fml, "n_rows": int(p.n_obs), "k_fe": int(p.k_fe),
                               "irls": dict(irls)},
                   "n_rows": n_obs, "seconds_total": round(time.time() - t0, 2)}
    res = MultinomialLogitResult(
        fml=spec.fml, depvar=spec.depvar, coefnames=names, beta=beta, vcov=V, vcov_type=vkey,
        df_t=df_t, n_obs=int(N) if float(N).is_integer() else N, all_vcovs=all_vcovs,
        n_clusters=n_clusters, collin_vars=dropped, diagnostics=diagnostics, weights=weights,
        weights_type=weights_type, model="multinomial logit", loglik=ll, loglik_null=ll0,
        pseudo_r2=float(1 - ll / ll0) if ll0 else np.nan, aic=float(aic), bic=float(bic),
        n_params=P + k_fe, converged=bool(irls.get("converged", True)),
        fe_names=list(spec.fe), n_levels=n_levels, levels=lv.to_list(), base=lv[0],
        _spec=rspec)
    if not keep_intermediates:
        p.cleanup()
    return res


_SEPARATED = re.compile(r"([\d,]+) levels of (\S+?)\^" + re.escape(_LVL))


def _translate(msg, fml):
    """The multinomial logit's version of the Poisson regression's
    separation warning, or None for any other message."""
    if "because of separation" not in msg or _LVL not in msg:
        return None
    cells = [f"{n} ({d}, outcome level) cells" for n, d in _SEPARATED.findall(msg)]
    return (f"{fml}: {', '.join(cells)} have no observations of that level: their "
            "probability is 0 (their effects are minus infinity), and fixef() shows a null")


class _Translating:
    """A logger that passes the Poisson regression's records on, its
    separation warning translated (`_translate`)."""

    def __init__(self, logger, fml):
        self._logger, self._fml = logger, fml

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(_translate(str(msg), self._fml) or msg, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._logger, name)


def _name(v):
    return str(v)
