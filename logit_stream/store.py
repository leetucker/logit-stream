"""Pass 0: the source -> hdfe-stream's block store.

Pass 0 reads the source once. It evaluates the outcome and the design
(the formula's Polars expressions), drops incomplete rows, and writes what
is left -- the design, the coded outcome, the weights, and integer codes for
the cluster, fixed-effect and choice-set variables -- as fixed-size blocks of
plain numpy arrays in a `hdfe_stream.core.store.BlockStore`. Every Newton
step then reads the blocks in order. Nothing row-sized is held in memory
beyond the block being worked on (and the next one, read ahead in a
background thread). The store, the narrow copy of the sample it is written
from and the grouping of choice sets are hdfe-stream's (`core.store`, whose
docstring gives the reasons); what is here is the models' part: the
outcome's levels, the design's moments, and the choice sets' and panels'
bookkeeping.

The store is model-independent: the conditional logit only needs its blocks
to hold whole choice sets, which `_build` arranges (see `_case_order`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from hdfe_stream.core.report import warn
from hdfe_stream.core.store import (
    BlockStore,
    PolarsEngine,
    block_rows,
    factorize,
    iter_batches,
    iter_groups,
    not_missing,
    write_sample,
)

PREFIX = "__ls_"
ROW = f"{PREFIX}row"            # the row's position in the source
Y = f"{PREFIX}y"                # the outcome, raw, then its code
W = f"{PREFIX}w"                # weights
OFF = f"{PREFIX}off"            # the offset
CODE = f"{PREFIX}code"          # factorization tables' code column


def xcol(j: int) -> str:
    return f"{PREFIX}x{j}"


def check_source(columns) -> None:
    clash = [c for c in columns if c.startswith(PREFIX)]
    if clash:
        raise ValueError(f"the data has columns whose names start with {PREFIX!r}, which "
                         f"logit_stream uses for its own columns: {clash}. Rename them.")


# --------------------------------------------------------------------------
# pass 0
# --------------------------------------------------------------------------

@dataclass
class _Pass0:
    """What pass 0 learned: the store and the facts about the sample."""

    store: BlockStore
    n_raw: int                      # rows in the source
    n_obs: int                      # rows in the sample
    wsum: float                     # sum of the weights (n_obs without)
    levels: list                    # outcome levels (multinomial, ordered)
    counts: np.ndarray              # weighted count of each level
    cluster_tables: dict            # cluster name -> its levels (DataFrame, CODE)
    fe_tables: dict                 # fixed effect -> its levels (DataFrame, CODE)
    n_cases: int = 0                # choice sets (conditional, mixed logit)
    wcases: float = 0.0             # their summed weight
    n_panels: int = 0               # panels (mixed logit)
    usable: pl.Expr = None          # the rows of the source the sample keeps
    xmeans: np.ndarray = None
    xsd: np.ndarray = None
    xbinary: np.ndarray = None      # the design columns that are all 0 or 1
    xnonzero: np.ndarray = None     # each design column's share of nonzeros
    seconds: float = 0.0
    units: dict = None              # conditional FE logit: the units with variation
    incomplete: tuple = None        # (choice sets dropped as incomplete, their key columns)


# The design's 0/1 columns (indicators of factor levels, binary covariates) are
# stored as one byte each and put back together as each block is read.
COMPACT_BINARY = True


def _build(lf: pl.LazyFrame, spec, kind: str, run_dir: Path, engine: PolarsEngine, *,
           weights: str | None, clusters: dict, case: list | None, levels: list | None,
           in_memory: bool, dtype, block_mb: float, log, panel: list | None = None,
           offset: pl.Expr | None = None, weights_type: str = "aweights",
           altwise: bool = False) -> _Pass0:
    """Pass 0 for one model (see the module docstring).

    `clusters` maps each clustering's name to its source columns; `case` is
    the conditional logit's choice-set columns, and `panel` the mixed
    logit's panel columns (a chooser's choice sets). `kind` is
    ``"mlogit"``, ``"ologit"``, ``"clogit"`` or ``"mixlogit"``.
    """
    import time
    t0 = time.time()
    schema = lf.collect_schema()
    check_source(schema.names())
    K = len(spec.x)
    fe_cols = {d: d.split("^") for d in spec.fe}
    raw = list(dict.fromkeys([c for cs in clusters.values() for c in cs]
                             + [c for cs in fe_cols.values() for c in cs]
                             + (case or []) + (panel or [])))
    for c in raw + ([weights] if weights else []):
        if c not in schema:
            raise ValueError(f"column {c!r} is not in the data")
    choice = kind in ("clogit", "mixlogit", "cfelogit")
    yexpr = spec.y
    if choice:
        yexpr = yexpr.cast(pl.Float64)
    xexprs = [ex.alias(xcol(j)) for j, (_, ex) in enumerate(spec.x)]
    wexpr = [pl.col(weights).cast(pl.Float64).alias(W)] if weights else []
    keep = ([yexpr.is_not_null()]
            + ([yexpr.is_not_nan()] if choice else [])
            + [ex.is_finite() for _, ex in spec.x]
            + ([pl.col(weights).cast(pl.Float64).is_finite()] if weights else [])
            + ([offset.is_finite()] if offset is not None else [])
            + not_missing(schema, raw))
    usable = pl.all_horizontal(keep).fill_null(False) if keep else pl.lit(True)
    # The sample is written with the source columns the model uses, not the
    # design they expand to: that copy is narrow (a categorical is one column,
    # not one per level), so Polars' streaming sink -- which holds about a row
    # group per thread of what it writes -- stays small, and so does the copy
    # on disk. The design is evaluated from it batch by batch below; the
    # formula's expressions are row-wise, so that gives the same rows.
    roots = [c for e in [yexpr, *[ex for _, ex in spec.x],
                         *([offset] if offset is not None else [])]
             for c in e.meta.root_names()]
    cols = list(dict.fromkeys([*roots, *([weights] if weights else []), *raw]))
    rows = block_rows(block_mb, K)
    design_path = run_dir / "sample.parquet"
    # A choice set with an incomplete alternative is dropped whole (Stata's
    # cmclogit default): fitted on its other alternatives, it would be a
    # different choice set. altwise=True keeps them.
    incomplete = None
    if kind in ("clogit", "mixlogit") and not altwise:
        skey = list(dict.fromkeys([*(panel or []), *case]))
        bad = engine.collect(lf.filter(~usable).select(skey).drop_nulls().unique(),
                             "incomplete choice sets")
        if bad.height:
            incomplete = (bad, skey)
            warn(f"{spec.fml}: {bad.height:,} choice sets dropped because some of their "
                 "alternatives have missing values (altwise=True keeps their complete "
                 "alternatives)")
    log("pass 0: writing the sample")
    write_sample(engine, lf, usable, cols, design_path, rows, ROW, drop=incomplete)
    evaluated = [pl.col(ROW), yexpr.alias(Y), *xexprs, *wexpr,
                 *([offset.alias(OFF)] if offset is not None else []),
                 *[pl.col(c) for c in raw if c not in (Y, W)]]
    design = pl.scan_parquet(design_path).select(evaluated)

    # facts about the sample, from the local copy
    stats = engine.collect(design.select(
        pl.len().alias("n"),
        *([pl.col(W).sum().alias("wsum"), pl.col(W).min().alias("wmin")] if weights else []),
        *[pl.col(xcol(j)).mean().alias(f"m{j}") for j in range(K)],
        *[pl.col(xcol(j)).std().alias(f"s{j}") for j in range(K)],
        *[((pl.col(xcol(j)) != 0) & (pl.col(xcol(j)) != 1)).sum().alias(f"b{j}")
          for j in range(K)],
        *[(pl.col(xcol(j)) != 0).mean().alias(f"z{j}") for j in range(K)]), "pass 0 moments")
    n_obs = int(stats["n"].item())
    n_raw = int(engine.collect(lf.select(pl.len()), "row count").item())
    if n_obs == 0:
        raise ValueError(f"{spec.fml}: no complete observations")
    if weights:
        if stats["wmin"].item() <= 0:
            raise ValueError("weights must be strictly positive")
        wsum = float(stats["wsum"].item())
    else:
        wsum = float(n_obs)
    # Analytic weights are written rescaled to average 1 over the sample (over
    # the choice sets for the conditional logit, set below), as Stata does for
    # maximum likelihood: the estimates do not change, but the iid vcov, the
    # log-likelihood, Firth's penalty and the random intercepts' Laplace
    # approximation would otherwise depend on the weights' scale. Frequency
    # weights are counts, and stay.
    choice_units = kind in ("clogit", "mixlogit", "cfelogit")
    wscale = [n_obs / wsum if weights and weights_type != "fweights" and not choice_units
              else 1.0]
    wsum *= wscale[0]
    xmeans = np.array([stats[f"m{j}"].item() for j in range(K)], dtype=float)
    xsd = np.array([stats[f"s{j}"].item() or 0.0 for j in range(K)], dtype=float)
    xbinary = np.array([stats[f"b{j}"].item() == 0 for j in range(K)], dtype=bool)
    xnonzero = np.array([stats[f"z{j}"].item() or 0.0 for j in range(K)], dtype=float)

    # the outcome's levels and their (weighted) counts
    wcount = pl.col(W).sum() if weights else pl.len().cast(pl.Float64)
    freq = engine.collect(design.group_by(Y).agg(wcount.alias("count")).sort(Y),
                          "outcome levels")
    if choice:
        bad = freq.filter(~pl.col(Y).is_in([0.0, 1.0]))
        if bad.height:
            raise ValueError(f"{spec.fml}: the outcome of a conditional logit must be 0 or 1 "
                             f"(1 for the chosen alternative); found {bad[Y].to_list()[:5]}")
        lv, counts = [0, 1], None
        ytable = None
    else:
        observed = freq[Y].to_list()
        if levels is not None:
            levels = list(levels)
            missing = [v for v in observed if v not in levels]
            if missing:
                raise ValueError(f"{spec.fml}: the outcome takes values not in levels=: "
                                 f"{missing[:10]}")
            absent = [v for v in levels if v not in observed]
            if absent:
                raise ValueError(f"{spec.fml}: levels {absent} of levels= never occur in the "
                                 "estimation sample")
            lv = levels
        else:
            lv = observed
        if len(lv) < 2:
            raise ValueError(f"{spec.fml}: the outcome takes only one value in the sample "
                             f"({lv})")
        cmap = dict(zip(freq[Y].to_list(), freq["count"].to_list()))
        counts = np.array([cmap[v] for v in lv], dtype=float) * wscale[0]
        ytable = pl.DataFrame({Y: pl.Series(lv, dtype=freq.schema[Y]),
                               "__ls_ycode": pl.Series(range(len(lv)), dtype=pl.Int32)})

    cluster_tables = {name: factorize(engine, design, cols, f"cluster {name}", CODE)
                      for name, cols in clusters.items()}
    fe_tables = {d: factorize(engine, design, cols, f"fixed effect {d}", CODE)
                 for d, cols in fe_cols.items()}

    store = BlockStore(run_dir / "blocks", in_memory, dtype,
                       binary=xbinary if COMPACT_BINARY else None)
    joins = ([(name, clusters[name], t) for name, t in cluster_tables.items()],
             [(d, fe_cols[d], t) for d, t in fe_tables.items()])

    panels_seen = [0]

    def convert(frame: pl.DataFrame, starts=None, pstarts=None):
        frame = frame.select(evaluated)                 # the design, for this batch
        pidx = None
        if pstarts is not None:
            pidx = np.arange(panels_seen[0], panels_seen[0] + len(pstarts) - 1, dtype=np.int64)
            panels_seen[0] += len(pstarts) - 1
        if ytable is not None:
            frame = frame.join(ytable, on=Y, how="left", maintain_order="left")
            y = frame["__ls_ycode"].to_numpy().astype(np.int32)
        else:
            y = frame[Y].to_numpy().astype(np.int8)
        cl = fe = None
        if joins[0]:
            codes = []
            for _, cols, t in joins[0]:
                codes.append(frame.select(cols).join(t, on=cols, how="left",
                                                     maintain_order="left")[CODE].to_numpy())
            cl = np.column_stack(codes).astype(np.int64)
        more = {}
        if joins[1]:
            codes = [frame.select(cols).join(t, on=cols, how="left", maintain_order="left")[CODE]
                     .to_numpy().astype(np.int64) for _, cols, t in joins[1]]
            fe = codes[0]
            more = {f"fe{k + 1}": c for k, c in enumerate(codes) if k}     # fe2, fe3, ...
        X = (frame.select([xcol(j) for j in range(K)]).to_numpy(order="c")
             if K else np.zeros((frame.height, 0)))
        store.add({"X": X.astype(np.float64, copy=False), "y": y,
                   "w": frame[W].to_numpy().astype(np.float64) * wscale[0] if weights else None,
                   "off": frame[OFF].to_numpy() if offset is not None else None,
                   "cl": cl, "fe": fe, **more, "starts": starts, "pstarts": pstarts,
                   "pidx": pidx,
                   "row": frame[ROW].to_numpy().astype(np.int64)})

    n_cases, wcases, n_panels = 0, 0.0, 0
    units = {}
    log(f"pass 0: writing {n_obs:,} rows as blocks of {rows:,}"
        + (" (in memory)" if in_memory else ""))
    if choice:
        ccols = list(dict.fromkeys(c for cs in clusters.values() for c in cs))
        n_cases, wcases, n_panels = _case_order(engine, design, design_path, case, weights,
                                                spec, rows, convert, run_dir, log, panel,
                                                ccols, wscale if weights_type != "fweights"
                                                else None, kind=kind, info=units)
    else:
        for frame in iter_batches(design_path, rows):
            convert(frame)
    design_path.unlink(missing_ok=True)
    return _Pass0(store=store, n_raw=n_raw, n_obs=n_obs, wsum=wsum, levels=lv, units=units,
                  incomplete=incomplete,
                  counts=counts, cluster_tables=cluster_tables, fe_tables=fe_tables,
                  n_cases=n_cases, wcases=wcases, n_panels=n_panels, usable=usable,
                  xmeans=xmeans, xsd=xsd, xbinary=xbinary, xnonzero=xnonzero,
                  seconds=round(time.time() - t0, 2))


def _case_order(engine, design, design_path, case, weights, spec, rows, convert, run_dir,
                log, panel=None, cluster_cols=(), wscale=None, kind="clogit", info=None):
    """Write the conditional logit's blocks so that each holds whole choice
    sets, each a run of consecutive rows; return the number of sets, their
    summed weight and the number of panels.

    With `panel` (the mixed logit's choosers), a choice set is identified by
    the panel and case columns together, a panel's sets are consecutive, and
    blocks hold whole panels; each block then also records its panels'
    first rows and their positions among all panels, from which the panel's
    simulation draws are generated, so that they do not depend on how the
    rows are divided into blocks.

    Long-format choice data usually arrives grouped by choice set already,
    and then the rows are written in their order; otherwise they are sorted
    by set in buckets (`hdfe_stream.core.store.iter_groups`)."""
    skey = list(dict.fromkeys([*(panel or []), *case]))
    unit = list(panel) if panel else skey

    def runs_of(cols):
        return pl.any_horizontal([(pl.col(c) != pl.col(c).shift()).fill_null(True)
                                  for c in cols])

    changed, uchanged = runs_of(skey), runs_of(unit)
    per = engine.collect(design.group_by(skey).agg(
        pl.col(Y).sum().alias("chosen"), pl.len().alias("size"),
        *([(pl.col(W).max() - pl.col(W).min()).alias("wspread"),
           pl.col(W).first().alias("wcase")] if weights else []),
        *[pl.col(c).n_unique().alias(f"__ls_nu_{i}") for i, c in enumerate(cluster_cols)]),
        "choice sets")
    n_cases = per.height
    wcases = float(per["wcase"].sum()) if weights else float(n_cases)
    if weights and wscale is not None:          # analytic weights: average 1 over the sets
        wscale[0] = n_cases / wcases
        wcases = float(n_cases)
    n_panels = per.select(panel).unique().height if panel else 0
    if kind == "cfelogit":
        # units whose outcome never varies carry no information (their rows stay,
        # and the likelihood skips them); the null, every arrangement of a unit's
        # 1s equally likely, is -sum log C(T, k)
        from scipy.special import gammaln
        T, k = per["size"].to_numpy().astype(float), per["chosen"].to_numpy()
        live = (k > 0) & (k < T)
        wu = per["wcase"].to_numpy() if weights else np.ones(len(T))
        if weights and wscale is not None:      # average 1 over the informative units
            wscale[0] = live.sum() / wu[live].sum()
        wu = wu * (wscale[0] if weights and wscale is not None else 1.0)
        lc = gammaln(T + 1) - gammaln(k + 1) - gammaln(T - k + 1)
        info.update(units=int(live.sum()), dropped=int((~live).sum()),
                    rows=int(T[live].sum()), wunits=float(wu[live].sum()),
                    loglik_null=float(-(lc[live] * wu[live]).sum()))
    bad = per.filter(pl.col("chosen") != 1) if kind != "cfelogit" else per.head(0)
    if bad.height:
        raise ValueError(
            f"{spec.fml}: every choice set must have exactly one chosen alternative "
            f"(outcome 1); {bad.height:,} of {n_cases:,} do not (e.g. "
            f"{bad.select(skey).head(3).to_dicts()}). Rows dropped for missing values can "
            "remove a set's chosen alternative.")
    if weights and per["wspread"].max() > 0:
        raise ValueError("weights must be constant within each choice set (a choice set is "
                         "one observation)")
    for i, c in enumerate(cluster_cols):
        spread = per.filter(pl.col(f"__ls_nu_{i}") > 1)
        if spread.height:
            raise ValueError(f"cluster variable {c!r} must be constant within each choice set "
                             f"(a choice set is one observation); {spread.height:,} sets "
                             f"have more than one value (e.g. "
                             f"{spread.select(skey).head(3).to_dicts()})")
    del per

    # blocks of whole units (sets, or panels with their sets consecutive)
    for body in iter_groups(engine, design_path, unit, rows, run_dir,
                            n_panels if panel else n_cases, [*skey, ROW],
                            also=[(skey, n_cases)] if panel else (), log=log):
        starts = np.append(np.flatnonzero(body.select(changed.alias("b"))["b"].to_numpy()),
                           body.height).astype(np.int64)
        pstarts = None
        if panel:
            pstarts = np.append(np.flatnonzero(body.select(uchanged.alias("b"))["b"]
                                               .to_numpy()), body.height).astype(np.int64)
        convert(body, starts, pstarts)
    return n_cases, wcases, n_panels
