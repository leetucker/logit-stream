"""Calibrated predictions of the conditional logit with absorbed
alternative constants: one dimension's constants solved from known totals
(`ConditionalLogitResult.predict(calibrate=...)`).

Each level's constant delta is the one at which its predicted choices,
summed over the choice sets, equal its total, given the coefficients and
any other terms: the contraction delta += log(total / predicted), which
rakes the sets x levels matrix of probabilities so that each set's row
sums to 1 and each level's column to its total. The solve is separate for
each group of levels that the sets link (a firm's establishments), so
`newdata` is streamed a batch of whole groups at a time:

1. one scan of `newdata` writes its rows, utilities and totals to disk;
2. the groups, by min-label propagation over the (set, level) pairs on
   disk, with pointer jumping on the levels' labels (each round two
   streaming aggregations; memory a table per set and one per level);
3. whole groups are packed into batches of about `CALIBRATE_ROWS` rows and
   the rows partitioned by batch;
4. each batch is solved in memory and its predictions written out.

The predictions come back as a lazy scan of those files in `newdata`'s
order. When `newdata` is a single batch, it is solved in memory without
the partitioning.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

CALIBRATE_ROWS = 2_000_000      # rows per batch (about 0.4 GB of working memory)
CALIBRATE_TOL = 1e-10
CALIBRATE_MAXITER = 100     # Newton steps
CG_MAXITER = 1_000
CONTRACT_ROUNDS = 100          # before Newton
MAX_STEP = 1.0                  # the largest Newton step in a constant
MAX_ROUNDS = 1_000              # of the label propagation


def calibrated(result, frame, type, prefix, case, calibrate, batch_rows=None, workdir=None):
    """`predict` with one dimension's constants solved from totals."""
    from hdfe_stream.core.report import warn
    from hdfe_stream.core.utils import as_lazy

    from .estimator import _base_dir
    if not isinstance(calibrate, dict) or len(calibrate) != 1:
        raise ValueError("calibrate= is {dimension: totals}, for one dimension")
    (name, totals), = calibrate.items()
    cols = result._spec["fe_cols"].get(name)
    if cols is None:
        raise ValueError(f"calibrate: {name!r} is not an absorbed dimension of this model "
                         f"({list(result._spec['fe_cols'])})")
    tot = as_lazy(totals).collect()
    other = [c for c in tot.columns if c not in cols]
    if len(other) != 1:
        raise ValueError(f"calibrate: the totals frame needs {cols} and one column of "
                         f"totals; it has {tot.columns}")
    batch_rows = int(batch_rows or CALIBRATE_ROWS)
    keep = frame.collect_schema().names()
    out_name = prefix or ("xb" if type == "link" else "prob")

    # the utilities without this dimension's constants, and each row's level and total
    spec = dict(result._spec)
    spec["effects"] = [e for e in spec.get("effects", []) if e[0] != name]
    bare = replace(result, _spec=spec)
    lf = bare._predict(frame.with_row_index("__row"), "link", "__xb", case)
    schema = lf.collect_schema()
    tot = tot.select(*cols, pl.col(other[0]).cast(pl.Float64).alias("__t")) \
        .with_row_index("__lev").with_columns(pl.col("__lev").cast(pl.Int64))
    tot = tot.with_columns(pl.col(c).cast(schema[c], strict=False) for c in cols)
    lf = lf.join(tot.lazy(), on=cols, how="left", nulls_equal=True)
    L = tot.height

    tmp = Path(tempfile.mkdtemp(prefix="logit_calibrate_", dir=_base_dir(workdir)))
    try:
        rows = tmp / "rows.parquet"
        lf.sink_parquet(rows, engine="streaming")
        n_rows, n_missing = pl.scan_parquet(rows).select(
            pl.len(), pl.col("__lev").is_null().sum()).collect(engine="streaming").row(0)
        if n_missing:
            raise ValueError(f"calibrate: {n_missing:,} rows of newdata have a {name} without "
                             "a total")
        stats = _Stats()
        if n_rows <= batch_rows:
            res = _solve(pl.read_parquet(rows), case, L, stats)
            shutil.rmtree(tmp, ignore_errors=True)
            stats.report(warn)
            return _output(res.sort("__row").lazy(), keep, type, out_name)
        batches = _batches(rows, case, L, batch_rows)
        parts = tmp / "parts"
        pl.scan_parquet(rows).join(batches.lazy(), on=case, how="left", nulls_equal=True) \
            .sink_parquet(pl.PartitionBy(str(parts), key="__b", include_key=False),
                          engine="streaming", mkdir=True)
        rows.unlink()
        out = tmp / "out"
        out.mkdir()
        for d in sorted(parts.iterdir()):
            res = _solve(pl.read_parquet(d / "*.parquet"), case, L, stats)
            res.select(*keep, "__row", "__u", "__p").write_parquet(out / f"{d.name}.parquet")
            shutil.rmtree(d)
            del res
        stats.report(warn)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    atexit.register(shutil.rmtree, tmp, True)
    return _output(pl.scan_parquet(out / "*.parquet").sort("__row"), keep, type, out_name)


def _output(lf, keep, type, out_name):
    col = pl.col("__u") if type == "link" else pl.col("__p")
    return lf.select(*keep, col.alias(out_name))


def _batches(rows, case, L, batch_rows):
    """Each set's batch: whole groups of linked levels (min-label propagation
    over the live (set, level) pairs) packed into batches of about
    `batch_rows` rows. A set without a live row is its own group."""
    scan = pl.scan_parquet(rows)
    edges = scan.filter(_live()).select(*case, "__lev")
    lab = np.arange(L, dtype=np.int64)
    for _ in range(MAX_ROUNDS):
        labs = pl.DataFrame({"__lev": np.arange(L, dtype=np.int64), "__g": lab}).lazy()
        per_set = edges.join(labs, on="__lev").group_by(case).agg(pl.col("__g").min())
        per_lev = edges.join(per_set, on=case, nulls_equal=True).group_by("__lev") \
            .agg(pl.col("__g").min()).collect(engine="streaming")
        new = lab.copy()
        np.minimum.at(new, per_lev["__lev"].to_numpy(), per_lev["__g"].to_numpy())
        while True:                                 # pointer jumping
            nxt = new[new]
            if np.array_equal(nxt, new):
                break
            new = nxt
        if np.array_equal(new, lab):
            break
        lab = new
    else:
        raise RuntimeError("calibrate: the groups of levels did not settle in "
                           f"{MAX_ROUNDS} rounds")
    labs = pl.DataFrame({"__lev": np.arange(L, dtype=np.int64), "__g": lab}).lazy()
    sets = scan.group_by(case).agg(pl.len().alias("__n")) \
        .join(edges.join(labs, on="__lev").group_by(case).agg(pl.col("__g").min()),
              on=case, how="left", nulls_equal=True) \
        .collect(engine="streaming")
    lone = sets["__g"].is_null()
    sets = sets.with_columns(pl.when(lone).then(L + lone.cast(pl.Int64).cum_sum())
                             .otherwise(pl.col("__g")).alias("__g"))
    groups = sets.group_by("__g").agg(pl.col("__n").sum()).sort("__g")
    before = groups["__n"].cum_sum() - groups["__n"]
    groups = groups.with_columns((before // batch_rows).cast(pl.Int64).alias("__b"))
    return sets.join(groups.select("__g", "__b"), on="__g").select(*case, "__b")


def _live():
    return (pl.col("__t") > 0) & pl.col("__xb").is_not_null() & pl.col("__xb").is_finite()


class _Stats:
    """The warnings, over the batches."""

    def __init__(self):
        self.groups = self.off = 0
        self.off_max = 0.0
        self.unconverged = 0
        self.step = 0.0

    def report(self, warn):
        if self.off:
            warn(f"calibrate: in {self.off:,} of {self.groups:,} groups the totals differ "
                 f"from the number of choice sets by more than 1% (by up to "
                 f"{self.off_max:.0%}); they were scaled to it")
        if self.unconverged:
            warn(f"calibrate: the constants did not converge in {CALIBRATE_MAXITER} Newton steps "
                 f"(largest step {self.step:.2g})")


def _solve(df, case, L, stats, tol=CALIBRATE_TOL, maxiter=CALIBRATE_MAXITER):
    """One batch of whole groups, in memory: the rows with ``__p`` (the
    probability; null for a set with no live row) and ``__u`` (the utility
    with the solved constant; null for a dead row)."""
    from .alt_fe import _link, _roots
    df = df.sort([*case, "__row"])
    if df.height == 0:
        return df.with_columns(pl.lit(None, pl.Float64).alias("__p"),
                               pl.lit(None, pl.Float64).alias("__u"))
    cid = df.select(pl.struct(case).rank("dense").alias("c"))["c"].to_numpy() - 1
    starts = np.r_[np.flatnonzero(np.r_[True, cid[1:] != cid[:-1]]), len(cid)] \
        .astype(np.int64)
    # the batch's levels, coded densely
    used, lev = np.unique(df["__lev"].to_numpy().astype(np.int64), return_inverse=True)
    lev = lev.astype(np.int64)
    Lb = len(used)
    target = np.zeros(Lb)
    target[lev] = df["__t"].to_numpy()
    xb = df["__xb"].fill_null(np.nan).to_numpy().astype(float)
    live = (target[lev] > 0) & np.isfinite(xb)
    # the groups of linked levels, and each one's number of sets
    parent = np.arange(Lb, dtype=np.int64)
    _link(lev[:, None], starts, live, np.ones(len(starts) - 1, bool), parent)
    root = _roots(parent)
    st0 = starts[:-1]
    n_sets = np.zeros(Lb)
    any_live = np.maximum.reduceat(live.astype(np.int64), st0) > 0
    first_live = np.maximum.reduceat(np.where(live, lev, -1), st0)
    np.add.at(n_sets, root[first_live[any_live]], 1.0)
    gsum = np.zeros(Lb)
    np.add.at(gsum, root, target)
    pos = gsum[root] > 0
    scale = np.where(pos, n_sets[root] / np.where(pos, gsum[root], 1.0), 0.0)
    g = np.unique(root[lev[live]])
    off_by = np.abs(n_sets[g] / np.where(gsum[g] > 0, gsum[g], 1.0) - 1)
    stats.groups += len(g)
    stats.off += int((off_by > 0.01).sum())
    stats.off_max = max(stats.off_max, float(off_by.max(initial=0.0)))
    target = target * scale
    ok = target > 0
    delta = np.where(ok, np.log(np.maximum(target, 1e-300)), 0.0)
    delta, p, S, step = _newton(xb, lev, live, starts, target, ok, delta, tol, maxiter)
    if np.max(np.abs(step), initial=0.0) >= tol:
        stats.unconverged += 1
        stats.step = max(stats.step, float(np.max(np.abs(step))))
    return df.with_columns(pl.Series("__p", np.where(S > 0, p, np.nan)).fill_nan(None),
                           pl.Series("__u", np.where(live, xb + delta[lev], np.nan))
                           .fill_nan(None))


def _probs(xb, lev, live, st0, sizes, delta):
    """Each row's probability in its set, each set's normalizer (0: no live
    row), and the objective sum_sets log sum exp."""
    u = np.where(live, xb + delta[lev], -np.inf)
    m = np.maximum.reduceat(u, st0)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.where(live, np.exp(u - np.repeat(m, sizes)), 0.0)
    s = np.add.reduceat(e, st0)
    S = np.repeat(s, sizes)
    p = np.where(S > 0, e / np.where(S > 0, S, 1.0), 0.0)
    lse = np.sum(np.where(s > 0, m + np.log(np.where(s > 0, s, 1.0)), 0.0))
    return p, S, lse


def _newton(xb, lev, live, starts, target, ok, delta, tol, maxiter):
    """The constants at which each level's predicted choices equal its
    target: contraction rounds delta += log(target / predicted) while they
    converge fast; then the maximum of the concave F(delta) = target'delta -
    sum_sets log sum exp(xb + delta), by Newton's method. The step is from
    conjugate gradients on the Hessian's products (sum over the sets of
    diag(p) - pp', never formed), at most MAX_STEP in any constant, and
    halved until F rises or the largest log(target / predicted) falls; a
    contraction step delta += log(target / predicted) when neither does.
    Converged when the largest log ratio is below `tol`. Returns delta,
    the probabilities, the sets' normalizers and the last log ratios."""
    st0 = starts[:-1]
    sizes = np.diff(starts)
    Lb = len(target)
    ok_rows = ok[lev] & live

    def hess(v, p):
        pv = np.where(ok_rows, p * v[lev], 0.0)
        s = np.repeat(np.add.reduceat(pv, st0), sizes)
        return np.bincount(lev, weights=pv - p * s, minlength=Lb) * ok

    def F(d):
        p, S, lse = _probs(xb, lev, live, st0, sizes, d)
        return p, S, float(target[ok] @ d[ok]) - lse

    def ratios(p):
        pred = np.bincount(lev, weights=p, minlength=Lb)
        step = np.zeros(Lb)
        step[ok] = np.log(target[ok] / np.maximum(pred[ok], 1e-300))
        return pred, step

    # contraction rounds first: fast when each group's sets see all its
    # levels (a firm's workers and its establishments), slow on long chains
    # of linked levels, where Newton's steps take over
    p, S, _ = _probs(xb, lev, live, st0, sizes, delta)
    seen = []
    for k in range(CONTRACT_ROUNDS):
        _, step = ratios(p)
        worst = np.max(np.abs(step), initial=0.0)
        if worst < tol:
            return delta, p, S, step
        if k >= 10 and worst > 0.5 * seen[k - 5]:   # not halving in 5 rounds
            break
        seen.append(worst)
        delta = delta + step
        p, S, _ = _probs(xb, lev, live, st0, sizes, delta)
    p, S, f = F(delta)
    for _ in range(maxiter):
        pred, step = ratios(p)
        worst = np.max(np.abs(step), initial=0.0)
        if worst < tol:
            break
        g = np.where(ok, target - pred, 0.0)
        diag = np.bincount(lev, weights=np.where(ok_rows, p * (1 - p), 0.0), minlength=Lb)
        M = np.where(diag > 0, 1.0 / np.where(diag > 0, diag, 1.0), 0.0)
        ridge = 1e-10 * float(diag.max(initial=0.0))
        d = _cg(lambda v, p=p, r=ridge: hess(v, p) + r * v, g, M,
                tol=1e-3 * min(1.0, float(np.abs(g).max())))
        big = np.max(np.abs(d), initial=0.0)
        if big > MAX_STEP:                          # a trust region on the log scale
            d *= MAX_STEP / big
        t = 1.0
        while t > 1e-8:
            p2, S2, f2 = F(delta + t * d)
            # (near the solution F's change is below its rounding error: the
            # step is then judged by the largest log ratio)
            if f2 > f or np.max(np.abs(ratios(p2)[1]), initial=0.0) < worst:
                break
            t /= 2
        if t > 1e-8:
            delta, p, S, f = delta + t * d, p2, S2, f2
        else:                                       # the contraction step instead
            delta = delta + step
            p, S, f = F(delta)
    return delta, p, S, step


def _cg(A, b, M, tol):
    """Preconditioned conjugate gradients for A x = b (A positive
    semidefinite, b in its range), to a relative residual `tol`."""
    x = np.zeros_like(b)
    r = b.copy()
    z = M * r
    q = z.copy()
    rz = r @ z
    nb = np.linalg.norm(b)
    for _ in range(CG_MAXITER):
        if np.linalg.norm(r) <= tol * nb:
            break
        Aq = A(q)
        qAq = q @ Aq
        if qAq <= 0:
            break
        a = rz / qAq
        x += a * q
        r -= a * Aq
        z = M * r
        rz, old = r @ z, rz
        q = z + (rz / old) * q
    return x
