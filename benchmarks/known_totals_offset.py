"""A proxy for the alternatives' constants against estimating them: the
table in docs/imputation/known_totals.md ("Why estimate the constants").

Workers choose among their firm's establishments with utility
-1.2 distance + the establishment's constant + Gumbel noise. Two models are
fitted on 3,000 firms whose workers' establishments are known, and impute
1,000 others where only each establishment's total is known:

  log-share offset         the log of each establishment's share of its
                           firm's workers (with half a worker added, so
                           that no share is 0) as an offset, in the fit
                           and at imputation;
  constants + calibration  the constants absorbed in the fit, then solved
                           from the totals at imputation
                           (predict(calibrate=)).

For each: the distance coefficient; the expected mean commute; the mean
probability of each worker's true establishment; and the share of workers
misallocated against the totals (the sum over establishments of
|expected count - total|, over the number of workers). Takes a few
seconds. The absorbed fit warns that it drops the establishments no
training worker chose (their constants are minus infinity); the offset
model keeps them, with half a worker each.

    python benchmarks/known_totals_offset.py
"""

import warnings

import numpy as np
import polars as pl

from logit_stream import clogit_stream

warnings.filterwarnings("ignore", message="calibrate:")
rng = np.random.default_rng(11)


def firms(n, base):
    """n firms of 2 to 10 establishments and 5 to 150 workers."""
    out = []
    for f in range(n):
        ne, nw = int(rng.integers(2, 11)), int(rng.integers(5, 150))
        ex, ey = rng.normal(size=(2, ne)) * 1.5
        delta = rng.normal(0, 1, ne)
        hx, hy = rng.normal(size=(2, nw)) * 1.5
        dist = np.hypot(ex[None] - hx[:, None], ey[None] - hy[:, None])
        u = -1.2 * dist + delta[None] + rng.gumbel(size=(nw, ne))
        ch = (u == u.max(1, keepdims=True)).astype(np.int8)
        out.append(pl.DataFrame({"case": np.repeat(np.arange(nw) + (base + f) * 1000, ne),
                                 "estab": np.tile(np.arange(ne) + (base + f) * 100, nw),
                                 "firm": base + f, "dist": dist.ravel(), "chosen": ch.ravel()}))
    return pl.concat(out)


def with_share(df):
    """`df` with the log of each establishment's share of its firm's
    workers (half a worker added to each count)."""
    sh = df.group_by("estab").agg(pl.col("chosen").sum().alias("n")).join(
        df.select("estab", "firm").unique(), on="estab")
    sh = sh.with_columns((pl.col("n") + 0.5) / (pl.col("n") + 0.5).sum().over("firm"))
    return df.join(sh.select("estab", pl.col("n").log().alias("lshare")), on="estab")


train, test = firms(3000, 0), firms(1000, 10_000)
offset = clogit_stream("chosen ~ dist", with_share(train), case="case", offset="lshare",
                       verbose=False)
absorbed = clogit_stream("chosen ~ dist | estab", train, case="case", verbose=False)
totals = test.group_by("estab").agg(pl.col("chosen").sum().cast(pl.Float64).alias("n"))
preds = {"log-share offset": (offset, offset.predict(with_share(test)).collect()),
         "constants + calibration": (absorbed, absorbed.predict(
             test.drop("chosen"), calibrate={"estab": totals}).collect())}

n_workers = test["case"].n_unique()
true_commute = test.filter(pl.col("chosen") == 1)["dist"].mean()
print(f"{'':24s} {'distance':>9s} {'mean commute':>13s} {'P(true estab)':>14s} "
      f"{'misallocated':>13s}")
print(f"{'truth':24s} {-1.2:9.3f} {true_commute:13.3f}")
for name, (fit, p) in preds.items():
    prob = pl.col("prob").fill_null(0)
    commute = p.select((prob * pl.col("dist")).sum()).item() / n_workers
    hit = p.join(test.select("case", "estab", "chosen"), on=["case", "estab"]) \
        .filter(pl.col("chosen") == 1).select(prob.mean()).item()
    mis = (p.group_by("estab").agg(prob.sum()).join(totals, on="estab")
           .select((pl.col("prob") - pl.col("n")).abs().sum()).item()) / n_workers
    print(f"{name:24s} {fit.beta[0]:9.3f} {commute:13.3f} {hit:14.3f} {mis:13.1%}")
