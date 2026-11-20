"""Constrained assignment (experimental): imputing workers' establishments
so that the establishments' imputed employment matches their known
totals, exactly or approximately.

Workers choose among their firm's establishments by commuting distance and
each establishment's attractiveness. The model is fitted on firms whose
workers' establishments are known; in other firms only each
establishment's total employment is known. Each imputation draws the
parameters, then each worker's establishment, four ways:

  calibrated  independent draws from the calibrated probabilities
              (predict(type="draw", calibrate=)): the counts match the totals
              only on average;
  exact       SwapMCMC(): the model's distribution given the counts equal
              the totals;
  soft        SwapMCMC(penalty=GaussianPenalty(...)): the counts near the
              totals, for totals measured with error;
  balanced    Balanced(): the calibrated probabilities, with the counts
              equal to the totals (the cube method).

Twice: with the totals equal to the true counts, and with totals measured
with error, off by about 1 + 5%. Finally, each method's mean commute is pooled over the
imputations by Rubin's rules, as a statistic of the whole file (no sampling
variance), so that its standard error is the imputation's alone.

    python examples/constrained_assignment.py
"""

import warnings

import numpy as np
import polars as pl

import logit_stream
from logit_stream import clogit_stream, pool
from logit_stream.assign import Balanced, GaussianPenalty, SwapMCMC

warnings.filterwarnings("ignore", category=logit_stream.ExperimentalWarning)

rng = np.random.default_rng(11)


def firms(n, base):
    """n firms of 2 to 10 establishments and 5 to 150 workers; utility
    -1.2 distance + the establishment's constant + Gumbel noise."""
    out = []
    for f in range(n):
        ne, nw = int(rng.integers(2, 11)), int(rng.integers(5, 150))
        ex, ey = rng.normal(size=(2, ne)) * 1.5
        hx, hy = rng.normal(size=(2, nw)) * 1.5
        dist = np.hypot(ex[None] - hx[:, None], ey[None] - hy[:, None])
        u = -1.2 * dist + rng.normal(0, 1, ne)[None] + rng.gumbel(size=(nw, ne))
        chosen = (u == u.max(1, keepdims=True)).astype(np.int8)
        out.append(pl.DataFrame({"case": np.repeat(np.arange(nw) + (base + f) * 1000, ne),
                                 "estab": np.tile(np.arange(ne) + (base + f) * 100, nw),
                                 "firm": base + f, "dist": dist.ravel(),
                                 "chosen": chosen.ravel()}))
    return pl.concat(out)


train, impute = firms(2000, 0), firms(500, 10_000)
fit = clogit_stream("chosen ~ dist | estab", train, case="case", verbose=False)
print(f"distance coefficient {fit.beta[0]:.3f} (se {np.sqrt(fit.vcov[0, 0]):.3f}; truth -1.2)")

true_n = (impute.group_by("estab").agg(pl.col("chosen").sum().cast(pl.Float64).alias("n"))
          .sort("estab"))
noise = rng.normal(size=true_n.height)
noisy = true_n.with_columns(
    (pl.col("n") * (1 + 0.05 * pl.Series(noise)) + rng.normal(size=true_n.height))
    .round().clip(0).alias("n"))
newdata = impute.drop("chosen")
truth = impute.filter(pl.col("chosen") == 1).select("case", pl.col("estab").alias("true"))


def independent(fit_m, totals, seed):
    """One draw per worker from the calibrated probabilities, independently."""
    return fit_m.predict(newdata, type="draw", seed=seed, prefix="assigned",
                         calibrate={"estab": totals}).select(
        "case", "estab", "dist", pl.col("assigned").fill_null(False)).collect()


def summary(draws, totals):
    """Over the imputations: the share of workers at their true
    establishment, the mean commute, and the establishments' mean absolute
    gap between imputed employment and the totals and the truth."""
    rows = []
    for d in draws:
        a = d.filter("assigned")
        hit = a.join(truth, on="case").select((pl.col("estab") == pl.col("true")).mean()).item()
        cnt = a.group_by("estab").len().join(totals, on="estab", how="right") \
            .join(true_n.rename({"n": "true_n"}), on="estab").fill_null(0)
        rows.append((hit, a["dist"].mean(),
                     (cnt["len"] - cnt["n"]).abs().mean(),
                     (cnt["len"] - cnt["true_n"]).abs().mean()))
    return np.array(rows).mean(0)


SAMPLERS = {"exact": SwapMCMC(), "soft": SwapMCMC(penalty=GaussianPenalty(sd=1, rel=0.05)),
            "balanced": Balanced()}


def draw(name, fit_m, totals, m):
    """Imputation m's establishments, by `name`, under the drawn parameters."""
    if name == "calibrated":
        return independent(fit_m, totals, m)
    return fit_m.sample_assignments(newdata, {"estab": totals}, sampler=SAMPLERS[name],
                                    seed=m).collect()


M = 5
params = fit.draw_parameters(M, seed=1)
true_commute = impute.filter(pl.col("chosen") == 1)["dist"].mean()
commutes = []                 # each imputation's mean commute, for pooling
for label, totals in [("totals = the true counts", true_n),
                      ("totals measured with error", noisy)]:
    print(f"\n{label} (true mean commute {true_commute:.3f}); means over {M} imputations")
    print(f"{'':12s} {'P(true estab)':>14s} {'mean commute':>13s} "
          f"{'|n - total|':>12s} {'|n - true n|':>13s}")
    for name in ("calibrated", *SAMPLERS):
        draws = [draw(name, f, totals, m) for m, f in enumerate(params)]
        hit, commute, gap_total, gap_true = summary(draws, totals)
        print(f"{name:12s} {hit:14.3f} {commute:13.3f} {gap_total:12.2f} {gap_true:13.2f}")
        commutes += [{"totals": label, "method": name, "imputation": m,
                      "estimate": d.filter("assigned")["dist"].mean()}
                     for m, d in enumerate(draws)]

# the mean commute of the imputed file, pooled over the imputations as a statistic of
# the whole file: no sampling variance (se=None), so its standard error is the imputation's
pooled = pool(pl.DataFrame(commutes), se=None, by=["totals", "method"])
print(f"\nmean commute, pooled over {M} imputations (truth {true_commute:.3f})")
print(pooled.select("totals", "method", "estimate", "se", "ci_low", "ci_high"))
