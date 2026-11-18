"""Constrained assignment (experimental): imputing workers' establishments
so that the establishments' imputed employment matches their reported
totals, exactly or approximately.

Workers choose among their firm's establishments by commuting distance and
each establishment's attractiveness. The model is fitted on firms whose
workers' establishments are known; in other firms only each
establishment's total employment is reported. Each imputation draws the
parameters, then each worker's establishment, four ways:

  calibrated  independent draws from the calibrated probabilities
              (predict(calibrate=)): the counts match the totals only on average;
  exact       SwapMCMC(): the model's distribution given the counts equal
              the totals;
  soft        SwapMCMC(penalty=GaussianPenalty(...)): the counts near the
              totals, for totals measured with error;
  balanced    Balanced(): the calibrated probabilities, with the counts
              equal to the totals (the cube method).

Twice: with the true counts reported, and with the reported totals off by
about 1 + 5%.

    python examples/constrained_assignment.py
"""

import warnings

import numpy as np
import polars as pl

import logit_stream
from logit_stream import clogit_stream
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

true_n = impute.group_by("estab").agg(pl.col("chosen").sum().cast(pl.Float64).alias("n"))
noise = rng.normal(size=true_n.height)
reported = true_n.with_columns(
    (pl.col("n") * (1 + 0.05 * pl.Series(noise)) + rng.normal(size=true_n.height))
    .round().clip(0).alias("n"))
newdata = impute.drop("chosen")
truth = impute.filter(pl.col("chosen") == 1).select("case", pl.col("estab").alias("true"))


def independent(fit_m, totals, seed):
    """One draw per worker from the calibrated probabilities: the row where
    the set's cumulative probability passes a uniform."""
    p = fit_m.predict(newdata, calibrate={"estab": totals}).lazy()
    cases = newdata.select(pl.col("case").unique(maintain_order=True))
    cases = cases.with_columns(pl.Series("u", np.random.default_rng(seed).random(cases.height)))
    prob = pl.col("prob").fill_null(0)
    cum = prob.cum_sum().over("case")
    return p.join(cases.lazy(), on="case").with_columns(
        ((cum - prob < pl.col("u") * prob.sum().over("case")) &
         (cum >= pl.col("u") * prob.sum().over("case")) & (prob > 0)).alias("assigned")) \
        .select("case", "estab", "dist", "assigned").collect()


def summary(draws, totals):
    """Over the imputations: the share of workers at their true
    establishment, the mean commute, and the establishments' mean absolute
    gap between imputed employment and the reported totals and the truth."""
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
for label, totals in [("totals = the true counts", true_n),
                      ("totals reported with error", reported)]:
    print(f"\n{label} (true mean commute {true_commute:.3f}); means over {M} imputations")
    print(f"{'':12s} {'P(true estab)':>14s} {'mean commute':>13s} "
          f"{'|n - total|':>12s} {'|n - true n|':>13s}")
    for name in ("calibrated", *SAMPLERS):
        draws = [draw(name, f, totals, m) for m, f in enumerate(params)]
        hit, commute, gap_total, gap_true = summary(draws, totals)
        print(f"{name:12s} {hit:14.3f} {commute:13.3f} {gap_total:12.2f} {gap_true:13.2f}")
