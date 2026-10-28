"""The conditional (McFadden) logit: long-format choice data, alternative-
specific constants, choosers' characteristics, unsorted rows, predictions,
and the multinomial logit as a conditional logit.

    python examples/conditional.py
"""

import numpy as np
import polars as pl

from logit_stream import clogit_stream, mlogit_stream
from logit_stream.simulate import RACES, simulate_choices, simulate_people

# One row per available alternative of each choice set: cost and time vary
# by alternative, income by chooser; 15% of the metro and car options are
# unavailable, so the sets have one, two or three rows (a set of one says
# nothing about the coefficients, and is harmless).
choices = simulate_choices(50_000, seed=1)
print(choices.head(5))
print(choices.group_by("case").agg(size=pl.len()).group_by("size").agg(sets=pl.len())
      .sort("size"))

# ---- the fit: C(alt) gives alternative-specific constants (car the base);
# income, constant within a set, enters interacted with the alternatives,
# one of which is collinear and dropped (its coefficient fixed at 0)
fit = clogit_stream("chosen ~ cost + time + C(alt) + income:C(alt)", choices, case="case",
                    vcov="hetero", verbose=False)
fit.summary()
print("dropped:", fit.collin_vars)
print(f"log-likelihood {fit.loglik:.1f}, equal shares {fit.loglik_null:.1f}")

# the willingness to pay for time: the ratio of the time and cost coefficients
b = fit.coef()
print(f"value of time: {b['time'] / b['cost']:.3f} (simulated: {-1.3 / -1.1:.3f})")

# ---- unsorted rows: pass 0 notices and sorts them by set, out of core
shuffled = choices.sample(fraction=1.0, shuffle=True, seed=2)
again = clogit_stream("chosen ~ cost + time + C(alt) + income:C(alt)", shuffled,
                      case="case", vcov="hetero", verbose=False)
print("shuffled rows, same estimates:", bool(np.abs(again.beta - fit.beta).max() < 1e-10))

# ---- clustering: repeated choices by the same person are not independent
panel = simulate_choices(50_000, tasks=5, seed=1)
by_person = clogit_stream("chosen ~ cost + time + C(alt) + income:C(alt)", panel,
                          case="case", vcov={"CRV1": "person"}, verbose=False)
print(by_person.tidy().select("Coefficient", "Estimate", "Std. Error"))

# ---- predictions: each alternative's probability within its set
p = fit.predict(choices).collect()
print(p.select("case", "alt", "chosen", "prob").head(5))
print(fit.resid().head(3).collect())

# a policy scenario: metro fares up 20%
dearer = choices.with_columns(
    cost=pl.when(pl.col("alt") == "metro").then(1.2 * pl.col("cost")).otherwise("cost"))
shares = (pl.concat([fit.predict(choices).with_columns(scenario=pl.lit("now")),
                     fit.predict(dearer).with_columns(scenario=pl.lit("fares +20%"))])
          .group_by("scenario", "alt").agg(pl.col("prob").sum())
          .with_columns(share=pl.col("prob") / pl.col("prob").sum().over("scenario"))
          .collect().pivot(on="scenario", index="alt", values="share")
          .select("alt", "now", "fares +20%").sort("alt"))
print(shares)

# ---- the multinomial logit is a conditional logit with alternative-specific
# constants and every covariate interacted with the alternatives
people = simulate_people(20_000, seed=1)
mnl = mlogit_stream("race ~ age10 + female + log_income", people, levels=RACES,
                    verbose=False)
long = (people.select("person_id", "race", "age10", "female", "log_income")
        .join(pl.DataFrame({"alt": RACES}), how="cross")
        .with_columns(chosen=(pl.col("race") == pl.col("alt")).cast(pl.Int8),
                      alt=pl.col("alt").cast(pl.Enum(RACES))))
cl = clogit_stream("chosen ~ C(alt) + age10:C(alt) + female:C(alt) + log_income:C(alt)",
                   long, case="person_id", verbose=False)
print(f"log-likelihoods: multinomial {mnl.loglik:.4f}, conditional {cl.loglik:.4f}")
print("Black:age10", round(mnl.coef()["Black:age10"], 6),
      round(cl.coef()["age10:C(alt)[Black]"], 6))
