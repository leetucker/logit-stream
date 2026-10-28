"""Absorbed fixed effects in the multinomial and ordered logit: the same fit
as indicators, the effects themselves, separation, and several fixed
effects through the Poisson regression.

    python examples/fixed_effects.py
"""

import numpy as np
import polars as pl

from logit_stream import mlogit_stream, ologit_stream
from logit_stream.simulate import RACES, simulate_people

people = simulate_people(200_000, seed=1)
X = "age10 + female + log_income + owner + urban"

# ---- one fixed effect: each PUMA has an intercept for every race level
race = mlogit_stream(f"race ~ {X} | puma", people, levels=RACES, vcov={"CRV1": "puma"},
                     verbose=False)
race.summary()
print(race.fixef("puma").head(4).collect())      # one column per non-base level

# the ordered logit's is a shift of the latent index, with cut1 = 0
educ = ologit_stream(f"education ~ {X} | puma", people, verbose=False)
print("cutpoints:", educ.cutpoints.round(3))

# ---- the same model as indicators: identical estimates and standard
# errors, without the indicator columns (here 50 states x 5 levels)
absorbed = mlogit_stream(f"race ~ {X} | state", people, levels=RACES, vcov="hetero",
                         verbose=False)
dummies = mlogit_stream(f"race ~ {X} + C(state)", people, levels=RACES, vcov="hetero",
                        verbose=False)
idx = [dummies.coefnames.index(n) for n in absorbed.coefnames]
print("largest difference in estimates:", f"{np.abs(absorbed.beta - dummies.beta[idx]).max():.1e}",
      " in standard errors:", f"{np.abs(absorbed.se - dummies.se[idx]).max():.1e}")
print(f"time: absorbed {absorbed.diagnostics['seconds_total']:.1f} s, "
      f"indicators {dummies.diagnostics['seconds_total']:.1f} s")

# ---- separation: a PUMA where nobody has some race gets probability 0
# for it (its effect is minus infinity), shown as a null
fx = race.fixef("puma").collect()
print("PUMA-level cells with no one of that race:",
      fx.select(pl.sum_horizontal(pl.all().exclude("puma", "n_obs").is_null().cast(int)))
        .sum().item())
print(race.diagnostics["separation"])

# ---- several fixed effects: PUMA and single year of age. The model is fitted
# as the equivalent Poisson regression (hdfe-stream's fepois_stream) on the
# data expanded by outcome level
two = mlogit_stream("race ~ female + log_income + owner + urban | puma + age", people,
                    levels=RACES, vcov={"CRV1": "puma"}, verbose=False)
print(two.tidy().head(6))
print(two.fixef("age").head(3).collect())
print(f"time: {two.diagnostics['seconds_total']:.1f} s")
p = two.predict(simulate_people(1_000, seed=2)).collect()
print(p.select("person_id", pl.col("^prob_.*$")).head(3))
