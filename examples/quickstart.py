"""Quickstart: every estimator takes a formula and a source, and returns a
result with the same reporting methods.

    python examples/quickstart.py
"""

from pathlib import Path

import polars as pl

from logit_stream import (
    clogit_stream,
    logit_stream,
    mixlogit_stream,
    mlogit_stream,
    nlogit_stream,
    ologit_stream,
)
from logit_stream.simulate import RACES, simulate_choices, simulate_people

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# A simulated population: people in PUMAs nested in states, with three
# demographic outcomes. Written to Parquet, which the estimators read in
# blocks; a glob ("people/*.parquet"), a LazyFrame or a DataFrame works too.
people = OUT / "people.parquet"
simulate_people(200_000, seed=1).write_parquet(people)

# ---- binary logit: hdfe-stream's feglm_stream, any number of fixed effects
fit = logit_stream("hispanic ~ age10 + female + log_income + owner + urban | puma",
                   people, vcov={"CRV1": "puma"}, verbose=False)
fit.summary()

# ---- multinomial logit: coefficients "<level>:<covariate>", first level the base
race = mlogit_stream("race ~ age10 + female + log_income + owner + urban + C(state)",
                     people, levels=RACES, verbose=False)
print(race.coef_matrix().head(6))

# ---- ordered logit: education is an Enum, so its levels are ordered
educ = ologit_stream("education ~ age10 + female + log_income + owner + urban | puma",
                     people, vcov="hetero", cluster="puma", verbose=False)
print("\ncutpoints:", educ.cutpoints.round(3))
print(educ.with_vcov({"CRV1": "puma"}).tidy())

# ---- conditional logit: long-format choice data, one row per alternative
choices = simulate_choices(20_000, seed=1)
print("\n", choices.head(6))
mode = clogit_stream("chosen ~ cost + time + C(alt) + income:C(alt)", choices,
                     case="case", verbose=False)
print(mode.tidy())

# ---- nested logit: the rail alternatives share a nest, and a dissimilarity
# parameter lambda (1 is the conditional logit)
car = (pl.col("income") * (pl.col("alt") == "car")).alias("income_car")
nests = {"rail": ["train", "metro"], "road": ["car"]}
nested = simulate_choices(20_000, nests=nests, lambdas=0.5, seed=1).with_columns(car)
nl = nlogit_stream("chosen ~ cost + time + C(alt) + income_car", nested, case="case",
                   alt="alt", nests=nests, verbose=False)
print(nl.tidy().select("Coefficient", "Estimate", "Std. Error"))

# ---- mixed logit: the time coefficient varies across choosers, each of whom
# makes ten choices
panel = simulate_choices(20_000, tasks=10, sd_time=1.0, seed=1).with_columns(car)
mx = mixlogit_stream("chosen ~ cost + time + C(alt) + income_car", panel, case="case",
                     panel="person", random={"time": "normal"}, draws=200,
                     vcov={"CRV1": "person"}, verbose=False)
print(mx.tidy().select("Coefficient", "Estimate", "Std. Error"))

# ---- reporting: hdfe-stream's methods, for every result
print(race.confint(joint=True, keep="female"))
print(race.wald_test(keep="log_income"))
race.summary_json(OUT / "race.json", coefficients=True)

# ---- predictions: lazy frames, computed when collected or sunk, so they
# work on data of any size
new = OUT / "new_people.parquet"
simulate_people(100_000, seed=2).write_parquet(new)
race.predict(new).sink_parquet(OUT / "race_probabilities.parquet")
print(pl.read_parquet(OUT / "race_probabilities.parquet")
        .select("person_id", pl.col("^prob_.*$")).head(3))
print(educ.predict(new, type="class").select("person_id", "education", "predicted").head(3)
          .collect())
print(mode.predict(choices).select("case", "alt", "chosen", "prob").head(6).collect())
