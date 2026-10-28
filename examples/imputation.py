"""Imputation at scale: fit on a survey where the characteristic is
observed, predict it as probabilities for a larger file where it is not,
fill in for groups the fit did not see, check calibration, and combine the
predictions with another source of information by Bayes' rule.

    python examples/imputation.py
"""

from pathlib import Path

import numpy as np
import polars as pl

from logit_stream import mlogit_stream
from logit_stream.simulate import RACES, simulate_people

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# The survey: 200,000 people with survey weights, race observed. One PUMA in
# fifty is missing from it, as small areas are from real samples.
survey = simulate_people(200_000, seed=1).filter(pl.col("puma") % 50 != 0)
# The records to impute: a million people, written to Parquet (in practice
# much larger, and never loaded whole). Their race is in the file only so
# that the example can check the imputations against it.
records = OUT / "records.parquet"
simulate_people(1_000_000, seed=2).write_parquet(records)

X = "age10 + female + log_income + owner + urban"
fit = mlogit_stream(f"race ~ {X} | puma", survey, levels=RACES, weights="weight",
                    vcov={"CRV1": "puma"}, verbose=False)
# a model without the PUMA effects, for the records in PUMAs the fit lacks
fallback = mlogit_stream(f"race ~ {X} + C(state)", survey, levels=RACES, weights="weight",
                         verbose=False)

# ---- predict, lazily, batch by batch, straight to Parquet
probs = [f"prob_{r}" for r in RACES]
main = fit.predict(pl.scan_parquet(records)).select("person_id", *probs)
back = (fallback.predict(pl.scan_parquet(records))
        .select("person_id", *[pl.col(c).alias(f"{c}_fb") for c in probs]))
(main.join(back, on="person_id", how="left", maintain_order="left")
     .select("person_id",
             pl.any_horizontal(pl.col(probs).is_null()).alias("fallback"),
             *[pl.coalesce(c, f"{c}_fb").alias(c) for c in probs])
     .sink_parquet(OUT / "race_probabilities.parquet"))

fe_probs = pl.read_parquet(OUT / "race_probabilities.parquet")
print(fe_probs.head(4))
print(f"{fe_probs['fallback'].sum():,} records in PUMAs the fit did not see")

# ---- the same with random PUMA intercepts: an unseen PUMA gets the mean
# effect, zero, and small PUMAs' effects are shrunk toward it
re = mlogit_stream(f"race ~ {X}", survey, levels=RACES, weights="weight", random="puma",
                   vcov={"CRV1": "puma"}, verbose=False)
re.predict(pl.scan_parquet(records)).select("person_id", *probs).sink_parquet(
    OUT / "race_probabilities_re.parquet")
imputed = pl.read_parquet(OUT / "race_probabilities_re.parquet")

# ---- calibration: the mean probabilities against the shares
truth = pl.read_parquet(records, columns=["person_id", "race"])
print(pl.DataFrame({"level": RACES,
                    "fixed": [fe_probs[c].mean() for c in probs],
                    "random": [imputed[c].mean() for c in probs],
                    "share": [(truth["race"] == r).mean() for r in RACES]}))


def scores(frame):
    """The mean log probability of each record's true level, how often it
    is the most probable level, and how often its probability is 0."""
    long = (frame.unpivot(index="person_id", on=probs, variable_name="race", value_name="p")
            .with_columns(pl.col("race").str.strip_prefix("prob_")))
    return _scores(long, "p")


def _scores(long, col):
    s = long.join(truth.rename({"race": "true"}), on="person_id")
    top = s.sort(col, descending=True).group_by("person_id").first()
    mine = s.filter(pl.col("race") == pl.col("true"))[col]
    return {"mean log probability": mine.log().mean(),
            "most probable is right": (top["race"] == top["true"]).mean(),
            "probability 0": (mine == 0).mean()}


# The fixed effects give a probability of exactly 0 to a level that a PUMA's
# survey sample does not have (its effect is minus infinity); some records
# in those PUMAs have it, so the mean log probability is minus infinity.
print(pl.DataFrame([{"model": "fixed effects", **scores(fe_probs)},
                    {"model": "random intercepts", **scores(imputed)}]))

# ---- combining with another source of information
# Suppose a second source, independent of this one, gives each record a
# signal s (here a category A-D) whose distribution depends on race, with
# known probabilities P(s | race) -- from a validation study, say. If s is
# independent of the model's covariates given race, Bayes' rule gives the
# posterior P(race | x, s), proportional to P(race | x) P(s | race).
LIKELIHOOD = pl.DataFrame({          # P(s | race): each race's column sums to 1
    "signal": ["A", "B", "C", "D"],
    "White": [0.70, 0.10, 0.10, 0.10],
    "Black": [0.15, 0.65, 0.10, 0.10],
    "Hispanic": [0.15, 0.10, 0.65, 0.10],
    "Asian": [0.10, 0.10, 0.10, 0.70],
    "AIAN": [0.40, 0.30, 0.20, 0.10],
    "Multiracial": [0.40, 0.25, 0.20, 0.15],
})
lik = LIKELIHOOD.unpivot(index="signal", variable_name="race", value_name="lik")

# draw each record's signal from its true race (the example's stand-in for
# the second source)
rng = np.random.default_rng(3)
cdf = LIKELIHOOD.select(RACES).to_numpy().cumsum(axis=0)
code = truth["race"].replace_strict(RACES, list(range(len(RACES)))).to_numpy()
u = rng.random(len(code))
signal = np.minimum((cdf[:, code] < u).sum(axis=0), 3)
evidence = truth.select("person_id").with_columns(
    signal=pl.Series(np.array(["A", "B", "C", "D"])[signal]))

posterior = (imputed.unpivot(index="person_id", on=probs, variable_name="race",
                             value_name="prior")
             .with_columns(pl.col("race").str.strip_prefix("prob_"))
             .join(evidence, on="person_id")
             .join(lik, on=["signal", "race"])
             .with_columns(post=pl.col("prior") * pl.col("lik"))
             .with_columns(pl.col("post") / pl.col("post").sum().over("person_id")))


print(pl.DataFrame([{"model": "random intercepts", **_scores(posterior, "prior")},
                    {"model": "and the signal", **_scores(posterior, "post")}]))

# the posterior stays calibrated: mean posterior probability against shares
print(posterior.group_by("race").agg(pl.col("post").mean()).join(
    truth.group_by("race").agg(share=pl.len() / truth.height), on="race").sort("race"))
