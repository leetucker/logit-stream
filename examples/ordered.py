"""The ordered logit and probit: cutpoints, residuals, predictions and a
fixed effect.

    python examples/ordered.py
"""

import polars as pl

from logit_stream import ologit_stream, oprobit_stream
from logit_stream.simulate import EDUCATION, simulate_people, truth_people

people = simulate_people(200_000, seed=1)
X = "age10 + female + log_income + owner + urban"

# education is a Polars Enum, so its levels are in the Enum's order
print(people["education"].dtype)

# ---- the ordered logit: slopes, then the cutpoints cut1 ... cut4
fit = ologit_stream(f"education ~ {X}", people, vcov="hetero", verbose=False)
print(fit.tidy())
print("cutpoints:", fit.cutpoints.round(3))

# simulate_people centres log income at 10.4 and adds PUMA shifts, which
# this model omits; the slopes are close to what it draws from
truth = truth_people()["education"]
print({k: (round(fit.coef()[k], 3), v) for k, v in truth.items() if not k.startswith("cut")})

# ---- a string outcome needs levels= (sorted alphabetically otherwise)
strings = people.with_columns(pl.col("education").cast(pl.String))
again = ologit_stream(f"education ~ {X}", strings, levels=EDUCATION, verbose=False)
print("same fit from strings with levels=:", bool(abs(again.beta - fit.beta).max() < 1e-10))

# ---- residuals: each level's probability, the probability-scale residual
# and the generalized residual
r = fit.resid().collect()
print(r.select("row_id", "education", "prob_High school", "resid_prob", "resid_gen").head(4))
print(r.select(pl.col("resid_prob", "resid_gen").mean()))

# ---- predictions for new data
new = simulate_people(1_000, seed=2)
p = fit.predict(new).collect()
print(p.select("person_id", pl.col("^prob_.*$")).head(3))
print(fit.predict(new, type="class").select("person_id", "education", "predicted")
         .head(3).collect())
print(fit.predict(new, type="link").select("person_id", "xb").head(3).collect())

# ---- the ordered probit: the same model with normal errors; coefficients
# about 1/1.7 of the logit's
probit = oprobit_stream(f"education ~ {X}", people, verbose=False)
print(pl.DataFrame({"coefficient": fit.coefnames, "logit": fit.beta, "probit": probit.beta,
                    "ratio": fit.beta / probit.beta}))
print(f"log-likelihood: logit {fit.loglik:.1f}, probit {probit.loglik:.1f}")

# the two residuals are the same for the logit (both are F(cut_{j-1} - xb) +
# F(cut_j - xb) - 1), not for the probit
for name, res in (("logit", r), ("probit", probit.resid().collect())):
    d = (res["resid_prob"] - res["resid_gen"]).abs().max()
    print(f"{name}: largest |resid_prob - resid_gen| {d:.2e}")

# ---- an absorbed fixed effect: each PUMA shifts the latent index, and
# cut1 is normalized to 0
fe = ologit_stream(f"education ~ {X} | puma", people, vcov={"CRV1": "puma"},
                   verbose=False)
print(fe.tidy())
print(fe.fixef().head(3).collect())
