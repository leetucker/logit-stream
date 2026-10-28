"""Binary outcomes: the logit and probit with fixed effects, and Firth's
penalized likelihood for separated or rare outcomes.

    python examples/binary.py
"""

import numpy as np
import polars as pl

from logit_stream import logit_stream, mlogit_stream, probit_stream
from logit_stream.simulate import simulate_people

people = simulate_people(200_000, seed=1)
X = "age10 + female + log_income + owner + urban"

# ---- logit and probit: hdfe-stream's feglm_stream, any number of fixed
# effects (here PUMAs, and the state's are absorbed by them)
logit = logit_stream(f"hispanic ~ {X} | puma", people, vcov={"CRV1": "puma"}, verbose=False)
probit = probit_stream(f"hispanic ~ {X} | puma", people, vcov={"CRV1": "puma"},
                       verbose=False)
print(pl.DataFrame({"coefficient": logit.coefnames, "logit": logit.beta,
                    "probit": probit.beta, "ratio": logit.beta / probit.beta}))
print(f"log-likelihood: logit {logit.loglik:.1f}, probit {probit.loglik:.1f}")
print(f"pseudo R2: {logit.pseudo_r2:.4f}, within the PUMAs: {logit.pseudo_r2_within:.4f}")

# PUMAs whose outcome never varies have infinite effects: dropped, and
# labelled in sample()
print(logit.sample().group_by("dropped_because").len().collect())

# fixed effects, and predicted probabilities for new data (lazily)
print(logit.fixef("puma").head(3).collect())
new = simulate_people(1_000, seed=2)
print(logit.predict(new, type="response").select("person_id", "puma", "prediction").head(3)
      .collect())

# ---- the same logit as a two-level multinomial logit: this package's own
# conventions (no pyfixest small-sample factor in the iid vcov, convergence
# on the Newton decrement), one fixed effect at most
two = mlogit_stream(f"hispanic ~ {X} | puma", people, levels=[0, 1],
                    vcov={"CRV1": "puma"}, verbose=False)
print("largest difference from logit_stream:",
      f"{np.max(np.abs(two.beta - logit.beta)):.1e}")

# ---- separation and Firth's penalized likelihood
# A small sample with a rare indicator that only ever has the outcome 0:
# its maximum-likelihood estimate is minus infinity.
rng = np.random.default_rng(3)
small = people.sample(400, seed=3).with_columns(
    rare=pl.Series(rng.random(400) < 0.03).cast(pl.Int8))
small = small.with_columns(hispanic=pl.when(pl.col("rare") == 1).then(0)
                           .otherwise(pl.col("hispanic")))
ml = mlogit_stream("hispanic ~ age10 + log_income + rare", small, levels=[0, 1],
                   verbose=False)
firth = mlogit_stream("hispanic ~ age10 + log_income + rare", small, levels=[0, 1],
                      firth=True, verbose=False)
print(pl.DataFrame({"coefficient": ml.coefnames, "ML": ml.beta, "ML se": ml.se,
                    "Firth": firth.beta, "Firth se": firth.se}))
