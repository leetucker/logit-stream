"""The nested logit: nests of alternatives that are closer substitutes, the
dissimilarity parameters, a test against the conditional logit, deeper
trees, predictions and substitution.

    python examples/nested.py
"""

import polars as pl
from scipy.stats import chi2

from logit_stream import clogit_stream, nlogit_stream
from logit_stream.simulate import simulate_choices

# Mode-choice data drawn from a nested logit: train and metro share a nest
# with dissimilarity parameter 0.5 (their unobserved utilities correlated,
# 1 - 0.5^2 = 0.75), the car is a nest of its own.
nests = {"rail": ["train", "metro"], "road": ["car"]}
data = (simulate_choices(50_000, nests=nests, lambdas=0.5, seed=1)
        # the chooser's income affects the car only; a variable constant within
        # a set has to be interacted with the alternatives, and the nested
        # logit refuses collinear covariates, so make the one column
        .with_columns(income_car=pl.col("income") * (pl.col("alt") == "car")))
FML = "chosen ~ cost + time + C(alt) + income_car"

fit = nlogit_stream(FML, data, case="case", alt="alt", nests=nests, vcov="hetero",
                    verbose=False)
fit.summary()

# ---- against the conditional logit, which is lambda = 1: the likelihood-
# ratio test, and the Wald test of lambda.rail = 1
cl = clogit_stream(FML, data, case="case", verbose=False)
lr = 2 * (fit.loglik - cl.loglik)
print(f"conditional logit {cl.loglik:.1f}, nested {fit.loglik:.1f}: LR {lr:.1f}, "
      f"p = {chi2.sf(lr, 1):.2g}")
print(fit.wald_test(keep="lambda.rail", q=1.0))
print(pl.DataFrame({"coefficient": cl.coefnames, "conditional": cl.beta,
                    "nested": fit.beta[:len(cl.beta)]}))

# ---- substitution: raise metro fares and see where its riders go. The
# conditional logit sends them in proportion to the other shares (IIA); the
# nested logit sends more of them to the train, its nest-mate.
dearer = data.with_columns(
    cost=pl.when(pl.col("alt") == "metro").then(1.5 * pl.col("cost")).otherwise("cost"))


def shares(model, frame):
    return (model.predict(frame).group_by("alt").agg(pl.col("prob").sum())
            .with_columns(pl.col("prob") / pl.col("prob").sum()).collect())


rows = []
for name, model in (("conditional", cl), ("nested", fit)):
    now, then = shares(model, data), shares(model, dearer)
    d = now.join(then, on="alt", suffix="_after").with_columns(
        change=pl.col("prob_after") - pl.col("prob"))
    lost = -d.filter(pl.col("alt") == "metro")["change"].item()
    for alt in ("train", "car"):
        gain = d.filter(pl.col("alt") == alt)["change"].item()
        rows.append({"model": name, "to": alt, "share of metro's loss": gain / lost})
print(pl.DataFrame(rows).pivot(on="to", index="model", values="share of metro's loss"))

# ---- a deeper tree: nests of nests. Four alternatives, public transport
# {rail {train, metro}, bus} against the car, with lambda 0.4 in the rail
# nest and 0.7 in the public one.
tree = {"public": {"rail": ["train", "metro"], "bus": ["bus"]}, "private": ["car"]}
four = (simulate_choices(50_000, alternatives=("train", "metro", "bus", "car"), nests=tree,
                         lambdas={"public": 0.7, "rail": 0.4}, seed=2)
        .with_columns(income_car=pl.col("income") * (pl.col("alt") == "car")))
deep = nlogit_stream(FML, four, case="case", alt="alt",
                     nests=tree, verbose=False)
print(deep.tidy().select("Coefficient", "Estimate", "Std. Error"))
# the two-level tree that puts all public transport in one nest is the
# deeper one with lambda.rail = lambda.public: rejected
flat = nlogit_stream(FML, four, case="case", alt="alt",
                     nests={"public": ["train", "metro", "bus"], "private": ["car"]},
                     verbose=False)
lr = 2 * (deep.loglik - flat.loglik)
print(f"log-likelihoods: two levels {flat.loglik:.1f}, three {deep.loglik:.1f}: "
      f"LR {lr:.1f}, p = {chi2.sf(lr, 1):.2g}")

# ---- clustering beyond the choice set, and weights constant within sets
panel = (simulate_choices(50_000, nests=nests, lambdas=0.5, tasks=5, seed=3)
         .with_columns(income_car=pl.col("income") * (pl.col("alt") == "car"),
                       w=1.0 + (pl.col("person") % 3)))
cl_fit = nlogit_stream(FML, panel, case="case", alt="alt", nests=nests,
                       vcov={"CRV1": "person"}, weights="w", verbose=False)
print(cl_fit.tidy().select("Coefficient", "Estimate", "Std. Error"))
