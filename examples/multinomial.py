"""The multinomial logit: parameterization, standard errors, weights,
penalized fits, collinearity and separation.

    python examples/multinomial.py
"""

import polars as pl

from logit_stream import etable, mlogit_stream
from logit_stream.simulate import RACES, simulate_people, truth_people

people = simulate_people(200_000, seed=1)
X = "age10 + female + log_income + owner + urban"

# ---- coefficients: "<level>:<covariate>", each level against the base
# (the first of levels=); the formula's covariates enter every equation
fit = mlogit_stream(f"race ~ {X}", people, levels=RACES, vcov="hetero",
                    cluster=["puma", "state"], verbose=False)
print(fit.tidy().head(7))
print(fit.coef_matrix())

# against the coefficients simulate_people draws from (it adds PUMA effects,
# which the model omits, so these are not exactly the truth's)
truth = truth_people()["race"]
print({k: (round(fit.coef()[k], 3), v) for k, v in truth.items() if k.endswith("urban")})

# ---- standard errors: one fit computes the robust vcov and the clusterings
# asked for in cluster=; with_vcov switches between them
ses = pl.DataFrame({
    "coefficient": fit.coefnames,
    "iid": fit.with_vcov("iid").se,
    "hetero": fit.se,
    "CRV1 puma": fit.with_vcov({"CRV1": "puma"}).se,
    "CRV1 state": fit.with_vcov({"CRV1": "state"}).se,
})
print(ses.filter(pl.col("coefficient").str.starts_with("Black:")))

# multi-way clustering (inclusion-exclusion) and the cluster jackknife
two_way = mlogit_stream(f"race ~ {X}", people, levels=RACES,
                        vcov={"CRV1": "state+age"}, verbose=False)
jack = mlogit_stream(f"race ~ {X}", people, levels=RACES, vcov={"CRV3": "state"},
                     verbose=False)
print(pl.DataFrame({"coefficient": fit.coefnames[:6],
                    "CRV1 state": fit.with_vcov({"CRV1": "state"}).se[:6],
                    "CRV1 state+age": two_way.se[:6], "CRV3 state": jack.se[:6]}))

# ---- weights: analytic (rescaled to average 1) or frequency
w = mlogit_stream(f"race ~ {X}", people, levels=RACES, weights="weight",
                  vcov="hetero", verbose=False)
print(f"analytic weights: n_obs {w.n_obs:,.0f}, Black:urban {w.coef()['Black:urban']:.4f} "
      f"(se {w.se[w.coefnames.index('Black:urban')]:.4f})")
counts = (people.with_columns(n=pl.lit(1)).group_by("race", "female", "urban", "owner")
          .agg(pl.col("n").sum()))
fw = mlogit_stream("race ~ female + urban + owner", counts, levels=RACES, weights="n",
                   weights_type="fweights", verbose=False)
rows = mlogit_stream("race ~ female + urban + owner", people, levels=RACES, verbose=False)
print(f"frequency weights: {counts.height} rows, n_obs {fw.n_obs:,.0f}; largest "
      f"difference from the row-level fit {abs(fw.beta - rows.beta).max():.1e} "
      f"(se {abs(fw.se - rows.se).max():.1e})")

# ---- several models in one call, and a regression table
multi = mlogit_stream("race ~ csw(age10 + female, log_income, owner + urban)", people,
                      levels=RACES, verbose=False)
etable(list(multi.all_fitted_models.values()), type="md", keep=["Black:"])  # prints

# ---- a ridge penalty, and collinear covariates
ridge = mlogit_stream(f"race ~ {X}", people, levels=RACES, l2=100.0, verbose=False)
print("ridge vs ML, Black:urban:", round(ridge.coef()["Black:urban"], 4),
      round(fit.coef()["Black:urban"], 4))
dup = people.with_columns(urban2=2 * pl.col("urban"))
col = mlogit_stream(f"race ~ {X} + urban2", dup, levels=RACES, verbose=False)
print("collinear, dropped:", col.collin_vars)
