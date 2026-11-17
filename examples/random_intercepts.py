"""Random intercepts: groups' intercepts drawn from a normal distribution
whose covariance is estimated; shrinkage, unseen groups, nested groupings,
random intercepts within a fixed effect, the ordered logit's, and small
groups (Laplace's bias, and adaptive quadrature).

    python examples/random_intercepts.py
"""

import numpy as np
import polars as pl

from logit_stream import logit_stream, mlogit_stream, ologit_stream
from logit_stream.simulate import RACES, simulate_people

# 200 PUMAs in 20 states; the PUMA effects are independent, with sd 0.5 in
# every race equation (simulate_people's puma_sd)
people = simulate_people(100_000, n_states=20, pumas_per_state=10, seed=1)
X = "age10 + female + log_income + owner + urban"

re = mlogit_stream(f"race ~ {X}", people, levels=RACES, random="puma",
                   vcov={"CRV1": "puma"}, verbose=False)
re.summary()
print(re.varcorr())
print(np.round(re.random_effects["puma"]["Sigma"], 3))

# ---- against the fixed effect: similar coefficients, and the predicted
# random effects of small PUMAs shrunk toward zero
fe = mlogit_stream(f"race ~ {X} | puma", people, levels=RACES, verbose=False)
print(pl.DataFrame({"coefficient": fe.coefnames, "fixed": fe.beta,
                    "random": [re.coef()[n] for n in fe.coefnames]}).head(6))
r = re.ranef("puma").collect()
f = fe.fixef("puma").collect()
both = (r.select("puma", "n_obs", pl.col("re_puma:Black").alias("random"))
        .join(f.select("puma", pl.col("fe_puma:Black").alias("fixed")), on="puma")
        .with_columns(pl.col("fixed") - pl.col("fixed").mean()))     # FE: centred
print(both.sort("n_obs").head(3))
print(both.sort("n_obs").tail(3))

# ---- prediction: a PUMA the fit did not see gets the mean effect, zero
new = simulate_people(1_000, n_states=20, pumas_per_state=10, seed=2)
new = new.with_columns(pl.when(pl.col("person_id") < 3).then(9_999).otherwise("puma")
                       .alias("puma"))
p = re.predict(new).collect()
print(p.select("person_id", "puma", "prob_White", "prob_Black").head(4))

# ---- nested groupings: PUMAs within states, each with its own Sigma
ns = mlogit_stream(f"race ~ {X}", people, levels=RACES, random=["state", "puma"],
                   vcov={"CRV1": "state"}, verbose=False)
print(ns.varcorr().filter(pl.col("term").str.starts_with("sd")))

# ---- random PUMA intercepts within fixed state effects
within = mlogit_stream(f"race ~ {X} | state", people, levels=RACES, random="puma",
                       vcov={"CRV1": "state"}, verbose=False)
print(within.fixef().head(3).collect())
print(within.ranef("puma").head(3).collect())

# ---- Laplace (the default) and the faster joint-mode approximation
jm = mlogit_stream(f"race ~ {X}", people, levels=RACES, random="puma",
                   random_method="joint-mode", vcov={"CRV1": "puma"}, verbose=False)
d = (jm.beta[:len(re.beta)] - re.beta) / re.se
print(f"joint mode vs Laplace: largest difference {np.abs(d).max():.3f} standard errors; "
      f"{jm.diagnostics['seconds_total']:.1f} s vs {re.diagnostics['seconds_total']:.1f} s")

# ---- the ordered logit: one shift per PUMA
educ = ologit_stream(f"education ~ {X}", people, random="puma", verbose=False)
print(educ.varcorr())

# ---- small groups: firms with about 3 rows each, within 50 industry-area
# cells (absorbed); a binary outcome; firm effects with sd 0.8 (variance 0.64)
rng = np.random.default_rng(0)
G, cells = 30_000, 50
cell = rng.integers(0, cells, G)
size = 1 + rng.geometric(0.5, G)                         # 1 + Geometric: mean 3
firm = np.repeat(np.arange(G), size)
eta = rng.normal(0, 0.8, cells)[cell] + rng.normal(0, 0.8, G)
y = (rng.random(len(firm)) < 1 / (1 + np.exp(-eta[firm]))).astype(int)
firms = pl.DataFrame({"y": y, "firm": firm, "cell": cell[firm]})
for method in ["laplace", "adaptive"]:
    sm = logit_stream("y ~ 1 | cell", firms, random="firm", random_method=method,
                      verbose=False)
    info = sm.random_effects["firm"]
    print(f"{method:8s}: variance {info['sd'][0] ** 2:.3f} (sd {info['sd'][0]:.3f}, "
          f"se {info['sd_se'][0]:.3f}); {sm.diagnostics['seconds_total']:.1f} s")
print(sm.ranef("firm").collect().sort("n_obs").head(3))
