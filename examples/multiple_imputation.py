"""Multiple imputation with group effects: workers in firms in industry x
area cells, a binary characteristic known for some workers and imputed for
the rest, with a random firm intercept within cell fixed effects.

For each imputation m:
  1. draw the parameters (coefficients, cell effects, the firm sd) from
     their estimated sampling distribution;
  2. draw each firm's intercept from its distribution given the firm's
     known workers, at the drawn parameters (firms without known workers:
     from N(0, sd^2));
  3. impute the unknown workers from the drawn log-odds plus their firm's
     intercept, one per firm, shared by its unknown workers.

Steps 1 and 2 are `draw_parameters(M, ranef=True)`, which reads the known
workers once for all M imputations; step 3 is `predict` and a Polars
expression.

    python examples/multiple_imputation.py
"""

import numpy as np
import polars as pl
from scipy.special import expit

from logit_stream import logit_stream

# ---- simulated data: 20,000 firms in 50 cells; firm sizes 1 + floor(exp N(1, 1.1));
# age sorts across firms; firm effects with sd 0.8; 25% of workers unknown
rng = np.random.default_rng(0)
G, n_cells = 20_000, 50
cell = rng.integers(0, n_cells, G)
size = 1 + np.floor(np.exp(rng.normal(1.0, 1.1, G))).astype(int)
firm = np.repeat(np.arange(G), size)
n = len(firm)
age10 = 4.0 + 0.8 * rng.normal(size=G)[firm] + 0.9 * rng.normal(size=n)   # age / 10
x = rng.normal(size=n)
p = expit(rng.uniform(-2.5, 2.5, n_cells)[cell][firm] - 0.3 * (age10 - 4) + 0.5 * x
          + rng.normal(0, 0.8, G)[firm])
truth = (rng.random(n) < p).astype(int)
known = rng.random(n) >= 0.25
workers = pl.DataFrame({"worker": np.arange(n), "firm": firm, "cell": cell[firm],
                        "age10": age10, "x": x,
                        "female": pl.Series(np.where(known, truth, -1)).replace(-1, None)})
obs = workers.filter(pl.col("female").is_not_null())
unknown = workers.filter(pl.col("female").is_null())

# ---- the model, on the known workers
fit = logit_stream("female ~ age10 + x | cell", obs, random="firm",
                   random_method="adaptive", verbose=False)
fit.summary()
print(fit.fixef().head(3).collect())             # cell effects with their standard errors

# ---- the imputations: parameters and firm intercepts drawn together (one read
# of the known workers for all ten); `newdata` adds the firms without known workers
M = 10
imputed = []
for m, draw in enumerate(fit.draw_parameters(M, seed=1, ranef=True, newdata=unknown)):
    if m < 2:
        print(f"imputation {m}: sd {draw.random_effects['firm']['sd'][0]:.4f}, "
              f"age10 {draw.beta[0]:.4f}")
        print(draw.ranef().head(3).collect())
    # the probability, with the firm's drawn intercept, then a Bernoulli draw: a
    # uniform from a hash of the worker's id (a different seed per imputation)
    u = pl.col("worker").hash(seed=m).cast(pl.Float64) / 2.0**64
    imputed.append(draw.predict(unknown, type="response", name="p")
                   .select("worker", (u < pl.col("p")).cast(pl.Int8).alias(f"female_{m}")))
completed = pl.concat(imputed, how="align").collect()     # one column per imputation

# ---- checks against the truth (which a real application does not have)
Y = completed.drop("worker").to_numpy().T                 # M x unknown workers
w = completed["worker"].to_numpy()
share = Y.mean(axis=1)
print(f"share female among the unknown: truth {truth[w].mean():.4f}, "
      f"imputed {share.mean():.4f} (between-imputation sd {share.std(ddof=1):.4f})")
# firm by firm, among the unknown workers of firms with 10 or more known workers: the
# imputed share follows the true one, which the cell effects alone cannot
nk = np.bincount(firm[known], minlength=G)
sel = nk[firm[w]] >= 10
fm = firm[w][sel]
nm = np.bincount(fm, minlength=G)
ids = np.flatnonzero(nm)
imp_share = np.bincount(fm, Y[:, sel].mean(axis=0), G)[ids] / nm[ids]
true_share = np.bincount(fm, truth[w][sel], G)[ids] / nm[ids]
print(f"{len(ids):,} firms with 10+ known workers: correlation of the imputed and true shares "
      f"of their unknown workers {np.corrcoef(imp_share, true_share)[0, 1]:.3f}")
