"""Random slopes by region in the conditional logit: workers choosing among
establishments, where the cost of distance differs between commuting zones.
Each zone gets its own deviation in the distance coefficient, with its sd
estimated, alongside absorbed establishment constants; the test of whether
the slope varies; predictions for a zone the fit did not see; and parameter
draws for multiple imputation.

    python examples/regional_slopes.py
"""

import time

import numpy as np
import polars as pl

from logit_stream import clogit_stream, lr_test

# ---- simulated data: 60 commuting zones, 2,000 establishments (each in a zone,
# with a constant drawn from N(0, 1)), 40,000 workers. A worker considers 5
# establishments in their home zone, at exponential distances; utility is
# (-1 + b_cz) dist + the establishment's constant + Gumbel noise, with the zones'
# b_cz drawn with sd 0.3
rng = np.random.default_rng(0)
R, E, W, J, sd = 60, 2_000, 40_000, 5, 0.3
b = rng.normal(0, sd, R)
estab_cz = rng.integers(0, R, E)
delta = rng.normal(0, 1, E)
by_cz = [np.flatnonzero(estab_cz == r) for r in range(R)]
home = rng.integers(0, R, W)
estab = np.concatenate([rng.choice(by_cz[h], J, replace=False) for h in home])
worker = np.repeat(np.arange(W), J)
cz = home[worker]
dist = rng.exponential(1.0, W * J)
u = (-1 + b[cz]) * dist + delta[estab] + rng.gumbel(size=W * J)
chosen = (u == np.repeat(u.reshape(W, J).max(1), J)).astype(int)
jobs = pl.DataFrame({"worker": worker, "estab": estab, "cz": cz, "dist": dist,
                     "chosen": chosen})
print(f"{W:,} workers, {jobs.height:,} rows; realized sd of the zones' slopes "
      f"{b.std(ddof=1):.3f}")

# ---- the fit: each zone's slope deviation integrated out by Laplace's
# approximation, the establishments' constants absorbed; and without the slopes
t = time.perf_counter()
fit = clogit_stream("chosen ~ dist | estab", jobs, case="worker", random={"cz": "dist"},
                    verbose=False)
t_slopes = time.perf_counter() - t
t = time.perf_counter()
plain = clogit_stream("chosen ~ dist | estab", jobs, case="worker", verbose=False)
print(f"time: {t_slopes:.1f} s with the slopes, {time.perf_counter() - t:.1f} s without")
fit.summary()
print(fit.varcorr())

# ---- each zone's predicted slope deviation against the truth: with about 670
# workers a zone, little shrinkage
r = fit.ranef().collect().sort("cz")
print(r.head(4))
print("correlation of the predicted and true deviations:",
      f"{np.corrcoef(r['re_cz:dist'], b)[0, 1]:.3f}")

# ---- does the slope vary at all? One variance on its boundary under the null:
# the 50:50 mixture of chi-squared(0) and chi-squared(1)
print(lr_test(plain, fit))

# ---- predictions for one worker in a zone the fit saw (0) and in one it did not
# (999): the seen zone's slope, or the mean slope; effects="average" averages
# the probabilities over the slope's distribution instead (here N(0, sd^2))
sets = pl.DataFrame({"worker": [0, 0, 1, 1], "estab": [int(by_cz[0][0]), int(by_cz[0][1])] * 2,
                     "cz": [0, 0, 999, 999], "dist": [0.2, 2.0] * 2})
print(fit.predict(sets, prefix="conditional")
      .join(fit.predict(sets, effects="average", prefix="averaged"),
            on=["worker", "estab"], maintain_order="left")
      .select("worker", "cz", "dist", "conditional", "averaged").collect())

# ---- for multiple imputation: joint draws of the coefficient, the constants and
# every zone's deviation, and the sd; a zone in newdata the fit did not see gets a
# draw from N(0, that draw's sd^2)
draws = fit.draw_parameters(5, seed=1, newdata=sets)
print("drawn sds:", np.round([float(np.sqrt(d.random_effects["cz"]["Sigma"]).item())
                              for d in draws], 3))
