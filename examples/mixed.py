"""The mixed logit: random coefficients, panels of repeated choices, the
number of draws, and the test against the conditional logit.

    python examples/mixed.py
"""

import numpy as np
import polars as pl
from scipy.stats import chi2, norm

from logit_stream import clogit_stream, mixlogit_stream
from logit_stream.simulate import simulate_choices

# 4,000 people make five choices each. Each person's time coefficient is
# drawn once, normal with mean -1.3 and sd 1, and holds for all five.
data = (simulate_choices(20_000, tasks=5, sd_time=1.0, seed=3)
        # the nested and mixed logit refuse collinear covariates, so income
        # enters as one column for the car rather than as income:C(alt)
        .with_columns(income_car=pl.col("income") * (pl.col("alt") == "car")))
FML = "chosen ~ cost + time + C(alt) + income_car"

fit = mixlogit_stream(FML, data, case="case", panel="person", random={"time": "normal"},
                      draws=500, vcov={"CRV1": "person"}, verbose=False)
fit.summary()
print("signed sd:", fit.diagnostics["sd_signed"])

# ---- against the conditional logit (sd = 0). The sd is on the boundary
# under the null, so the chi-squared(1) p-value is conservative.
cl = clogit_stream(FML, data, case="case", vcov={"CRV1": "person"}, verbose=False)
lr = 2 * (fit.loglik - cl.loglik)
print(f"conditional logit {cl.loglik:.1f}, mixed {fit.loglik:.1f}: LR {lr:.1f}, "
      f"p = {chi2.sf(lr, 1):.2g}")
print(pl.DataFrame({"coefficient": cl.coefnames, "conditional": cl.beta,
                    "mixed": fit.beta[:len(cl.beta)]}))

# ---- the share of people for whom more time is a good thing
mu, sd = fit.coef()["time"], fit.coef()["sd.time"]
print(f"P(time coefficient > 0) = {norm.cdf(mu / sd):.3f} (simulated: {norm.cdf(-1.3):.3f})")

# ---- the panel matters: treating each choice as a different person
# (panel=None) asks a different question of the same data
alone = mixlogit_stream(FML, data, case="case", random={"time": "normal"}, draws=500,
                        verbose=False)
print(pl.DataFrame({"coefficient": fit.coefnames, "panel": fit.beta, "no panel": alone.beta}))

# ---- the number of draws: the simulated likelihood's error falls as the
# draws rise; Halton draws need far fewer than pseudo-random ones
rows = []
for draws in (50, 200, 500):
    for halton in (True, False):
        f = mixlogit_stream(FML, data, case="case", panel="person",
                            random={"time": "normal"}, draws=draws, halton=halton,
                            seed=7, verbose=False)
        rows.append({"draws": draws, "halton": halton, "time": f.coef()["time"],
                     "sd.time": f.coef()["sd.time"], "loglik": f.loglik,
                     "seconds": f.diagnostics["seconds_total"]})
print(pl.DataFrame(rows))

# ---- a coefficient that does not vary: cost's sd is 0 in the simulation.
# At sd = 0 the likelihood is flat in the sd to first order, so its estimate
# converges only at the rate n^(-1/4), and its standard error means little
# there (in other samples of this size the estimate ranges from 0 to about
# 0.3). The likelihood-ratio test, against half a chi-squared(1) for the
# boundary, is the one to use.
both = mixlogit_stream(FML, data, case="case", panel="person",
                       random={"time": "normal", "cost": "normal"}, draws=500,
                       verbose=False)
print(both.tidy().select("Coefficient", "Estimate", "Std. Error"))
lr = 2 * (both.loglik - fit.loglik)
print(f"LR for sd.cost = 0: {lr:.2f}, p = {0.5 * chi2.sf(lr, 1):.2f}")

# ---- a lognormal coefficient is positive: for a cost, use the negated column
neg = data.with_columns(neg_cost=-pl.col("cost"))
logn = mixlogit_stream("chosen ~ neg_cost + time + C(alt) + income_car", neg, case="case",
                       panel="person", random={"time": "normal", "neg_cost": "lognormal"},
                       draws=500, verbose=False)
print(logn.tidy().select("Coefficient", "Estimate", "Std. Error"))
m, v = logn.coef()["neg_cost"], logn.coef()["sd.neg_cost"]
print(f"cost coefficient: median {-np.exp(m):.3f}, mean {-np.exp(m + v ** 2 / 2):.3f} "
      "(simulated: -1.1 for everyone)")
