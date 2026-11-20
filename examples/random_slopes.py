"""Random slopes in the binary logit: students in schools in districts,
where the effect of last year's score on proficiency differs between
schools. Each school gets a random intercept and a random slope in `prior`,
correlated, within district fixed effects; the test of whether the slope
varies at all; predictions for seen and unseen schools; and parameter draws
with each school's effects, for multiple imputation.

    python examples/random_slopes.py
"""

import time

import numpy as np
import polars as pl
from scipy.special import expit

from logit_stream import logit_stream, lr_test

# ---- simulated data: 1,500 schools in 30 districts, about 8 students each
# (1 + Poisson(7)); each school's intercept and slope in `prior` are drawn with
# sds 0.8 and 0.5 and correlation 0.3; `ses` and `prior` have fixed effects
# -0.3 and 0.7
rng = np.random.default_rng(0)
G, n_districts = 1_500, 30
sd0, sd1, rho = 0.8, 0.5, 0.3
district = rng.integers(0, n_districts, G)
size = 1 + rng.poisson(7, G)
school = np.repeat(np.arange(G), size)
n = len(school)
S = np.array([[sd0 ** 2, rho * sd0 * sd1], [rho * sd0 * sd1, sd1 ** 2]])
a = rng.multivariate_normal([0, 0], S, G)
ses = rng.normal(size=n)
prior = rng.normal(size=n)
p = expit(rng.uniform(-1, 1, n_districts)[district][school] - 0.3 * ses + 0.7 * prior
          + a[school, 0] + a[school, 1] * prior)
students = pl.DataFrame({"school": school, "district": district[school], "ses": ses,
                         "prior": prior, "proficient": (rng.random(n) < p).astype(int)})
print(f"{G:,} schools, {n:,} students")

# ---- the fit: a random intercept ("1") and slope in prior by school, by adaptive
# Gauss-Hermite quadrature in two dimensions
t = time.perf_counter()
fit = logit_stream("proficient ~ ses + prior | district", students,
                   random={"school": ["1", "prior"]}, verbose=False)
print(f"time: {time.perf_counter() - t:.1f} s")
fit.summary()
print(fit.varcorr())                         # true: sds 0.8 and 0.5, correlation 0.3

# ---- each school's predicted intercept and slope (the conditional modes), with
# their conditional sds; with eight students a school's slope is shrunk far toward 0
r = fit.ranef().collect()
print(r.head(4))
print("correlation of the predicted and true slopes:",
      f"{np.corrcoef(r.sort('school')['re_school:prior'], a[:, 1])[0, 1]:.2f}")

# ---- does the slope vary at all? Against the random intercept alone: the slope's
# variance is on the boundary under the null, so the reference is the 50:50
# mixture of chi-squared(1) and chi-squared(2) (Stram and Lee 1994)
intercept_only = logit_stream("proficient ~ ses + prior | district", students,
                              random="school", random_method="adaptive", verbose=False)
print(lr_test(intercept_only, fit, boundary=True))

# ---- predictions: a seen school gets its intercept and slope, an unseen one
# (school 99999) neither; effects="average" averages the probability over the
# effects' distribution instead (the school's conditional one, or the prior)
new = pl.DataFrame({"school": [0, 0, 99_999, 99_999], "district": [0] * 4,
                    "ses": [0.0] * 4, "prior": [-1.0, 1.0, -1.0, 1.0]})
print(fit.predict(new, type="response", name="conditional")
      .join(fit.predict(new, type="response", effects="average", name="averaged"),
            on=["school", "prior"], maintain_order="left")
      .select("school", "prior", "conditional", "averaged").collect())

# ---- for multiple imputation: draws of the coefficients, the district effects
# and Sigma, then of each school's intercept and slope given its students
draws = fit.draw_parameters(5, seed=1, ranef=True)
print("drawn slope sds:", np.round([d.random_effects["school"]["sd"][1] for d in draws], 3))
