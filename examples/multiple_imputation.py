"""Multiple imputation with group effects: students in schools in
districts, a binary characteristic known for some students and imputed for
the rest, with a random school intercept within district fixed effects.

For each imputation m:
  1. draw the parameters (coefficients, district effects, the school sd)
     from their estimated sampling distribution;
  2. draw each school's intercept from its distribution given the school's
     known students, at the drawn parameters (schools without known
     students: from N(0, sd^2));
  3. impute the unknown students from the drawn log-odds plus their
     school's intercept, one per school, shared by its unknown students.

Steps 1 and 2 are `draw_parameters(M, ranef=True)`, which reads the known
students once for all M imputations; step 3 is `predict(type="draw")`.

Then each completed data set is analyzed as if it were complete, and the M
analyses are combined by Rubin's rules: `pool` for scalar statistics (each
district's share proficient), `pool_matrix` for a regression's
coefficients, with a joint test.

    python examples/multiple_imputation.py
"""

import numpy as np
import polars as pl
from scipy.special import expit

from logit_stream import logit_stream, pool, pool_matrix

# ---- simulated data: 20,000 schools (a sample of students in each) in 50 districts;
# sample sizes 1 + floor(exp N(1, 1.1)); `ses`, a socioeconomic index, sorts across
# schools; `prior`, last year's standardized score; school effects with sd 0.8;
# the characteristic, `proficient`, missing for 25% of the students
rng = np.random.default_rng(0)
G, n_districts = 20_000, 50
district = rng.integers(0, n_districts, G)
size = 1 + np.floor(np.exp(rng.normal(1.0, 1.1, G))).astype(int)
school = np.repeat(np.arange(G), size)
n = len(school)
ses = 0.8 * rng.normal(size=G)[school] + 0.9 * rng.normal(size=n)
prior = rng.normal(size=n)
p = expit(rng.uniform(-2.5, 2.5, n_districts)[district][school] - 0.3 * ses + 0.5 * prior
          + rng.normal(0, 0.8, G)[school])
truth = (rng.random(n) < p).astype(int)
known = rng.random(n) >= 0.25
students = pl.DataFrame({"student": np.arange(n), "school": school,
                         "district": district[school], "ses": ses, "prior": prior,
                         "proficient": pl.Series(np.where(known, truth, -1)).replace(-1, None)})
obs = students.filter(pl.col("proficient").is_not_null())
unknown = students.filter(pl.col("proficient").is_null())

# ---- the model, on the known students
fit = logit_stream("proficient ~ ses + prior | district", obs, random="school",
                   random_method="adaptive", verbose=False)
fit.summary()
print(fit.fixef().head(3).collect())             # district effects with their standard errors

# ---- the imputations: parameters and school intercepts drawn together (one read
# of the known students for all ten); `newdata` adds the schools without known students
M = 10
imputed = []
for m, draw in enumerate(fit.draw_parameters(M, seed=1, ranef=True, newdata=unknown)):
    if m < 2:
        print(f"imputation {m}: sd {draw.random_effects['school']['sd'][0]:.4f}, "
              f"ses {draw.beta[0]:.4f}")
        print(draw.ranef().head(3).collect())
    # a draw from the probability, with the school's drawn intercept; the uniform
    # depends on the seed and the student's id, so the step is lazy and reproducible
    imputed.append(draw.predict(unknown, type="draw", seed=m, id="student", name="p")
                   .select("student", pl.col("p").alias(f"proficient_{m}")))
completed = pl.concat(imputed, how="align").collect()     # one column per imputation

# ---- checks against the truth (which a real application does not have)
Y = completed.drop("student").to_numpy().T                 # M x unknown students
w = completed["student"].to_numpy()
share = Y.mean(axis=1)
print(f"share proficient among the unknown: truth {truth[w].mean():.4f}, "
      f"imputed {share.mean():.4f} (between-imputation sd {share.std(ddof=1):.4f})")
# school by school, among the unknown students of schools with 10 or more known students:
# the imputed share follows the true one, which the district effects alone cannot
nk = np.bincount(school[known], minlength=G)
sel = nk[school[w]] >= 10
fm = school[w][sel]
nm = np.bincount(fm, minlength=G)
ids = np.flatnonzero(nm)
imp_share = np.bincount(fm, Y[:, sel].mean(axis=0), G)[ids] / nm[ids]
true_share = np.bincount(fm, truth[w][sel], G)[ids] / nm[ids]
print(f"{len(ids):,} schools with 10+ known students: correlation of the imputed and true shares "
      f"of their unknown students {np.corrcoef(imp_share, true_share)[0, 1]:.3f}")

# ---- analyses of the completed data, pooled by Rubin's rules
known_y = obs.select("student", "proficient")
completed_sets = [pl.concat([known_y, completed.select("student", pl.col(f"proficient_{m}")
                                                      .alias("proficient"))])
                  .join(students.drop("proficient"), on="student") for m in range(M)]

# scalar statistics: each district's share proficient, with its binomial standard
# error as if the data were complete; one row per imputation and district
shares = pl.concat([
    d.group_by("district").agg(pl.col("proficient").mean().alias("estimate"), pl.len().alias("n"))
     .with_columns((pl.col("estimate") * (1 - pl.col("estimate")) / pl.col("n")).sqrt()
                   .alias("se"), pl.lit(m).alias("imputation"))
    for m, d in enumerate(completed_sets)])
pooled = pool(shares, by="district").sort("district")
print(pooled.select("district", "estimate", "se", "ci_low", "ci_high", "fmi").head(3))
true_share = (students.with_columns(pl.Series("truth", truth)).group_by("district")
              .agg(pl.col("truth").mean()).sort("district"))
inside = ((pooled["ci_low"] <= true_share["truth"]) & (true_share["truth"] <= pooled["ci_high"]))
print(f"95% intervals covering the district's true share: {inside.sum()} of {len(inside)}; "
      f"mean fraction of missing information {pooled['fmi'].mean():.3f}")

# a vector: the coefficients of an analysis model fitted to each completed data set
# (the imputation model's form, so that the two agree on what is estimated)
ANALYSIS = dict(random="school", random_method="adaptive", verbose=False)
fits = [logit_stream("proficient ~ ses + prior | district", d, **ANALYSIS)
        for d in completed_sets]
coefs = pool_matrix(fits)
print(coefs.tidy().select("term", "estimate", "se", "df", "riv", "fmi"))
full = logit_stream("proficient ~ ses + prior | district",
                    students.with_columns(pl.Series("proficient", truth)), **ANALYSIS)
print(f"the complete data (unknown in practice): ses {full.beta[0]:.4f} (se {full.se[0]:.4f}), "
      f"prior {full.beta[1]:.4f} (se {full.se[1]:.4f})")
test = coefs.wald_test(["ses", "prior"], [-0.3, 0.5])      # jointly, the simulation's values
print(f"joint test of ses = -0.3 and prior = 0.5: F = {test['statistic']:.3f} on "
      f"({test['df1']}, {test['df2']:.0f}) df, p = {test['p']:.3f}")
