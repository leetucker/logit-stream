# Multiple imputation

{download}`Download the script: multiple_imputation.py <../../examples/multiple_imputation.py>`

[Predicting at scale](predicting.md) predicts a characteristic as
probabilities at the estimates. Multiple imputation (Rubin 1987) instead
fills in the missing values several times, so that analyses of the
completed data can carry the uncertainty of the imputation. For this to
work, each imputation must use parameters drawn from their sampling
distribution, not the estimates alone ("proper" imputation). With group
effects, it must also use one draw of each group's intercept, shared by
the group's missing members, so that the imputed values keep the
within-group correlation that the model implies.

The script's setting is students in schools, and schools in districts.
A binary characteristic, whether a student is proficient on a test, is
known for three-quarters of the students in a sample from each school.
The covariates are a socioeconomic index, which varies across schools as
well as within them, and the student's score the year before. The model is
a random school intercept within district fixed effects, fitted on the
known students:

```python
fit = logit_stream("proficient ~ ses + prior | district", obs, random="school",
                   random_method="adaptive")
fit.fixef().head(3).collect()
```

```text
┌──────────┬─────────────┬─────────────┬───────┐
│ district ┆ fe_district ┆ se_district ┆ n_obs │
╞══════════╪═════════════╪═════════════╪═══════╡
│ 0        ┆ -2.106585   ┆ 0.093696    ┆ 1705  │
│ 1        ┆ -1.729583   ┆ 0.084462    ┆ 1795  │
│ 2        ┆ -2.337273   ┆ 0.101678    ┆ 1560  │
└──────────┴─────────────┴─────────────┴───────┘
```

Each fixed effect now has a standard error, `se_district`. It is the level's
variance given the coefficients, plus what the coefficients' uncertainty
adds through the level's covariance with them.

## Drawing the parameters and the school intercepts

```python
for m, draw in enumerate(fit.draw_parameters(10, seed=1, ranef=True, newdata=unknown)):
    imputed.append(draw.predict(unknown, type="draw", seed=m, id="student", name="p")
                   .select("student", pl.col("p").alias(f"proficient_{m}")))
```

`draw_parameters(n)` returns `n` copies of the result, each at one draw
from the normal approximation to the estimates' sampling distribution.
Each copy's `beta`, `fixef()`, `random_effects` and `predict` are the
draw's. A draw covers the slopes, the school sd and every district
effect jointly, including their covariances: a district's effect moves
with the slopes when its students' mean covariates are far from zero.
Nothing is read from the data. The district effects' block of the
information is diagonal, because each school is in one district, so each
district is drawn from its distribution given the slopes and the sd, using
pieces the fit already computed.

:::{admonition} Experimental
:class: warning
The draws of each group's effects given its rows (`ranef=True`,
`draw_ranef`) are checked against brute-force posteriors only, and warn
with `logit_stream.ExperimentalWarning`; so do the parameter draws of the
multinomial logit's adaptive fit.
:::

With `ranef=True`, each draw also carries a draw of every school's
intercept, from its distribution given the school's known students at that
draw's parameters:

$$
p(\delta_g \mid \text{its known students}) \;\propto\;
\prod_{i \in g} P(y_i \mid \eta_i + \delta_g) \times \varphi(\delta_g;\, 0, \tau^2),
$$

with $\eta_i$ each known student's own log-odds at the draw. The
intercept is therefore measured from the same log-odds that it is added to
at imputation. Each school's draw is exact. The density is evaluated on a
grid around the school's mode and sampled by its inverse cdf, so small
schools and schools whose known students all share one value get their
skewed distributions. All ten imputations' intercepts come from **one
read** of the known students, sorted by school and streamed in batches of
whole schools.

`newdata=unknown` adds the schools that have no known students. Each gets
a draw from N(0, τ²), and that draw, like every school's, is shared by all
the school's rows in `predict`:

```text
┌────────┬───────────┬───────────┬───────┐
│ school ┆ re_school ┆ sd_school ┆ n_obs │
╞════════╪═══════════╪═══════════╪═══════╡
│ 0      ┆ 1.317758  ┆ 0.647606  ┆ 5     │
│ 1      ┆ -1.441454 ┆ 0.777959  ┆ 1     │
│ 2      ┆ 1.585933  ┆ 0.647202  ┆ 4     │
└────────┴───────────┴───────────┴───────┘
```

`re_school` is the draw. `sd_school` is the school's conditional sd at
the draw's parameters.

The imputation is then `predict(type="draw")`: a draw from each
student's probability, with the school's drawn intercept. The draw's
uniform depends only on the seed and the student's identifier (`id=`), so
the step stays lazy and streams, and gives the same imputation however the
rows are ordered or split into batches. Use a different seed per
imputation.

```text
share proficient among the unknown: truth 0.4431, imputed 0.4435 (between-imputation sd 0.0031)
1,694 schools with 10+ known students: correlation of the imputed and true shares of their unknown students 0.754
```

`fit.draw_ranef()` draws the school intercepts once, at the estimates or at
a single draw. `draw_parameters` works for models without fixed effects
(from `vcov`, whichever type it is), with one absorbed fixed effect, and
with a random intercept fitted by `random_method="adaptive"`. The group
intercepts (`ranef=True`, `draw_ranef`) are available for the binary,
ordered and multinomial logit and probit with one random grouping.

**A multinomial characteristic.** With `mlogit_stream(...,
random_method="adaptive")`, a draw covers the coefficients, $\Sigma$ and
the fixed effects of every level jointly, and `ranef=True` draws each
group's J - 1 intercepts together, from their joint distribution given the
group's known rows. A group's draws therefore keep the correlations
between its levels that $\Sigma$ implies. The parameter draws take
$\Sigma$ on the scale of its log-Cholesky factor, so every draw is a valid
covariance matrix. The group draws are by importance resampling around
each group's mode (1,000 candidates from a multivariate t; see
`draw_ranef`). With J = 5, 20,000 groups and 60,000 known rows, ten
draws of everything took 16 seconds, against 92 for the fit. The
imputation is then `predict(type="draw")`, a level drawn from each row's
probabilities (the column `drawn`). The ordered logit's is the same, and
the conditional logit's draws one alternative per choice set.

**Random slopes.** If a covariate's effect varies across groups, with
`random={"school": ["1", "prior"]}` (see [random
slopes](../examples/random_intercepts.md#random-slopes)), the draws carry
each group's slope as well as its intercept, so the imputations keep how
the relation differs between schools. As with the intercepts, the
imputation model should include such a relation if the analyses will look
for it.

## Analyzing the completed data: Rubin's rules

Each completed data set, the known values with one imputation of the
others, is analyzed as if it were complete. The M analyses are then
combined by Rubin's rules (Rubin 1987). For an estimate $Q_m$ with variance
$U_m$ from each,

$$
\bar Q = \frac{1}{M} \sum_m Q_m , \qquad
T = \bar U + \Bigl(1 + \frac{1}{M}\Bigr) B ,
$$

with $\bar U$ the mean of the $U_m$ (the within-imputation variance) and
$B$ the variance of the $Q_m$ (the between-imputation variance, the
uncertainty that the missing values add). The pooled standard error is
$\sqrt{T}$, on a t distribution whose degrees of freedom shrink as $B$
grows relative to $\bar U$. The fraction of missing information (`fmi`) is
roughly $B$'s share of $T$.

**Scalar statistics: `pool`.** Give it a frame with one row per imputation
and statistic, here each district's share proficient with its binomial
standard error:

```python
shares = pl.concat([
    d.group_by("district").agg(pl.col("proficient").mean().alias("estimate"),
                               pl.len().alias("n"))
     .with_columns((pl.col("estimate") * (1 - pl.col("estimate")) / pl.col("n")).sqrt()
                   .alias("se"))
    for d in completed_sets])
pool(shares, by="district")
```

```text
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ district ┆ estimate ┆ se       ┆ ci_low   ┆ ci_high  ┆ fmi      │
╞══════════╪══════════╪══════════╪══════════╪══════════╪══════════╡
│ 0        ┆ 0.140691 ┆ 0.008735 ┆ 0.123338 ┆ 0.158044 ┆ 0.330332 │
│ 1        ┆ 0.184958 ┆ 0.008676 ┆ 0.167895 ┆ 0.202021 ┆ 0.163343 │
│ 2        ┆ 0.118299 ┆ 0.008275 ┆ 0.101917 ┆ 0.134681 ┆ 0.284042 │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
95% intervals covering the district's true share: 50 of 50; mean fraction of missing information 0.237
```

A quarter of the students' values are imputed, and on average a quarter
of each share's variance is the imputation's. `pool` also returns the
degrees of freedom, t and p for the estimate against 0, `within` and
`between`, and `riv`, the relative increase in variance $(1 + 1/M)B /
\bar U$. Without standard errors (`se=None`), a statistic is taken to have
no sampling variance, as for a count over a whole population, and all of
its variance is the imputation's. `df_com=` gives Barnard and Rubin's
(1999) small-sample degrees of freedom.

**Vectors: `pool_matrix`.** For a regression's coefficients, give it the
M fitted results (or arrays of estimates and covariance matrices). The
total covariance is $\bar U + (1 + 1/M) B$ with $B$ the coefficients'
covariance across imputations, and `wald_test` tests several of them
jointly, by Li, Raghunathan and Rubin's (1991) $D_1$ statistic:

```python
fits = [logit_stream("proficient ~ ses + prior | district", d, random="school",
                     random_method="adaptive") for d in completed_sets]
coefs = pool_matrix(fits)
coefs.tidy()
coefs.wald_test(["ses", "prior"], [-0.3, 0.5])
```

```text
┌───────┬───────────┬──────────┬────────────┬──────────┬──────────┐
│ term  ┆ estimate  ┆ se       ┆ df         ┆ riv      ┆ fmi      │
╞═══════╪═══════════╪══════════╪════════════╪══════════╪══════════╡
│ ses   ┆ -0.297689 ┆ 0.007983 ┆ 864.086736 ┆ 0.113656 ┆ 0.104128 │
│ prior ┆ 0.502022  ┆ 0.00975  ┆ 134.338021 ┆ 0.349226 ┆ 0.269628 │
└───────┴───────────┴──────────┴────────────┴──────────┴──────────┘
the complete data (unknown in practice): ses -0.2988 (se 0.0076), prior 0.5045 (se 0.0084)
joint test of ses = -0.3 and prior = 0.5: F = 0.065 on (2, 327) df, p = 0.937
```

The pooled estimates are close to those from the complete data, with
slightly larger standard errors: the price of the missing values. The
prior score's coefficient loses more information (`fmi` 0.27, against
0.10): it is measured from each student's own outcome, a quarter of which
are imputed, while much of the socioeconomic index's variation is between
schools, whose other students are known. The analysis model here has the
imputation model's form, a random school intercept within districts. An
analysis model that leaves out the school effects estimates a different
quantity (its coefficients are attenuated, -0.27 and 0.45 here, with or
without missing values), so its pooled estimates should be compared with
the same model's complete-data fit, not with the simulation's values. More
generally, the imputation model should include whatever the analyses will
use: a relation it leaves out is imputed as absent.

The tests check `pool` against R's mice (`pool.scalar`, including the
Barnard–Rubin degrees of freedom) and `wald_test` against mitml's $D_1$.

## Where these methods have been used

Proper multiple imputation from a logit model, with the coefficients drawn
from the normal approximation to their sampling distribution before each
imputation, is the standard method for binary and categorical variables. It
is, for example, how the `logreg` and `polyreg` methods of R's mice package
impute (van Buuren and Groothuis-Oudshoorn 2011). With clustered data, such
as students in schools or patients in hospitals, the imputation model has
to keep the clustering. That is done with group effects, drawn for each
group, in joint-model imputation (Schafer and Yucel 2002) and in chained
equations (Resche-Rigon and White 2018). Enders, Mistler and Keller (2016)
review both approaches for educational data. A long-standing large-scale
use in education is the plausible values of national assessments, such as
NAEP. They are multiple imputations of each student's proficiency from a
model with background covariates, drawn so that analyses combine them by
Rubin's rules (Mislevy 1991).

## Checks of the draws

:::{dropdown} What the tests compare the draws with
The tests compare the covariance that the draws follow with independent
references:

- **One fixed effect:** a fit with the levels as dummy variables (standard
  errors and cross-covariances agree to 1e-6).
- **Adaptive random intercept within a fixed effect:** brute-force
  integration with a numerical Hessian (to 3e-3).
- **Group intercepts:** each group's conditional mean and sd match brute-force
  integration (to 2e-5 and 1e-4). The draws' probability integral transform
  under brute force's cdf is uniform, for binary and ordered outcomes, for
  groups of one to 400 rows, and for groups whose rows all share one
  outcome.
- **Multinomial group intercepts (J = 3):** each group's conditional means
  and sds match a brute-force 2-D grid to within their Monte Carlo error.
  The probability integral transforms of each intercept, and of their sum,
  are uniform.
- **Multinomial parameter draws:** the coefficients' covariance matches
  `vcov`, $\Sigma$'s spread matches its standard errors, and the fixed
  effects' spread matches their `fixef()` standard errors.

In 60 simulated data sets (50 fixed-effect levels, about 50,000 rows), the
fixed effects' standard errors were calibrated: across 3,000 estimates, the
z-statistics had sd 0.99, and 95.7% of the 95% intervals covered the true
effect. The slope's standard error matched the estimates' spread with 10,000
groups: the spread was 0.98 times the mean standard error, over 100 data
sets. With 3,000 groups and a covariate sorted across groups, the standard
error was 10–20% too large, by the same amount under the Laplace fit. That
error makes the imputations' variance slightly conservative.
:::

## References

- Barnard, J. and D. B. Rubin (1999). Small-sample degrees of freedom with
  multiple imputation. *Biometrika* 86(4), 948--955.
- Enders, C. K., S. A. Mistler and B. T. Keller (2016). Multilevel
  multiple imputation: a review and evaluation of joint modeling and
  chained equations imputation. *Psychological Methods* 21(2), 222--240.
- Li, K. H., T. E. Raghunathan and D. B. Rubin (1991). Large-sample
  significance levels from multiply imputed data using moment-based
  statistics and an F reference distribution. *Journal of the American
  Statistical Association* 86(416), 1065--1073.
- Mislevy, R. J. (1991). Randomization-based inference about latent
  variables from complex samples. *Psychometrika* 56(2), 177--196.
- Resche-Rigon, M. and I. R. White (2018). Multiple imputation by chained
  equations for systematically and sporadically missing multilevel data.
  *Statistical Methods in Medical Research* 27(6), 1634--1649.
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys.*
  Wiley.
- Schafer, J. L. and R. M. Yucel (2002). Computational strategies for
  multivariate linear mixed-effects models with missing values. *Journal
  of Computational and Graphical Statistics* 11(2), 437--457.
- van Buuren, S. and K. Groothuis-Oudshoorn (2011). mice: multivariate
  imputation by chained equations in R. *Journal of Statistical Software*
  45(3), 1--67.

:::{dropdown} The full script: `examples/multiple_imputation.py`
```{literalinclude} ../../examples/multiple_imputation.py
:language: python
```
:::
