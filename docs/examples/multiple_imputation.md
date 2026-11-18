# Multiple imputation

{download}`Download the script: multiple_imputation.py <../../examples/multiple_imputation.py>`

[Imputation at scale](imputation.md) predicts a characteristic as
probabilities at the estimates. Multiple imputation (Rubin 1987) instead
fills in the missing values several times, so that analyses of the
completed data can carry the uncertainty of the imputation. For this to
work, each imputation must use parameters drawn from their sampling
distribution, not the estimates alone ("proper" imputation). With group
effects, it must also use one draw of each group's intercept, shared by
the group's missing members, so that the imputed values keep the
within-group correlation that the model implies.

The script's setting is matched employer-employee data. Workers are in
firms, firms are in industry × area cells, and a binary characteristic is
known for three-quarters of the workers. The model is a random firm
intercept within cell fixed effects, fitted on the known workers:

```python
fit = logit_stream("female ~ age10 + x | cell", obs, random="firm",
                   random_method="adaptive")
fit.fixef().head(3).collect()
```

```text
┌──────┬───────────┬──────────┬───────┐
│ cell ┆ fe_cell   ┆ se_cell  ┆ n_obs │
╞══════╪═══════════╪══════════╪═══════╡
│ 0    ┆ -0.928469 ┆ 0.098759 ┆ 1705  │
│ 1    ┆ -0.551466 ┆ 0.090419 ┆ 1795  │
│ 2    ┆ -1.159156 ┆ 0.106257 ┆ 1560  │
└──────┴───────────┴──────────┴───────┘
```

Each fixed effect now has a standard error, `se_cell`. It is the level's
variance given the coefficients, plus what the coefficients' uncertainty
adds through the level's covariance with them.

## Drawing the parameters and the firm intercepts

```python
for m, draw in enumerate(fit.draw_parameters(10, seed=1, ranef=True, newdata=unknown)):
    u = pl.col("worker").hash(seed=m).cast(pl.Float64) / 2.0**64
    imputed.append(draw.predict(unknown, type="response", name="p")
                   .select("worker", (u < pl.col("p")).cast(pl.Int8).alias(f"female_{m}")))
```

`draw_parameters(n)` returns `n` copies of the result, each at one draw
from the normal approximation to the estimates' sampling distribution.
Each copy's `beta`, `fixef()`, `random_effects` and `predict` are the
draw's. A draw covers the slopes, the firm sd and every cell effect
jointly, including their covariances: a cell's effect moves with the
slopes when its workers' mean covariates are far from zero. Nothing is
read from the data. The cell effects' block of the information is
diagonal, because each firm is in one cell, so each cell is drawn from its
distribution given the slopes and the sd, using pieces the fit already
computed.

With `ranef=True`, each draw also carries a draw of every firm's
intercept, from its distribution given the firm's known workers at that
draw's parameters:

$$
p(\delta_g \mid \text{its known workers}) \;\propto\;
\prod_{i \in g} P(y_i \mid \eta_i + \delta_g) \times \varphi(\delta_g;\, 0, \tau^2),
$$

with $\eta_i$ each known worker's own log-odds at the draw. The
intercept is therefore measured from the same log-odds that it is added to
at imputation. Each firm's draw is exact. The density is evaluated on a
grid around the firm's mode and sampled by its inverse cdf, so small
firms and firms whose known workers all share one value get their skewed
distributions. All ten imputations' intercepts come from **one read** of
the known workers, sorted by firm and streamed in batches of whole firms.

`newdata=unknown` adds the firms that have no known workers. Each gets a
draw from N(0, τ²), and that draw, like every firm's, is shared by all
the firm's rows in `predict`:

```text
┌──────┬───────────┬──────────┬───────┐
│ firm ┆ re_firm   ┆ sd_firm  ┆ n_obs │
╞══════╪═══════════╪══════════╪═══════╡
│ 0    ┆ 1.317758  ┆ 0.647606 ┆ 5     │
│ 1    ┆ -1.441454 ┆ 0.777959 ┆ 1     │
│ 2    ┆ 1.585933  ┆ 0.647202 ┆ 4     │
└──────┴───────────┴──────────┴───────┘
```

`re_firm` is the draw. `sd_firm` is the firm's conditional sd at the
draw's parameters.

The imputation is then `predict(type="response")`, the probability with
the firm's drawn intercept, followed by a Bernoulli draw as a Polars
expression. The uniform comes from a hash of the worker's identifier, with
a different seed per imputation, so the whole step stays lazy and streams.

```text
share female among the unknown: truth 0.4431, imputed 0.4442 (between-imputation sd 0.0029)
1,694 firms with 10+ known workers: correlation of the imputed and true shares of their unknown workers 0.744
```

`fit.draw_ranef()` draws the firm intercepts once, at the estimates or at
a single draw. `draw_parameters` works for models without fixed effects
(from `vcov`, whichever type it is), with one absorbed fixed effect, and
with a random intercept fitted by `random_method="adaptive"`. The firm
intercepts (`ranef=True`, `draw_ranef`) are available for the binary,
ordered and multinomial logit and probit with one random grouping.

**A multinomial characteristic.** With `mlogit_stream(...,
random_method="adaptive")`, a draw covers the coefficients, $\Sigma$ and
the cell effects of every level jointly, and `ranef=True` draws each
firm's J - 1 intercepts together, from their joint distribution given the
firm's known workers. A firm's draws therefore keep the correlations
between its levels that $\Sigma$ implies. The parameter draws take
$\Sigma$ on the scale of its log-Cholesky factor, so every draw is a valid
covariance matrix. The firm draws are by importance resampling around
each firm's mode (1,000 candidates from a multivariate t; see
`draw_ranef`). With J = 5, 20,000 firms and 60,000 known workers, ten
draws of everything took 16 seconds, against 92 for the fit. The
imputation is then `predict(type="response")`, giving `prob_<level>`
columns, and a categorical draw from them per worker.

## Checks of the draws

The tests compare the covariance that the draws follow with independent
references:

- **One fixed effect:** a fit with the cells as dummy variables (standard
  errors and cross-covariances agree to 1e-6).
- **Adaptive random intercept within a fixed effect:** brute-force
  integration with a numerical Hessian (to 3e-3).
- **Firm intercepts:** each firm's conditional mean and sd match brute-force
  integration (to 2e-5 and 1e-4). The draws' probability integral transform
  under brute force's cdf is uniform, for binary and ordered outcomes, for
  firms of one to 400 rows, and for firms whose rows all share one
  outcome.
- **Multinomial firm intercepts (J = 3):** each firm's conditional means
  and sds match a brute-force 2-D grid to within their Monte Carlo error.
  The probability integral transforms of each intercept, and of their sum,
  are uniform.
- **Multinomial parameter draws:** the coefficients' covariance matches
  `vcov`, $\Sigma$'s spread matches its standard errors, and the cell
  effects' spread matches their `fixef()` standard errors.

In 60 simulated data sets (50 cells, about 50,000 rows), the cell effects'
standard errors were calibrated: across 3,000 cell estimates, the
z-statistics had sd 0.99, and 95.7% of the 95% intervals covered the true
effect. The slope's standard error matched the estimates' spread with 10,000
firms: the spread was 0.98 times the mean standard error, over 100 data
sets. With 3,000 firms and a covariate sorted across firms, the standard
error was 10–20% too large, by the same amount under the Laplace fit. That
error makes the imputations' variance slightly conservative.

:::{dropdown} The full script: `examples/multiple_imputation.py`
```{literalinclude} ../../examples/multiple_imputation.py
:language: python
```
:::

## References

- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys.*
  Wiley.
