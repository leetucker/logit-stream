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

## Step 1: draw the parameters

```python
for draw in fit.draw_parameters(10, seed=1):
    tau = draw.random_effects["firm"]["sd"][0]
    eta = draw.predict(workers, type="link").collect()["prediction"].to_numpy()
```

`draw_parameters(n)` returns `n` copies of the result, each at one draw from
the normal approximation to the estimates' distribution. Each copy's
`beta`, `fixef()`, `random_effects` and `predict` are the draw's. The draw
covers the slopes, the firm sd and every cell effect jointly, including
their covariances: a cell's effect moves with the slopes when its workers'
mean covariates are far from zero. Nothing is read from the data to make
a draw. The cell effects' block of the information is diagonal, because
each firm is in one cell, so each cell is drawn from its distribution
given the slopes and the sd, using pieces the fit already computed.

A drawn result's `predict` gives each worker's log-odds **without** the firm
intercept: cell effect plus covariates, at the draw. Its `ranef()` raises,
since the firm intercepts are not part of the draw (step 2).

`draw_parameters` works for models without fixed effects (from `vcov`,
whichever type it is), with one absorbed fixed effect, and with a random
intercept fitted by `random_method="adaptive"`. It is not available for
the Laplace or joint-mode random intercepts, several fixed effects, or the
mixed and nested logit.

## Step 2: draw each firm's intercept

Given a draw, firm *g*'s intercept has the distribution

$$
p(\delta_g \mid \text{its known workers}) \;\propto\;
\prod_{i \in g} \Lambda(\eta_i + \delta_g)^{y_i}\,[1 - \Lambda(\eta_i + \delta_g)]^{1 - y_i}
\;\times\; \varphi(\delta_g;\, 0, \tau^2),
$$

with $\eta_i$ the draw's log-odds of the firm's known workers. Each worker
has their own $\eta_i$: the intercept is measured from the same log-odds
it is added to in step 3. The script draws from this distribution on a
grid around each firm's mode, in about 30 lines of numpy
(`draw_firm_effects`). A firm with no known workers draws from
N(0, τ²). This step is not in the package yet.

```python
    delta = draw_firm_effects(f_obs, y_obs, eta[known], G, tau, rng)
```

## Step 3: impute

```python
    y[miss] = rng.random(miss.sum()) < expit(eta[miss] + delta[firm[miss]])
```

```text
share female among the unknown: truth 0.4431, imputed 0.4428 (between-imputation sd 0.0016)
1,694 firms with 10+ known workers: correlation of the imputed and true shares of their unknown workers 0.758
```

## Checks of the draws

The tests compare the covariance that the draws follow with independent
references:

- **One fixed effect:** a fit with the cells as dummy variables (standard
  errors and cross-covariances agree to 1e-6).
- **Adaptive random intercept within a fixed effect:** brute-force
  integration with a numerical Hessian (to 3e-3).

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
