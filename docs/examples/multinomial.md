# Multinomial logit

{download}`Download the script: multinomial.py <../../examples/multinomial.py>`

$$
P(y_i = j \mid x_i) = \frac{\exp(x_i'\beta_j)}{\sum_{l} \exp(x_i'\beta_l)},
\qquad \beta_{\text{base}} = 0 .
$$

```python
import polars as pl
from logit_stream import etable, mlogit_stream
from logit_stream.simulate import RACES, simulate_people

people = simulate_people(200_000, seed=1)
X = "age10 + female + log_income + owner + urban"
fit = mlogit_stream(f"race ~ {X}", people, levels=RACES, vcov="hetero",
                    cluster=["puma", "state"])
```

## Parameterization

The coefficients of each non-base level are its log-odds against the base
level, named `"<level>:<covariate>"` in one flat vector, equation by equation:
`Black:Intercept`, `Black:age10`, ..., then `Hispanic:Intercept`, ... This is
statsmodels' `MNLogit` and Stata's `mlogit` parameterization, and the
coefficients and model-based standard errors match statsmodels' to rounding.
`coef_matrix()` shows them as a table, a column per non-base level:

```text
┌─────────────┬───────────┬───────────┬───────────┬───────────┬─────────────┐
│ Coefficient ┆ Black     ┆ Hispanic  ┆ Asian     ┆ AIAN      ┆ Multiracial │
╞═════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════════════╡
│ Intercept   ┆ 2.204253  ┆ 1.982456  ┆ -4.930454 ┆ 0.811849  ┆ -2.574906   │
│ age10       ┆ -0.096926 ┆ -0.238137 ┆ -0.052053 ┆ -0.026215 ┆ -0.294412   │
│ female      ┆ 0.044251  ┆ 0.007279  ┆ 0.026936  ┆ 0.00196   ┆ 0.037175    │
│ log_income  ┆ -0.351537 ┆ -0.304331 ┆ 0.23807   ┆ -0.448976 ┆ -0.031082   │
│ owner       ┆ -0.505478 ┆ -0.269263 ┆ -0.096139 ┆ 0.116991  ┆ -0.334384   │
│ urban       ┆ 0.858253  ┆ 0.579504  ┆ 0.972178  ┆ -0.84106  ┆ 0.311796    │
└─────────────┴───────────┴───────────┴───────────┴───────────┴─────────────┘
```

The urban coefficients are close to those `simulate_people` draws from
(`truth_people()`: 0.9, 0.6, 1.0, -0.8, 0.3); they are not exactly the
truth's because the simulation adds PUMA effects that this model omits.

scikit-learn's `LogisticRegression` uses the symmetric parameterization (a
coefficient vector for every level); without a penalty the differences of its
coefficients from the base level's are these. With a penalty the two
parameterizations penalize different things, so the penalized fits differ
(see [`l2`](#penalized-fits)).

The outcome can be strings, a categorical or Enum, or integers. Its levels
are its distinct values in the estimation sample, sorted (an Enum by its
categories), unless `levels=` gives them; the first is the base. A level of
`levels=` that does not occur is an error, as is an observed value not among
them.

## Standard errors

`vcov=` takes `"iid"` (the inverse of the information matrix), `"hetero"`
(the sandwich, times N / (N - 1)) or `{"CRV1": var}` (clustered, times
G / (G - 1)) -- Stata's small-sample conventions for maximum-likelihood
estimators. statsmodels' `cov_type="HC0"` and `"cluster"` (with
`use_correction=False`) are the same matrices without those factors.
`cluster=` computes further clusterings at fit time, for `with_vcov`:

```python
fit.with_vcov({"CRV1": "state"}).tidy()
```

```text
┌──────────────────┬──────────┬──────────┬───────────┬────────────┐
│ coefficient      ┆ iid      ┆ hetero   ┆ CRV1 puma ┆ CRV1 state │
╞══════════════════╪══════════╪══════════╪═══════════╪════════════╡
│ Black:Intercept  ┆ 0.081302 ┆ 0.081387 ┆ 0.151528  ┆ 0.161009   │
│ Black:age10      ┆ 0.004644 ┆ 0.004638 ┆ 0.004493  ┆ 0.004303   │
│ Black:female     ┆ 0.012923 ┆ 0.012924 ┆ 0.011748  ┆ 0.012406   │
│ Black:log_income ┆ 0.00774  ┆ 0.007737 ┆ 0.014042  ┆ 0.014054   │
│ Black:owner      ┆ 0.020758 ┆ 0.020735 ┆ 0.021796  ┆ 0.019453   │
│ Black:urban      ┆ 0.013743 ┆ 0.013731 ┆ 0.029272  ┆ 0.029564   │
└──────────────────┴──────────┴──────────┴───────────┴────────────┘
```

The PUMA effects the model omits are shared within PUMAs, so clustering
doubles the standard errors of the intercept, income and urban
coefficients, which vary mostly between PUMAs. The iid and robust variances
each cost nothing beyond the fit and one more read of the data
respectively; the clustered ones keep a cluster x parameter array of score
sums in memory.

**Multi-way clustering**, `{"CRV1": "state+age"}`, is inclusion-exclusion
over the one-way clusterings state, age and state^age (each computed in
the same read), every term scaled by G / (G - 1) with the smaller one-way
G, and df G - 1: hdfe-stream's and pyfixest's convention. It can have
negative variances, which a warning reports (clustering by a variable with
two values, such as `urban`, is a quick way to get them).

**The cluster jackknife**, `{"CRV3": "state"}`, is (G - 1) / G times the sum
over clusters of d_g d_g', d_g the change in the estimates from dropping
cluster g (MacKinnon, Nielsen and Webb's CV3). For a likelihood the exact
d_g would take G refits, so it is one Newton step from the estimates,
-(H - H_g)^-1 s_g, from each cluster's score and information: one more read,
and G x P^2 numbers in memory. With 12 clusters of 200 rows the standard
errors were within 2% of the refitted jackknife's (tested). One-way, for
the multinomial and ordered logit without fixed effects or random
intercepts. With 50 states, the three differ little:

```text
┌──────────────────┬────────────┬────────────────┬────────────┐
│ coefficient      ┆ CRV1 state ┆ CRV1 state+age ┆ CRV3 state │
╞══════════════════╪════════════╪════════════════╪════════════╡
│ Black:Intercept  ┆ 0.161009   ┆ 0.162748       ┆ 0.165879   │
│ Black:age10      ┆ 0.004303   ┆ 0.003041       ┆ 0.004313   │
│ Black:female     ┆ 0.012406   ┆ 0.012039       ┆ 0.012438   │
│ ...
```

## Weights

`weights="w"` multiplies each row's log-likelihood by `w`. With
`weights_type="fweights"` each row counts as `w` observations (`n_obs` is
their sum, and the fit is exactly that of the data with each row repeated);
with the default `"aweights"` the weights change the estimates but not the
count of observations, and are rescaled to average 1 (Stata's convention for
maximum likelihood), so that the iid vcov and the log-likelihood do not
depend on their scale. Survey weights are analytic weights with a robust or
clustered vcov -- the same standard errors as Stata's `[pweight]`.

Frequency weights make a table of counts as good as the rows it counts:
with only categorical covariates, the 200,000 people collapse to 48 rows,

```python
counts = (people.with_columns(n=pl.lit(1)).group_by("race", "female", "urban", "owner")
          .agg(pl.col("n").sum()))
fw = mlogit_stream("race ~ female + urban + owner", counts, levels=RACES, weights="n",
                   weights_type="fweights")
```

```text
frequency weights: 48 rows, n_obs 200,000; largest difference from the row-level fit 1.0e-15 (se 1.3e-16)
```

## Several models and a table

pyfixest's multiple-estimation syntax works (`csw()`, `sw()`, several
outcomes), and returns a `LogitMulti`; `etable` makes a regression table of
results, through pyfixest's maketables:

```python
multi = mlogit_stream("race ~ csw(age10 + female, log_income, owner + urban)", people,
                      levels=RACES)
etable(list(multi.all_fitted_models.values()), type="md", keep=["Black:"])
```

```text
|                                | ('race', '(1)')   | ('race', '(2)')   | ('race', '(3)')   |
|:-------------------------------|:------------------|:------------------|:------------------|
| ('coef', 'Black × Intercept')  | -0.779***         | 2.451***          | 2.204***          |
|                                |  (0.024)          |  (0.080)          |  (0.081)          |
| ('coef', 'Black × age10')      | -0.142***         | -0.125***         | -0.097***         |
|                                |  (0.004)          |  (0.004)          |  (0.005)          |
| ...
```

## Penalized fits

`l2=` subtracts a ridge penalty `l2/2 * sum(beta^2)` from the log-likelihood,
on every coefficient but the intercepts. It is the cure for separated
coefficients (see below), and with two levels it is scikit-learn's
`LogisticRegression(C=1/l2)` exactly. The reported standard errors are then
those of the penalized fit (the inverse of the penalized information), which
understate the uncertainty of a biased estimator; use them with care.
`l2=100` moves `Black:urban` from 0.858 to 0.840 here. For a binary
outcome, or a multinomial one, `firth=True` is the other cure (see [binary
outcomes](binary.md#separated-covariates-and-firths-penalized-likelihood)).

## Collinearity and separation

A covariate that is a linear combination of others (an indicator of a level
that never occurs, a variable that duplicates another) is found from the
first step's information matrix, dropped from every equation with a warning,
and listed in `collin_vars`, as pyfixest does:

```python
dup = people.with_columns(urban2=2 * pl.col("urban"))
mlogit_stream(f"race ~ {X} + urban2", dup, levels=RACES).collin_vars     # ['urban2']
```

A covariate that *separates* the outcome -- every row with some level of an
indicator has the same outcome, or never has some outcome -- has no finite
estimate. Newton's method then converges linearly instead of quadratically,
with the estimate drifting toward infinity; the fit warns when a coefficient
grows implausibly large with an exploding standard error, or when it took
more than 15 steps. Merge the level, drop the covariate, or fit with `l2=`
or `firth=True`.

## Options

| option | default | |
|---|---|---|
| `tol` | `1e-8` | the Newton decrement at which to stop (see [how it works](../how_it_works.md)) |
| `maxiter` | `50` | the most Newton steps |
| `storage` | `"disk"` | where the blocks live between steps (`"memory"` keeps them in memory) |
| `dtype` | `"float64"` | the blocks' precision on disk (`"float32"` halves it) |
| `block_mb` | `64` | the size of a block |
| `chunk_mb` | `64` | the most memory a chunk's working arrays take |
| `polars_engine` | `"cpu"` | `"gpu"` runs pass 0's queries on cudf-polars |
| `device` | `"cpu"` | `"cuda"` forms each Newton step's sums on an NVIDIA GPU (experimental; see [on a GPU](../how_it_works.md#on-a-gpu)) |

:::{dropdown} The full script: `examples/multinomial.py`
```{literalinclude} ../../examples/multinomial.py
:language: python
```
:::
