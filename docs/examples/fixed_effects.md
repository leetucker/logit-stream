# Fixed effects

{download}`Download the script: fixed_effects.py <../../examples/fixed_effects.py>`

The multinomial and ordered logit absorb a fixed effect, written after `|`
as in pyfixest:

```python
from logit_stream import mlogit_stream, ologit_stream
from logit_stream.simulate import RACES, simulate_people

people = simulate_people(200_000, seed=1)
X = "age10 + female + log_income + owner + urban"
race = mlogit_stream(f"race ~ {X} | puma", people, levels=RACES, vcov={"CRV1": "puma"})
race.summary()
```

```text
### race ~ age10 + female + log_income + owner + urban | puma
Multinomial logit (logit_stream, Newton)   vcov: CRV1:puma
obs: 200,000   outcome race: 6 levels, base 'White'
log-likelihood: -197403.3   null: -214642.29   pseudo R2: 0.080315   AIC: 403814.59   BIC: 449782.75
within pseudo R2 (beyond puma): 0.034303   fixed-effect-only log-likelihood: -204415.43
Newton steps: 5 (converged, decrement 6.76e-14)
fixed effects: puma (1,020 levels)
┌────────────────────────┬───────────┬────────────┬────────────┬─────────────┐
│ Coefficient            ┆ Estimate  ┆ Std. Error ┆ t value    ┆ Pr(>|t|)    │
╞════════════════════════╪═══════════╪════════════╪════════════╪═════════════╡
│ Black:age10            ┆ -0.098908 ┆ 0.004651   ┆ -21.26601  ┆ 2.3439e-100 │
│ Black:female           ┆ 0.044363  ┆ 0.012182   ┆ 3.641625   ┆ 0.000271    │
│ Black:log_income       ┆ -0.356017 ┆ 0.008351   ┆ -42.632552 ┆ 0.0         │
│ Black:owner            ┆ -0.521123 ┆ 0.022099   ┆ -23.581663 ┆ 5.9447e-123 │
│ Black:urban            ┆ 0.899742  ┆ 0.014813   ┆ 60.741816  ┆ 0.0         │
│ …                      ┆ …         ┆ …          ┆ …          ┆ …           │
```

With the PUMA effects in the model, as in the simulation, the coefficients
are closer to those it draws from than without them (`Black:urban` 0.90
against 0.9; 0.86 without). An interaction of columns is one fixed effect
(`| state^year`). The binary logit (`logit_stream`) absorbs any number,
through hdfe-stream; the multinomial logit too, [differently](#several-fixed-effects).

## What is estimated

In the multinomial logit, each level $g$ of the fixed effect has an intercept
for every outcome level, $\alpha_{gj}$, relative to the group's reference
level (the base, if the group has it):

```python
race.fixef("puma").collect()
```

```text
┌──────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬───────┐
│ puma ┆ fe_puma:Whi ┆ fe_puma:Bla ┆ fe_puma:His ┆ fe_puma:Asi ┆ fe_puma:AIA ┆ fe_puma:Mul ┆ n_obs │
│      ┆ te          ┆ ck          ┆ panic       ┆ an          ┆ N           ┆ tiracial    ┆       │
╞══════╪═════════════╪═════════════╪═════════════╪═════════════╪═════════════╪═════════════╪═══════╡
│ 0    ┆ 0.0         ┆ 1.605433    ┆ 2.251736    ┆ -5.119055   ┆ 0.155857    ┆ -2.07634    ┆ 405   │
│ 1    ┆ 0.0         ┆ 1.854129    ┆ 2.363205    ┆ -5.923569   ┆ 1.042276    ┆ -2.521916   ┆ 414   │
│ 2    ┆ 0.0         ┆ 2.014074    ┆ 2.650479    ┆ -4.855238   ┆ 2.41925     ┆ -2.463164   ┆ 108   │
│ 3    ┆ 0.0         ┆ 2.637434    ┆ 2.3408      ┆ -5.039432   ┆ null        ┆ null        ┆ 109   │
└──────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴───────┘
```

In the ordered logit, it is a shift $\alpha_g$ of the latent index, with
`cut1` fixed at 0 (`ologit_stream(f"education ~ {X} | puma", people)`).

That is the model with an indicator for every level, and it gives the same
estimates and standard errors (iid, robust and clustered alike), which the
test suite checks; it is just cheaper to fit. With 50 states:

```python
absorbed = mlogit_stream(f"race ~ {X} | state", people, levels=RACES, vcov="hetero")
dummies = mlogit_stream(f"race ~ {X} + C(state)", people, levels=RACES, vcov="hetero")
```

```text
largest difference in estimates: 1.2e-15  in standard errors: 1.2e-09
time: absorbed 0.9 s, indicators 2.1 s
```

With 1,020 PUMAs the indicators would be 5,100 columns, and a dense
information matrix of 26 million elements per Newton step.

## How

A model with $G$ levels and $P$ coefficients has $P + G(J-1)$ parameters, too
many for a dense information matrix at $G$ in the thousands. But each group's
effects touch only that group's rows, so the information matrix has an arrow
shape: a dense $P \times P$ block for the coefficients, a small block per
group for its effects, and a $P \times (J-1)$ block per group between them.
Each Newton step accumulates those blocks in one read of the data and solves
the system exactly by eliminating the group blocks (the Schur complement);
the coefficients' variance is the inverse of the reduced matrix, which is
exactly the corresponding block of the inverse of the whole matrix. The
robust and clustered variances use each row's score net of the part the
fixed effects absorb. The rows need not be sorted by group.

Memory grows with $G \times (J-1) \times P$: 2,462 PUMAs, five outcome levels
and 150 coefficients take about 15 MB. A fit refuses to start above 8 GB.

## Separation

A group whose outcome never varies (in the multinomial logit, every row has
the same level; in the ordered, every row is at the lowest or the highest
level) has infinite effects, and its rows say nothing about the coefficients:
it is dropped before the fit, with a warning, and recorded in
`diagnostics["separation"]` and `sample()`'s `dropped_because`, as
hdfe-stream does for the binary logit.

In the multinomial logit, a group that never has some outcome level gets
probability zero for it (its effect is minus infinity): `fixef()` shows a
null for it, as for PUMA 3 above, and `predict()` a probability of 0, and
the fit is the one without those cells. Here 621 of the 5,100 PUMA-level
cells are empty, mostly for the two rarest levels. With many small groups
that is a real limitation for prediction -- a record of a level its PUMA's
sample lacks gets probability 0 for its true level -- which [random
intercepts](random_intercepts.md) avoid (see
[imputation](imputation.md#fixed-or-random-effects)).

## Several fixed effects

With two or more (`"race ~ female + log_income | puma + age"`),
`mlogit_stream` fits the equivalent Poisson regression with hdfe-stream's
`fepois_stream`. Each observation $i$ becomes $J$ rows, one per outcome
level $j$, with the count $1[y_i = j]$; the regression has a fixed effect
per observation, each of the model's fixed effects interacted with the
level, and each covariate interacted with the non-base levels. Concentrated
over the observations' effects -- which make each row's mean its
probability, $\sum_j \mu_{ij} = 1$ -- its log-likelihood is the multinomial
logit's (Baker 1994; the same "Poisson trick" equates the conditional logit
with a Poisson regression with a fixed effect per choice set). So:

- the coefficients are the multinomial logit's, and the log-likelihood is
  the Poisson's plus the (weighted) number of observations;
- the information for the coefficients, net of all the effects, is the
  same, so the iid variance is, once pyfixest's small-sample factor is
  replaced by logit-stream's (none for the iid; G / (G - 1) for clusters);
- the robust variance is the Poisson's clustered by observation (the
  multinomial score of an observation is the sum of its rows'), with
  $N / (N - 1)$ -- not available with frequency weights, which that
  clustering would square;
- the data grow $J$-fold, and the regression has one fixed effect per
  observation, so it costs more than the one-fixed-effect solver above;
- the result is a `MultinomialLogitResult` with `fixef()` for each dimension
  (a column per level, the base level's 0 where the group has it) and
  `predict()`; `sample()` labels only missing values (the levels dropped for
  separation are cells of the expanded data, not observations), and `l2=`,
  `start=` and random intercepts are not available;
- `device="cuda"` is then fepois_stream's (its fixed effects' solve on the
  GPU), and `poisson_options=` passes any of fepois_stream's other options
  (`solver`, `precond`, `fixef_rm`, ...).

```python
two = mlogit_stream("race ~ female + log_income + owner + urban | puma + age", people,
                    levels=RACES, vcov={"CRV1": "puma"})
two.fixef("age").collect()
```

```text
UserWarning: race ~ female + log_income + owner + urban | puma + age: 621 (puma, outcome
level) cells, 2 (age, outcome level) cells have no observations of that level: their
probability is 0 (their effects are minus infinity), and fixef() shows a null
┌─────┬──────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬───────┐
│ age ┆ fe_age:White ┆ fe_age:Blac ┆ fe_age:Hisp ┆ fe_age:Asia ┆ fe_age:AIAN ┆ fe_age:Mult ┆ n_obs │
│     ┆              ┆ k           ┆ anic        ┆ n           ┆             ┆ iracial     ┆       │
╞═════╪══════════════╪═════════════╪═════════════╪═════════════╪═════════════╪═════════════╪═══════╡
│ 18  ┆ 0.0          ┆ 1.352714    ┆ 1.417932    ┆ -2.359284   ┆ 0.616152    ┆ -1.108606   ┆ 379   │
│ 19  ┆ 0.0          ┆ 0.969493    ┆ 0.967301    ┆ -2.29046    ┆ -0.415205   ┆ -0.718626   ┆ 227   │
│ 20  ┆ 0.0          ┆ 0.977531    ┆ 0.987516    ┆ -3.046074   ┆ 0.301195    ┆ -0.961219   ┆ 302   │
└─────┴──────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴───────┘
time: 10.4 s
```

`age10` is left out: it is collinear with single years of age, which the
fit refuses with an error. The test suite checks the route against the
one-fixed-effect fit with the other dimension as indicators: the same
estimates, log-likelihood, variances, parameter count and predictions.

## Few rows per group

With few rows per group, the coefficients of a fixed-effects logit are
biased (the incidental-parameter problem; Neyman and Scott, 1948): with two
observations per group, the binary logit's are twice the truth. The bias
falls with the number of rows per group, roughly as 1 / T. Absorbing a
fixed effect is meant for groups with many rows each -- PUMAs, counties,
schools. For panels of individuals with a few periods each, the binary
logit has two remedies: the analytic bias correction
(`logit_stream(..., bias_correction=True)`) and Chamberlain's conditional
logit (`cfelogit_stream`), which conditions the effects out; neither is
available for the multinomial or ordered logit. See [incidental
parameters](incidental_parameters.md).

:::{dropdown} The full script: `examples/fixed_effects.py`
```{literalinclude} ../../examples/fixed_effects.py
:language: python
```
:::
