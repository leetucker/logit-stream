# Ordered logit and probit

{download}`Download the script: ordered.py <../../examples/ordered.py>`

$$
P(y_i \le j \mid x_i) = F(\kappa_j - x_i'\beta), \qquad
\kappa_1 < \dots < \kappa_{J-1},
$$

with $F$ the logistic distribution function (`ologit_stream`) or the
standard normal's (`oprobit_stream`): a positive coefficient moves
probability toward the higher levels.

```python
import polars as pl
from logit_stream import ologit_stream, oprobit_stream
from logit_stream.simulate import EDUCATION, simulate_people

people = simulate_people(200_000, seed=1)
X = "age10 + female + log_income + owner + urban"
fit = ologit_stream(f"education ~ {X}", people, vcov="hetero")
fit.tidy()
```

```text
┌─────────────┬───────────┬────────────┬────────────┬─────────────┐
│ Coefficient ┆ Estimate  ┆ Std. Error ┆ t value    ┆ Pr(>|t|)    │
╞═════════════╪═══════════╪════════════╪════════════╪═════════════╡
│ age10       ┆ 0.045762  ┆ 0.002861   ┆ 15.996862  ┆ 1.3438e-57  │
│ female      ┆ 0.242993  ┆ 0.008052   ┆ 30.178821  ┆ 4.4924e-200 │
│ log_income  ┆ 0.811634  ┆ 0.005046   ┆ 160.833563 ┆ 0.0         │
│ owner       ┆ 0.281224  ┆ 0.011119   ┆ 25.291536  ┆ 3.9581e-141 │
│ urban       ┆ 0.265801  ┆ 0.008153   ┆ 32.603152  ┆ 3.7004e-233 │
│ cut1        ┆ 6.756594  ┆ 0.052894   ┆ 127.73735  ┆ 0.0         │
│ cut2        ┆ 8.192264  ┆ 0.053519   ┆ 153.071066 ┆ 0.0         │
│ cut3        ┆ 9.237465  ┆ 0.054303   ┆ 170.108772 ┆ 0.0         │
│ cut4        ┆ 10.483533 ┆ 0.055288   ┆ 189.617347 ┆ 0.0         │
└─────────────┴───────────┴────────────┴────────────┴─────────────┘
```

`fit.cutpoints` is $\kappa_1, \dots, \kappa_4$ as an array. The slopes are
close to the ones `simulate_people` draws from (0.05, 0.25, 0.85, 0.3,
0.25), less the PUMA shifts this model omits.

## Parameterization

There is no intercept: the cutpoints carry the location (a constant in the
formula is collinear with them and dropped). The cutpoints are the last
coefficients, `cut1` ... `cut{J-1}`, estimated directly rather than as
increments, so their standard errors are reported as they are by Stata's
`ologit` and R's `polr`. statsmodels' `OrderedModel` estimates the first
cutpoint and the logs of the increments instead; its slopes, and their
standard errors, are the same as these, and its cutpoints transform into
these.

A Newton step that would put the cutpoints out of order is halved until it
does not; the log-likelihood is concave in the slopes and cutpoints together,
so this is safe.

## Levels

The outcome's levels are ordered as `levels=`, or else sorted: integers by
value, an Enum by its categories, strings alphabetically (which is rarely
what an ordered outcome wants: give `levels=` or make it an Enum).
`education` is an Enum; as strings it needs `levels=` for the same fit:

```python
strings = people.with_columns(pl.col("education").cast(pl.String))
ologit_stream(f"education ~ {X}", strings, levels=EDUCATION)
```

## Residuals

`resid()` gives, for each row of the estimation sample, the probability of
each level (`prob_<level>`) and two residuals for the outcome as a whole:

- `resid_prob`, Li and Shepherd's (2012) probability-scale residual,
  P(Y < y) - P(Y > y): in [-1, 1], 0 on average under the model, and the
  same as statsmodels' `OrderedModel` reports (`resid_prob`), which the test
  suite checks;
- `resid_gen`, the generalized residual (Gourieroux et al. 1987), the
  derivative of the row's log-likelihood in its linear index: its products
  with the covariates sum to zero at the estimates, as the score does.

```python
fit.resid().collect()
```

```text
┌────────┬──────────────┬──────────────────┬────────────┬───────────┐
│ row_id ┆ education    ┆ prob_High school ┆ resid_prob ┆ resid_gen │
╞════════╪══════════════╪══════════════════╪════════════╪═══════════╡
│ 0      ┆ Less than HS ┆ 0.276661         ┆ -0.849079  ┆ -0.849079 │
│ 1      ┆ Some college ┆ 0.038366         ┆ -0.816462  ┆ -0.816462 │
│ 2      ┆ Bachelor's   ┆ 0.048776         ┆ -0.427149  ┆ -0.427149 │
│ 3      ┆ Bachelor's   ┆ 0.228077         ┆ 0.422086   ┆ 0.422086  │
└────────┴──────────────┴──────────────────┴────────────┴───────────┘
```

For the ordered *logit* the two are the same. With $a = F(\kappa_{j-1} -
x'\beta)$ and $b = F(\kappa_j - x'\beta)$ for a row at level $j$, the
probability-scale residual is $a - (1 - b)$, and since the logistic
density is $F(1 - F)$, the generalized residual is
$(a(1-a) - b(1-b)) / (b - a) = a + b - 1$ too. For the ordered probit they
differ (the generalized residual is unbounded); the script checks both:

```text
logit: largest |resid_prob - resid_gen| 7.88e-15
probit: largest |resid_prob - resid_gen| 2.15e+00
```

## Predictions

`predict(type="response")` (the default) returns each level's probability,
`type="class"` the most probable level, `type="link"` the index
$x'\beta$ (with the fixed effect, if any):

```python
fit.predict(new).collect()
fit.predict(new, type="class").collect()
```

```text
┌───────────┬───────────────────┬────────────────┬───────────────────┬─────────────────┬───────────────┐
│ person_id ┆ prob_Less than HS ┆ prob_High      ┆ prob_Some         ┆ prob_Bachelor's ┆ prob_Graduate │
│           ┆                   ┆ school         ┆ college           ┆                 ┆               │
╞═══════════╪═══════════════════╪════════════════╪═══════════════════╪═════════════════╪═══════════════╡
│ 0         ┆ 0.109424          ┆ 0.231098       ┆ 0.254371          ┆ 0.241318        ┆ 0.16379       │
│ 1         ┆ 0.028216          ┆ 0.080533       ┆ 0.148869          ┆ 0.289165        ┆ 0.453218      │
│ 2         ┆ 0.026633          ┆ 0.076496       ┆ 0.143304          ┆ 0.285609        ┆ 0.467958      │
└───────────┴───────────────────┴────────────────┴───────────────────┴─────────────────┴───────────────┘
```

## The ordered probit

`oprobit_stream` is the same model with normal errors, and takes the same
options except random intercepts and `firth=`. Its coefficients are the
logit's divided by about 1.7, the ratio of the two distributions' scales,
and it fits these (logistic) data slightly worse:

```python
probit = oprobit_stream(f"education ~ {X}", people)
```

```text
┌─────────────┬───────────┬──────────┬──────────┐
│ coefficient ┆ logit     ┆ probit   ┆ ratio    │
╞═════════════╪═══════════╪══════════╪══════════╡
│ age10       ┆ 0.045762  ┆ 0.02662  ┆ 1.719099 │
│ female      ┆ 0.242993  ┆ 0.140909 ┆ 1.724473 │
│ log_income  ┆ 0.811634  ┆ 0.475319 ┆ 1.707556 │
│ ...
log-likelihood: logit -300550.7, probit -300686.3
```

## Fixed effects

With an absorbed fixed effect (`"education ~ age10 + female | puma"`), each
PUMA's effect shifts its rows' latent index, and `cut1` is normalized to 0
(the effects carry the location; `cutpoints` reports the 0), so the table
starts at `cut2`. See [fixed effects](fixed_effects.md), and for random
PUMA shifts, [random intercepts](random_intercepts.md).

Everything else -- standard errors, weights, `l2`, the options -- is as for
the [multinomial logit](multinomial.md).

:::{dropdown} The full script: `examples/ordered.py`
```{literalinclude} ../../examples/ordered.py
:language: python
```
:::
