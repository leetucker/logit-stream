# Incidental parameters

{download}`Download the script: incidental_parameters.py <../../examples/incidental_parameters.py>`

A fixed effect estimated from few observations is estimated badly, and in a
nonlinear model that error does not average out: it biases the
coefficients of every covariate. This is the incidental-parameter problem
(Neyman and Scott 1948). It matters for panels of individuals observed a
few times each -- people over a few survey waves, firms over a few years --
and not for groups with many rows each, such as PUMAs, counties or schools,
for which absorbing a fixed effect is the right tool (see [fixed
effects](fixed_effects.md)).

For the binary logit the bias is famously large: with two periods, the
fixed-effects estimates converge to *twice* the true coefficients
(Andersen 1970; Abrevaya 1997). It falls as $1/T$ with the number of
periods $T$. logit-stream has two remedies for it, for the binary logit:
the analytic bias correction, and Chamberlain's conditional logit, which
avoids the bias altogether.

## The bias, and two remedies

`simulate_panel` draws balanced panels with a binary outcome,
$P(y_{it} = 1) = \Lambda(\alpha_i + 1.0\, x_{1it} - 0.5\, x_{2it})$, with
the unit effects $\alpha_i$ correlated with $x_1$. With about 20,000 rows
each, and panels of 2 to 20 periods:

```python
import polars as pl
from logit_stream import cfelogit_stream, logit_stream, mlogit_stream
from logit_stream.simulate import simulate_panel

panel = simulate_panel(n_units=20_000 // T, periods=T, seed=T)
fe = logit_stream("y ~ x1 + x2 | unit", panel)                        # fixed effects
bc = logit_stream("y ~ x1 + x2 | unit", panel, bias_correction=True)  # corrected
cfe = cfelogit_stream("y ~ x1 + x2 | unit", panel)                    # conditional
```

```text
x1 (truth 1.0):
┌─────┬───────┬──────────┬────────────────┬─────────────┬────────────────┐
│ T   ┆ units ┆ FE logit ┆ bias-corrected ┆ conditional ┆ conditional se │
╞═════╪═══════╪══════════╪════════════════╪═════════════╪════════════════╡
│ 2   ┆ 3826  ┆ 2.121804 ┆ -0.286282      ┆ 1.060902    ┆ 0.037015       │
│ 3   ┆ 3832  ┆ 1.609728 ┆ 0.782674       ┆ 0.99475     ┆ 0.028639       │
│ 4   ┆ 3462  ┆ 1.442644 ┆ 0.955286       ┆ 1.014671    ┆ 0.026185       │
│ 6   ┆ 2686  ┆ 1.250438 ┆ 0.991791       ┆ 1.001767    ┆ 0.023981       │
│ 10  ┆ 1827  ┆ 1.157703 ┆ 1.016559       ┆ 1.018936    ┆ 0.022389       │
│ 20  ┆ 965   ┆ 1.081461 ┆ 1.016803       ┆ 1.01709     ┆ 0.021866       │
└─────┴───────┴──────────┴────────────────┴─────────────┴────────────────┘
```

(`units` counts those whose outcome varies; the others carry no
information about the coefficients and are dropped by all three.) The
fixed-effects estimate is twice the truth at $T = 2$ and still 8% too large
at $T = 20$. The bias correction removes most of the bias from $T = 4$ or
so, and is useless below: it is a first-order expansion in $1/T$. The
conditional logit is consistent at every $T$, within a standard error or
two of the truth throughout.

## The analytic bias correction

`bias_correction=True` (for `logit_stream` and `probit_stream`, from
hdfe-stream) applies the analytic correction of Fernández-Val and Weidner
(2016), which builds on Hahn and Newey (2004) and Fernández-Val (2009),
computed as R's alpaca computes it (`biasCorr`; hdfe-stream's tests match
it to 1e-7). It estimates the leading $O(1/T)$ term of the bias from the
fitted model and subtracts it; then the fixed effects are re-estimated with
the corrected coefficients held, and the variances, residuals, fixed
effects and predictions are those of that fit. At $T = 4$:

```python
bc = logit_stream("y ~ x1 + x2 | unit", panel, bias_correction=True)
bc.diagnostics["bias_correction"]
```

```text
uncorrected: {'x1': 1.443, 'x2': -0.712}
bias:        {'x1': 0.487, 'x2': -0.239}
corrected:   {'x1': 0.955, 'x2': -0.473}
```

What it handles that the conditional logit does not:

- **the probit** as well as the logit;
- **two fixed effects** (unit and period, `| unit + year`), both corrected;
- **predetermined regressors**, such as a lagged outcome:
  `bias_correction={"time": "year", "lags": 2}` adds the term from the
  scores' correlation within each unit over a window of L periods;
- **the fixed effects and probabilities**: the fit estimates the unit
  effects, so it predicts probabilities and has fitted values and
  residuals.

It warns when the median rows per level of a corrected effect is below
five; as the table shows, the warning is to be taken seriously. See
[hdfe-stream's guide](https://leetucker.github.io/hdfe-stream/examples/glm.html#incidental-parameter-bias)
for the details.

## Chamberlain's conditional logit

Given the number of 1s in a unit's outcomes, $k_i = \sum_t y_{it}$, the
probability of which periods they fell in does not depend on the unit's
effect: $\alpha_i$ is in every period's odds equally, so it cancels from

$$
P\Big(y_{i1}, \dots, y_{iT} \,\Big|\, \textstyle\sum_t y_{it} = k_i\Big) =
\frac{\exp\big(\sum_t y_{it}\, x_{it}'\beta\big)}
     {\sum_{d \in B_{k_i}} \exp\big(\sum_t d_t\, x_{it}'\beta\big)},
$$

with $B_k$ the arrangements of $k$ 1s among the unit's $T$ periods. The
conditional likelihood has no incidental parameters, so its estimates are
consistent however few the periods (Rasch 1960; Andersen 1970; Chamberlain
1980): it is Stata's `xtlogit, fe` and `clogit`, and R's `survival::clogit`.

```python
cfe = cfelogit_stream("y ~ x1 + x2 | unit", panel, vcov="hetero")
cfe.summary()
```

```text
### y ~ x1 + x2 | unit
Conditional fixed-effects logit (logit_stream, Newton)   vcov: hetero
units: 3,462 with variation (1,538 all 0 or all 1, dropped)   rows: 13,848   unit: unit
log-likelihood: -4008.9359   null: -5303.3442   pseudo R2: 0.244074   AIC: 8021.8718   BIC: 8034.171
Newton steps: 5 (converged, decrement 8.68e-11)
┌─────────────┬───────────┬────────────┬───────────┬─────────────┐
│ Coefficient ┆ Estimate  ┆ Std. Error ┆ t value   ┆ Pr(>|t|)    │
╞═════════════╪═══════════╪════════════╪═══════════╪═════════════╡
│ x1          ┆ 1.014671  ┆ 0.026503   ┆ 38.285351 ┆ 0.0         │
│ x2          ┆ -0.502678 ┆ 0.022535   ┆ -22.30616 ┆ 3.2198e-110 │
└─────────────┴───────────┴────────────┴───────────┴─────────────┘
```

- **The formula's one fixed effect is the unit.** Other effects, such as
  periods, enter as indicators (`+ C(year)`); covariates constant within
  every unit are dropped, since their effect is the unit's.
- **Units whose outcome never varies** carry no information and are
  dropped (`n_units_dropped`).
- **The unit effects are conditioned out, not estimated**, so there are no
  predicted probabilities, residuals or average partial effects:
  `predict()` gives only the linear index $x'\beta$, and `fixef()` and
  `resid()` raise. The coefficients are log odds ratios within a unit.
- **The robust variance** treats units as the observations (clustered by
  unit); weights and clusters must be constant within units. The
  log-likelihood is the conditional one, and its null has every
  arrangement of a unit's 1s equally likely.
- **The cost** is a recursion over each unit's rows for the elementary
  symmetric polynomial of degree $k_i$: $O(T k P^2)$ per unit ($k \le
  T/2$, by symmetry), so long panels with many 1s are slow. Each unit's
  rows are written together in pass 0, sorted if they are not already.
- **Logit only.** There is no such sufficient statistic for the probit,
  and the conditional logit needs strictly exogenous covariates (no lagged
  outcome).

## Random effects, or ignoring the effects

Random intercepts (`random=`; see [random
intercepts](random_intercepts.md)) estimate no incidental parameters, but
assume the effects are independent of the covariates. Here they are not
($x_1$ is correlated with $\alpha$), and the random-effects estimates are
biased, as are those of the logit that ignores the effects; at $T = 4$:

```text
┌─────────────┬───────┬──────────┬────────────────┬───────────────┬─────────────┐
│ coefficient ┆ truth ┆ pooled   ┆ random effects ┆ fixed effects ┆ conditional │
╞═════════════╪═══════╪══════════╪════════════════╪═══════════════╪═════════════╡
│ x1          ┆ 1.0   ┆ 1.223204 ┆ 1.288213       ┆ 1.442644      ┆ 1.014671    │
│ x2          ┆ -0.5  ┆ -0.43976 ┆ -0.486827      ┆ -0.712199     ┆ -0.502678   │
└─────────────┴───────┴──────────┴────────────────┴───────────────┴─────────────┘
```

($x_2$ is independent of the effects, and the random-effects estimate of
its coefficient is close.) A common compromise, the correlated random
effects of Mundlak (1978), adds each unit's means of the time-varying
covariates to the random-effects model, to absorb the part of the effect
correlated with them; it assumes that part is linear in the means, as it is
in this simulation:

```python
means = panel.with_columns(x1_bar=pl.col("x1").mean().over("unit"),
                           x2_bar=pl.col("x2").mean().over("unit"))
mlogit_stream("y ~ x1 + x2 + x1_bar + x2_bar", means, levels=[0, 1], random="unit")
```

```text
correlated random effects: {'x1': 0.993, 'x2': -0.489}
```

Unlike the conditional logit, it estimates the effects' distribution, and
so predicts probabilities.

## Which to use

| | rows per group | probabilities and effects | probit | two-way, lags | note |
|---|---|---|---|---|---|
| fixed effects (`| unit`) | many | yes | yes | yes | biased as 1/T |
| `bias_correction=True` | about 5 or more | yes | yes | yes | first-order; poor for very short panels |
| `cfelogit_stream` | any | no (the index only) | no | periods as indicators; no lags | consistent for any T |
| random intercepts | any | yes | no | nested or crossed groupings | needs effects independent of the covariates |
| correlated random effects | any | yes | no | nested or crossed groupings | needs the effects' dependence on the covariates linear in their means |

## The multinomial and ordered logit

Neither correction is available for the multinomial or ordered logit with
a fixed effect, so with few rows per group their fixed-effects estimates
are biased in the same way. Conditional approaches exist for both --
Chamberlain (1980) for the multinomial logit, and the "blow-up and
cluster" estimator of Baetschmann, Staub and Winkelmann (2015) for the
ordered logit, which fits the binary conditional logit to every
dichotomization of the outcome -- but are not built in. The ordered one
takes a few lines: stack the dichotomizations $1[y \ge k]$, give each unit
and cut its own group, fit `cfelogit_stream`, and cluster by the unit. On
a simulated ordered panel of four periods:

```python
stacked = pl.concat([ordered.with_columns(yk=(pl.col("y") >= k).cast(pl.Int8),
                                          group=pl.col("unit") * 10 + k)
                     for k in (1, 2, 3)])
buc = cfelogit_stream("yk ~ x | group", stacked, vcov={"CRV1": "unit"})
```

```text
ordered logit, x (truth 1.0): fixed effects 1.206, blow-up and cluster 0.968 (se 0.026)
```

Firth's penalized likelihood (`firth=True`, [binary
outcomes](binary.md#separated-covariates-and-firths-penalized-likelihood))
corrects a different, $O(1/n)$ bias, of models with a fixed number of
parameters; it is not a remedy for incidental parameters, and is not
available with fixed effects.

## References

- Abrevaya, J. (1997). The equivalence of two estimators of the
  fixed-effects logit model. *Economics Letters* 55(1), 41--43.
- Andersen, E. B. (1970). Asymptotic properties of conditional
  maximum-likelihood estimators. *Journal of the Royal Statistical
  Society, Series B* 32(2), 283--301.
- Baetschmann, G., K. E. Staub and R. Winkelmann (2015). Consistent
  estimation of the fixed effects ordered logit model. *Journal of the
  Royal Statistical Society, Series A* 178(3), 685--703.
- Chamberlain, G. (1980). Analysis of covariance with qualitative data.
  *Review of Economic Studies* 47(1), 225--238.
- Fernández-Val, I. (2009). Fixed effects estimation of structural
  parameters and marginal effects in panel probit models. *Journal of
  Econometrics* 150(1), 71--85.
- Fernández-Val, I. and M. Weidner (2016). Individual and time effects in
  nonlinear panel models with large N, T. *Journal of Econometrics*
  192(1), 291--312.
- Firth, D. (1993). Bias reduction of maximum likelihood estimates.
  *Biometrika* 80(1), 27--38.
- Hahn, J. and W. Newey (2004). Jackknife and analytical bias reduction
  for nonlinear panel models. *Econometrica* 72(4), 1295--1319.
- Mundlak, Y. (1978). On the pooling of time series and cross section
  data. *Econometrica* 46(1), 69--85.
- Neyman, J. and E. L. Scott (1948). Consistent estimates based on
  partially consistent observations. *Econometrica* 16(1), 1--32.
- Rasch, G. (1960). *Probabilistic Models for Some Intelligence and
  Attainment Tests.* Copenhagen: Danish Institute for Educational
  Research.

:::{dropdown} The full script: `examples/incidental_parameters.py`
```{literalinclude} ../../examples/incidental_parameters.py
:language: python
```
:::
