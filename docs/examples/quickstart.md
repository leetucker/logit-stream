# Quickstart

{download}`Download the script: quickstart.py <../../examples/quickstart.py>`

Every estimator takes a formula and a source, and returns a result with the
same reporting methods. The data here are `simulate_people`, a simulated
population in PUMAs nested in states with three demographic outcomes, and
`simulate_choices`, a mode-choice survey (see [the
examples](index.md#the-simulated-data)).

```python
import polars as pl
from logit_stream import (clogit_stream, logit_stream, mixlogit_stream, mlogit_stream,
                          nlogit_stream, ologit_stream)
from logit_stream.simulate import RACES, simulate_choices, simulate_people

simulate_people(200_000, seed=1).write_parquet("people.parquet")
```

The source can be a Parquet path or glob (`"people/*.parquet"`), a Polars
LazyFrame or a DataFrame. A path or a LazyFrame is never collected whole:
one pass writes the evaluated design as blocks, and each Newton step reads
them back (see [how it works](../how_it_works.md)).

## Binary logit

```python
fit = logit_stream("hispanic ~ age10 + female + log_income + owner + urban | puma",
                   "people.parquet", vcov={"CRV1": "puma"})
fit.summary()
```

```text
### hispanic ~ age10 + female + log_income + owner + urban | puma
Streaming HDFE (logit)   vcov: CRV1:puma
obs: 199,928   puma: 1,019
dropped for separation: 72 observations
FE dof: 1,019   deviance: 166607.57   log-likelihood: -83303.783   pseudo R2: 0.074157   within pseudo R2: 0.034597   IRLS steps: 6
┌─────────────┬───────────┬────────────┬────────────┬─────────────┐
│ Coefficient ┆ Estimate  ┆ Std. Error ┆ t value    ┆ Pr(>|t|)    │
╞═════════════╪═══════════╪════════════╪════════════╪═════════════╡
│ age10       ┆ -0.203589 ┆ 0.004636   ┆ -43.912405 ┆ 0.0         │
│ female      ┆ -0.004303 ┆ 0.013582   ┆ -0.316784  ┆ 0.751408    │
│ log_income  ┆ -0.307143 ┆ 0.007946   ┆ -38.65597  ┆ 0.0         │
│ owner       ┆ -0.2886   ┆ 0.018975   ┆ -15.20915  ┆ 3.0751e-52  │
│ urban       ┆ 0.507099  ┆ 0.014793   ┆ 34.279751  ┆ 1.5723e-257 │
└─────────────┴───────────┴────────────┴────────────┴─────────────┘
```

The binary logit is hdfe-stream's `feglm_stream(..., family="logit")`: any
number of fixed effects, and an `hdfe_stream.HDFEResult`. One PUMA where
nobody is Hispanic was dropped: its effect would be minus infinity. See
[binary outcomes](binary.md).

## Multinomial logit

```python
race = mlogit_stream("race ~ age10 + female + log_income + owner + urban + C(state)",
                     "people.parquet", levels=RACES)
race.coef_matrix()          # one column per non-base level
```

```text
┌─────────────┬───────────┬───────────┬───────────┬───────────┬─────────────┐
│ Coefficient ┆ Black     ┆ Hispanic  ┆ Asian     ┆ AIAN      ┆ Multiracial │
╞═════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════════════╡
│ Intercept   ┆ 1.912202  ┆ 2.028708  ┆ -4.925554 ┆ 0.623884  ┆ -2.581456   │
│ age10       ┆ -0.096861 ┆ -0.238997 ┆ -0.052485 ┆ -0.024918 ┆ -0.294096   │
│ female      ┆ 0.044466  ┆ 0.007317  ┆ 0.028917  ┆ 0.001275  ┆ 0.038708    │
│ log_income  ┆ -0.345641 ┆ -0.303357 ┆ 0.246604  ┆ -0.454916 ┆ -0.030683   │
│ ...
```

The coefficients are named `"<level>:<covariate>"` (`Black:age10`), each
level's log-odds against the base, the first of `levels=` (by default the
levels are sorted). See [the multinomial logit](multinomial.md).

## Ordered logit

```python
educ = ologit_stream("education ~ age10 + female + log_income + owner + urban | puma",
                     "people.parquet", vcov="hetero", cluster="puma")
educ.cutpoints                              # [0, 1.5, 2.605, 3.919]: cut1 is 0 with a fixed effect
educ.with_vcov({"CRV1": "puma"}).tidy()     # clustered, computed at fit time
```

`education` is a Polars Enum, so its levels are in the Enum's order; a
string outcome needs `levels=` to be ordered other than alphabetically.
`vcov=` is the variance reported, and `cluster=` adds clustered variances to
switch to afterwards with `with_vcov()`. See [the ordered logit](ordered.md).

## Conditional logit

Choice data are in long format, one row per alternative a chooser had:

```python
choices = simulate_choices(20_000, seed=1)
```

```text
┌──────┬────────┬───────┬──────────┬──────────┬───────────┬────────┐
│ case ┆ person ┆ alt   ┆ cost     ┆ time     ┆ income    ┆ chosen │
╞══════╪════════╪═══════╪══════════╪══════════╪═══════════╪════════╡
│ 0    ┆ 0      ┆ train ┆ 0.873804 ┆ 1.326497 ┆ -0.025688 ┆ 0      │
│ 0    ┆ 0      ┆ metro ┆ 1.084125 ┆ 0.602175 ┆ -0.025688 ┆ 1      │
│ 0    ┆ 0      ┆ car   ┆ 1.70636  ┆ 0.640045 ┆ -0.025688 ┆ 0      │
│ 1    ┆ 1      ┆ train ┆ 0.546318 ┆ 0.593783 ┆ 1.220234  ┆ 1      │
│ 1    ┆ 1      ┆ car   ┆ 1.116408 ┆ 1.376573 ┆ 1.220234  ┆ 0      │
│ ...
```

```python
mode = clogit_stream("chosen ~ cost + time + C(alt) + income:C(alt)", choices,
                     case="case")
```

```text
┌──────────────────────┬───────────┬────────────┐
│ Coefficient          ┆ Estimate  ┆ Std. Error │
╞══════════════════════╪═══════════╪════════════╡
│ cost                 ┆ -1.090395 ┆ 0.024031   │
│ time                 ┆ -1.328022 ┆ 0.022693   │
│ C(alt)[T.metro]      ┆ 0.702338  ┆ 0.022375   │
│ C(alt)[T.train]      ┆ 0.326122  ┆ 0.023765   │
│ income:C(alt)[car]   ┆ 0.291912  ┆ 0.021479   │
│ income:C(alt)[metro] ┆ -0.001697 ┆ 0.019579   │
└──────────────────────┴───────────┴────────────┘
```

`case=` names the choice sets. `C(alt)` gives alternative-specific
constants; the chooser's income, the same for every alternative, enters
interacted with them, and one of those interactions is dropped as collinear
(with a warning). See [the conditional logit](conditional.md).

## Nested and mixed logit

The nested logit groups alternatives that are closer substitutes; here the
data are drawn with train and metro in a nest with dissimilarity
parameter 0.5. The nested and mixed logit refuse collinear covariates
rather than drop them, so income enters as one column for the car:

```python
car = (pl.col("income") * (pl.col("alt") == "car")).alias("income_car")
nests = {"rail": ["train", "metro"], "road": ["car"]}
nested = simulate_choices(20_000, nests=nests, lambdas=0.5, seed=1).with_columns(car)
nl = nlogit_stream("chosen ~ cost + time + C(alt) + income_car", nested, case="case",
                   alt="alt", nests=nests)
```

```text
│ lambda.rail     ┆ 0.505899  ┆ 0.013479   │
```

The mixed logit lets coefficients vary across choosers. Here each person
makes ten choices, with a time coefficient of their own drawn from a
normal distribution with sd 1:

```python
panel = simulate_choices(20_000, tasks=10, sd_time=1.0, seed=1).with_columns(car)
mx = mixlogit_stream("chosen ~ cost + time + C(alt) + income_car", panel, case="case",
                     panel="person", random={"time": "normal"}, draws=200,
                     vcov={"CRV1": "person"})
```

```text
│ time            ┆ -1.341122 ┆ 0.03438    │
│ ...
│ sd.time         ┆ 0.983878  ┆ 0.035214   │
```

See [the nested logit](nested.md) and [the mixed logit](mixed.md).

## Reporting

Results have hdfe-stream's reporting methods:

```python
race.summary()                              # printed report
race.summary_json("race.json", coefficients=True)   # JSON, for pipelines
race.confint(joint=True, keep="female")     # simultaneous intervals
race.wald_test(keep="log_income")           # chi-squared, by default
race.with_vcov({"CRV1": "puma"})            # another vcov computed at fit time
```

```text
┌────────────────────┬───────────┬──────────┐
│ Coefficient        ┆ 2.5%      ┆ 97.5%    │
╞════════════════════╪═══════════╪══════════╡
│ Black:female       ┆ 0.011001  ┆ 0.077931 │
│ Hispanic:female    ┆ -0.035001 ┆ 0.049635 │
│ Asian:female       ┆ -0.014444 ┆ 0.072277 │
│ AIAN:female        ┆ -0.118751 ┆ 0.121302 │
│ Multiracial:female ┆ -0.077234 ┆ 0.15465  │
└────────────────────┴───────────┴──────────┘
{'statistic': 3831.059227999039, 'pvalue': 0.0, 'df1': 5, 'df2': None, 'distribution': 'chi2'}
```

and `logit_stream.etable([fit1, fit2])` makes a regression table (through
pyfixest's maketables; see [the multinomial logit](multinomial.md#several-models-and-a-table)).

## Predictions

`predict()`, `resid()` and `sample()` return Polars LazyFrames, computed when
collected or sunk, so they work on data of any size:

```python
simulate_people(100_000, seed=2).write_parquet("new_people.parquet")
race.predict("new_people.parquet").sink_parquet("race_probabilities.parquet")
educ.predict("new_people.parquet", type="class")    # the most probable level
mode.predict(choices)                               # each alternative's probability
```

```text
┌───────────┬────────────┬────────────┬───────────────┬────────────┬───────────┬──────────────────┐
│ person_id ┆ prob_White ┆ prob_Black ┆ prob_Hispanic ┆ prob_Asian ┆ prob_AIAN ┆ prob_Multiracial │
╞═══════════╪════════════╪════════════╪═══════════════╪════════════╪═══════════╪══════════════════╡
│ 0         ┆ 0.815131   ┆ 0.070111   ┆ 0.038708      ┆ 0.055168   ┆ 0.012033  ┆ 0.008848         │
│ 1         ┆ 0.610945   ┆ 0.205745   ┆ 0.077714      ┆ 0.094728   ┆ 0.004909  ┆ 0.00596          │
│ 2         ┆ 0.576954   ┆ 0.22796    ┆ 0.094314      ┆ 0.083071   ┆ 0.006051  ┆ 0.01165          │
└───────────┴────────────┴────────────┴───────────────┴────────────┴───────────┴──────────────────┘
```

For predicting at scale, see [imputation](imputation.md).

:::{dropdown} The full script: `examples/quickstart.py`
```{literalinclude} ../../examples/quickstart.py
:language: python
```
:::
