# Conditional logit

{download}`Download the script: conditional.py <../../examples/conditional.py>`

McFadden's conditional logit, for choices among alternatives that have
attributes of their own:

$$
P(i \text{ chooses } a) = \frac{\exp(x_{ia}'\beta)}{\sum_{b \in C_i} \exp(x_{ib}'\beta)},
$$

with $C_i$ the alternatives available to chooser $i$.

```python
from logit_stream import clogit_stream
from logit_stream.simulate import simulate_choices

choices = simulate_choices(50_000, seed=1)       # one row per available alternative
fit = clogit_stream("chosen ~ cost + time + C(alt) + income:C(alt)", choices,
                    case="case", vcov="hetero")
fit.summary()
```

```text
### chosen ~ cost + time + C(alt) + income:C(alt)
Conditional logit (logit_stream, Newton)   vcov: hetero
choice sets: 50,000   alternatives (rows): 134,988   case: case
log-likelihood: -39388.55   null: -48517.828   pseudo R2: 0.188163   AIC: 78789.099   BIC: 78842.018
Newton steps: 5 (converged, decrement 3.13e-10)
dropped as collinear: ['income:C(alt)[train]']
┌──────────────────────┬───────────┬────────────┬────────────┬─────────────┐
│ Coefficient          ┆ Estimate  ┆ Std. Error ┆ t value    ┆ Pr(>|t|)    │
╞══════════════════════╪═══════════╪════════════╪════════════╪═════════════╡
│ cost                 ┆ -1.066812 ┆ 0.01502    ┆ -71.027893 ┆ 0.0         │
│ time                 ┆ -1.276233 ┆ 0.014123   ┆ -90.362828 ┆ 0.0         │
│ C(alt)[T.metro]      ┆ 0.698818  ┆ 0.014122   ┆ 49.483069  ┆ 0.0         │
│ C(alt)[T.train]      ┆ 0.313728  ┆ 0.01505    ┆ 20.846348  ┆ 1.6451e-96  │
│ income:C(alt)[car]   ┆ 0.283802  ┆ 0.013351   ┆ 21.256278  ┆ 2.8840e-100 │
│ income:C(alt)[metro] ┆ -0.000973 ┆ 0.012268   ┆ -0.079313  ┆ 0.936784    │
└──────────────────────┴───────────┴────────────┴────────────┴─────────────┘
```

The simulation's coefficients are -1.1 on cost and -1.3 on time, constants
of 0.7 and 0.3 for metro and train against the car, and 0.3 on income for
the car (against the train, whose income coefficient is the one dropped).
The value of time, the ratio of the time and cost coefficients, is 1.196
(1.182 in the simulation).

## Data

The data are in long format: one row per alternative of each choice set
(`case=` names the column, or columns, that identify a set), the outcome 1 on
the chosen alternative and 0 on the others. Every set must have exactly one
chosen alternative after rows with missing values are dropped, which is
checked. Alternatives a chooser did not have are simply absent, so sets can
have different sizes:

```text
┌──────┬───────┐
│ size ┆ sets  │
╞══════╪═══════╡
│ 1    ┆ 1133  │
│ 2    ┆ 12746 │
│ 3    ┆ 36121 │
└──────┴───────┘
```

(A set with one alternative says nothing about the coefficients, and is
harmless.) A set with an alternative whose values are missing is dropped
whole, with a warning, as Stata's `cmclogit` does; `altwise=True` drops only
the incomplete alternatives.

The rows of a set need not be consecutive. Pass 0 checks whether they are
(long-format data usually are), and if not, sorts them by set out of core:
it hash-partitions the rows into buckets small enough to sort in memory, as
hdfe-stream does with its fixed effects. Shuffling the rows gives the same
estimates.

## Covariates

There is no intercept: anything constant within a choice set cancels from the
probabilities. Alternative-specific constants are `C(alt)` (or your own
indicators); a characteristic of the chooser enters interacted with the
alternatives (`income:C(alt)`), one of whose coefficients is then not
identified and is dropped as collinear, which fixes its level to zero (the
reference). A variable that does not vary within any set is dropped likewise.

`offset=` adds a known term to each alternative's utility, such as
McFadden's correction for sampled alternatives (the log of each
alternative's probability of being in the sampled set).

## Inference

The independent unit is the choice set: the robust variance sums the scores
over each set's rows, so it equals the variance clustered by the set (with
the same small-sample factor), and a clustering must be constant within sets,
as must weights. The null log-likelihood is that of equal shares within each
set, $-\sum_i \log |C_i|$, which xlogit and Biogeme report too.

Repeated choices by the same person are not independent of one another.
With five choice sets per person, cluster by the person:

```python
panel = simulate_choices(50_000, tasks=5, seed=1)
clogit_stream("chosen ~ cost + time + C(alt) + income:C(alt)", panel, case="case",
              vcov={"CRV1": "person"})
```

(Here the five choices share only the person's income, so clustering
changes the standard errors little. When they share a taste, the mixed
logit models it: see [the mixed logit](mixed.md).)

## Predictions and scenarios

`predict()` returns each row's choice probability within its set, computed as
a window over the sets (`case=` names them in new data); `resid()` adds the
utility and the response residual:

```text
┌────────┬──────┬────────┬───────────┬──────────┬───────────┐
│ row_id ┆ case ┆ chosen ┆ xb        ┆ prob     ┆ resid     │
╞════════╪══════╪════════╪═══════════╪══════════╪═══════════╡
│ 0      ┆ 0    ┆ 1.0    ┆ -1.852232 ┆ 0.460618 ┆ 0.539382  │
│ 1      ┆ 0    ┆ 0.0    ┆ -2.06842  ┆ 0.371067 ┆ -0.371067 │
│ 2      ┆ 0    ┆ 0.0    ┆ -2.858966 ┆ 0.168315 ┆ -0.168315 │
└────────┴──────┴────────┴───────────┴──────────┴───────────┘
```

Predicting on changed data gives a scenario. With metro fares 20% higher:

```python
dearer = choices.with_columns(
    cost=pl.when(pl.col("alt") == "metro").then(1.2 * pl.col("cost")).otherwise("cost"))
fit.predict(dearer)
```

```text
┌───────┬─────────┬────────────┐
│ alt   ┆ now     ┆ fares +20% │
╞═══════╪═════════╪════════════╡
│ car   ┆ 0.25618 ┆ 0.269995   │
│ metro ┆ 0.38842 ┆ 0.354744   │
│ train ┆ 0.3554  ┆ 0.37526    │
└───────┴─────────┴────────────┘
```

The metro's lost riders go to the car and the train in proportion to their
shares, as the conditional logit requires (the independence of irrelevant
alternatives). The [nested logit](nested.md) relaxes that.

## The multinomial logit as a conditional logit

A multinomial logit is a conditional logit with alternative-specific constants
and every covariate interacted with the alternatives. Expanding each person
to one row per race:

```python
long = (people.select("person_id", "race", "age10", "female", "log_income")
        .join(pl.DataFrame({"alt": RACES}), how="cross")
        .with_columns(chosen=(pl.col("race") == pl.col("alt")).cast(pl.Int8),
                      alt=pl.col("alt").cast(pl.Enum(RACES))))
cl = clogit_stream("chosen ~ C(alt) + age10:C(alt) + female:C(alt) + log_income:C(alt)",
                   long, case="person_id")
```

```text
log-likelihoods: multinomial -21274.5015, conditional -21274.5015
Black:age10 -0.098192 -0.098192
```

The test suite checks that the two give the same estimates, standard
errors and log-likelihood, and that the conditional logit agrees with
statsmodels' `ConditionalLogit`.

## Absorbed alternative constants

A constant for each level of a characteristic of the alternatives -- an
establishment, a product, a destination -- is absorbed by naming it after
`|`, as for the other models' fixed effects:

```python
fit = clogit_stream("chosen ~ dist | estab", jobs, case="worker")
fit.fixef().head(3).collect()            # fe_estab, se_estab, n_obs, n_chosen
```

The constants vary across a choice set's alternatives, so they do not
cancel, as a characteristic of the chooser would. This is the conditional
logit that a Poisson regression with chooser and alternative fixed effects
reproduces, fitted as the conditional logit. More than one dimension is
allowed (``| estab + county_sex``); with several, only the constants' sums
are identified, and `fixef()` reports them as fitted.

**How they are solved.** The constants' information couples two levels
only when some choice set contains both. It is therefore block-diagonal
over the groups of levels that the choice sets link, which the first pass
finds from the data. With workers choosing among their firm's
establishments the groups are the firms, without the firm being named. With
`fe_solver="auto"` (the default) each group's block is dense, and Newton's
steps eliminate the constants exactly, group by group, when the largest
group has at most 5,000 levels and the blocks fit in memory. Otherwise
`"contraction"` solves the constants by count matching (each level's
predicted choices equal its observed ones) and the coefficients' information
net of them by conjugate gradients, with no blocks. Both give the same
estimates; the diagnostics' `alternative_constants` entry records which was
used. With 4.1 million rows (324,000 workers, 30,000 establishments in 5,000
firms, three of them with 150 establishments and 5,000 workers), the blocks
took 8 seconds and the contraction 18.

A level never chosen, or chosen in every set where it appears, has an
infinite constant. It is dropped with its alternatives (and, for the second
kind, its sets), with a warning. With one dimension each constant is
reported relative to its group's reference level (the most chosen, at 0).
Its standard error (`se_<name>`) is that of the difference, and
`draw_parameters` draws the coefficients and every constant jointly. Both
need one dimension and the blocks. Robust (``vcov="hetero"``) and clustered
(``{"CRV1": "firm"}``) variances work with either solver and any number of
dimensions. Each choice set's score for the coefficients is taken net of the
constants, which gives the same standard errors as the dummy-coded fit's.

**Calibrating to known totals.** `predict(..., calibrate={"estab": totals})`
solves each establishment's constant from a known total instead: the one at
which its predicted choices over `newdata`'s sets equal the total, given the
coefficients. That is what imputation needs where the establishments are new
(no workers of theirs were in the fit) but their employment is known. Each
draw from `draw_parameters` calibrates under its own coefficients. Within
each firm the totals are first scaled to its number of choice sets, with a
warning if they differ by more than 1%.

The solve needs all of a group's rows at once: a firm, or more generally the
levels that `newdata`'s choice sets link together. So `newdata` is streamed
through disk a batch of whole groups at a time, about 2 million rows per
batch, or the largest group if that is bigger. The groups are found by label
propagation over the (set, establishment) pairs on disk. Each batch's
constants come from contraction rounds, delta += log(total / predicted),
which converge in a few rounds when every worker in a firm faces all of its
establishments. When they slow down, as on long chains of linked levels,
Newton's method finishes the solve. The predictions come back as a lazy scan
of the batches' files, in `newdata`'s order, so they can be sunk to disk
without ever being in memory. The files are removed when Python exits. With
4.1 million rows (a 99 MB frame) the calibration took 4 seconds. Peak memory
rose by 0.5 GB with the default batches and by 0.36 GB with batches of
500,000 rows, including the collected result.

In a simulation (3,000 firms of 2 to 10 establishments and 5 to 150 workers
in the fit, 1,000 new firms to impute, true distance coefficient -1.2), the
two approaches compare as follows. In the first, the log of each
establishment's share of its firm's workers is an offset, both in the fit
and at imputation; in the second, the constants are absorbed and then
calibrated:

| | distance coefficient | mean commute (truth 1.797) | P(the true establishment) | workers placed against the totals |
|---|---|---|---|---|
| log-share offset | -1.012 | 1.805 | 0.445 | 18% misallocated |
| constants + calibration | -1.231 | 1.800 | 0.470 | exact |

The offset attenuates the coefficient by a sixth. Aggregate commutes barely
change, but the establishments' totals do not hold. The absorbed
constants' estimate is 2.5% too large: the maximum-likelihood estimate's
own bias with about 13 workers per establishment, as with other fixed
effects estimated from few observations each.

## Related models

- **Nests of alternatives:** `nlogit_stream`, the [nested logit](nested.md).
- **Coefficients that vary across choosers:** `mixlogit_stream`, the
  [mixed logit](mixed.md).
- **Fixed effects of the chooser** do not apply: a characteristic of the
  chooser cancels. Alternative constants do (above).
  For panels of binary outcomes, Chamberlain's conditional fixed-effects
  logit is `cfelogit_stream` (see [incidental
  parameters](incidental_parameters.md)).

:::{dropdown} The full script: `examples/conditional.py`
```{literalinclude} ../../examples/conditional.py
:language: python
```
:::
