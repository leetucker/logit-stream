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
### chosen ~ 1 + cost + time + C(alt) + income:C(alt)
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
constants, which gives the same standard errors as the dummy-coded fit's. The
tests also check the estimate and its standard errors (iid, robust and
clustered) against pyfixest's `fepois` with choice-set and establishment
fixed effects, the same model as a Poisson regression.

**Calibrating to known totals.** Where the alternatives are new (no
choosers of theirs were in the fit) but each one's total is known, such as
an establishment's employment, `predict(..., calibrate={"estab": totals})`
solves their constants from the totals instead, given the coefficients.
Imputing the choices themselves to match the totals is the subject of
[choices with known totals](../imputation/known_totals.md).

## Random slopes by region

:::{admonition} Experimental
:class: warning
Random slopes in the conditional logit are new; their interface may change.
:::

A coefficient can vary across regions: the effect of distance on where
people work may differ between dense and sparse labour markets. With
`random={"cz": "dist"}`, each level of `cz` gets its own deviation b_g in
the distance coefficient, drawn from N(0, $\Sigma$), with $\Sigma$
estimated. The script for this section, `examples/regional_slopes.py`
(at the end of the page), simulates 40,000 workers each choosing among 5
of the 2,000 establishments in their home commuting zone, 60 zones whose
distance slopes have sd 0.3 (0.272 as drawn):

```python
fit = clogit_stream("chosen ~ dist | estab", jobs, case="worker", random={"cz": "dist"})
fit.summary()
fit.ranef().collect()          # each region's re_cz:dist and its conditional sd_cz:dist
```

```text
random slopes by cz (60 regions, the choice set's); Laplace approximation, sd (se): dist: 0.2549 (0.0255)
┌─────────────┬───────────┬────────────┬────────────┬─────────────┐
│ Coefficient ┆ Estimate  ┆ Std. Error ┆ t value    ┆ Pr(>|t|)    │
╞═════════════╪═══════════╪════════════╪════════════╪═════════════╡
│ dist        ┆ -0.984589 ┆ 0.034636   ┆ -28.427112 ┆ 9.3504e-178 │
└─────────────┴───────────┴────────────┴────────────┴─────────────┘
┌─────┬────────────┬────────────┬───────┬──────────┐
│ cz  ┆ re_cz:dist ┆ sd_cz:dist ┆ n_obs ┆ n_chosen │
╞═════╪════════════╪════════════╪═══════╪══════════╡
│ 0   ┆ -0.142993  ┆ 0.082217   ┆ 659   ┆ 659      │
│ 1   ┆ -0.015153  ┆ 0.074786   ┆ 677   ┆ 677      │
│ 2   ┆ 0.090469   ┆ 0.069408   ┆ 684   ┆ 684      │
│ ...
```

`fit.varcorr()` gives the sd with its standard error and confidence
interval. With about 670 workers a zone, the predicted slopes correlate
0.947 with the true ones.

`dist`'s coefficient is then the mean slope, and a region's slope is that
plus its deviation. Several covariates may vary (`{"cz": ["dist", "x2"]}`,
with their covariance). A row's region can be its choice set's (a worker's
home commuting zone, shared by every alternative in the set) or the
alternative's own (an establishment's commuting zone, which can differ
between a worker's alternatives). The data decide which, and a set may mix
regions. Absorbed constants (`| estab`) are allowed with either.

**How it is fitted.** The regions' slopes enter like extra coefficients,
pulled toward zero by their prior, $\Sigma^{-1}$. They are solved with the
coefficients and the absorbed constants in the same Newton steps, with the
constants eliminated group by group as above. $\Sigma$ maximizes Laplace's
approximation of the marginal likelihood, with the slopes integrated out,
as for the [random intercepts](random_intercepts.md). With regions of
hundreds or thousands of choice sets, the approximation is very close to
exact. The tests check the fit against exact maximum likelihood (a dense
grid around each region's mode), to 0.005 standard errors, and against an
independent implementation of the Laplace approximation where the regions'
slopes do not separate (the alternatives' regions, and two slopes), to
1e-8. In a simulation with 2.3 million rows (384,000 workers in 5,000
firms, 60 regions), the fit took 4.6 seconds, against 1.9 without the
slopes, and estimated the slopes' sd at 0.291 (the regions' realized sd was
0.283). The regions times the slopes are at most 2,000.

**Inference.** The coefficients' iid variance is net of the constants and
the slopes, given $\Sigma$. A clustered variance needs clusters that
contain whole regions (`{"CRV1": "state"}`, or the region itself);
`"hetero"` is not available, since a region's choice sets share its
slopes. `lr_test(fit_without, fit)` tests whether the slope varies at all,
with the boundary mixture for its variance:

```text
{'statistic': 475.28, 'df': 1, 'p': 1.1e-105, 'reference': '0.5 chi2(0) + 0.5 chi2(1)', 'boundary': True, ...}
```

**Imputation.** `predict` adds each region's slopes, and 0 for a region the
fit did not see. `predict(effects="average")` averages the probabilities
over the slopes' distribution instead: each region's conditional one, or
N(0, $\Sigma$) for an unseen region (for regions shared by a set's
alternatives). `draw_parameters(n, newdata=...)` draws the coefficients,
every region's slopes and the constants jointly, and $\Sigma$; the regions
in `newdata` that the fit did not see get draws from N(0, that draw's
$\Sigma$). Calibration to totals and constrained assignment then use each
draw's slopes (see [choices with known totals](../imputation/known_totals.md)).
For a worker choosing between a near and a far establishment, in zone 0
and in a zone the fit did not see:

```text
┌────────┬─────┬──────┬─────────────┬──────────┐
│ worker ┆ cz  ┆ dist ┆ conditional ┆ averaged │
╞════════╪═════╪══════╪═════════════╪══════════╡
│ 0      ┆ 0   ┆ 0.2  ┆ 0.840223    ┆ 0.839226 │
│ 0      ┆ 0   ┆ 2.0  ┆ 0.159777    ┆ 0.160774 │
│ 1      ┆ 999 ┆ 0.2  ┆ 0.802581    ┆ 0.792951 │
│ 1      ┆ 999 ┆ 2.0  ┆ 0.197419    ┆ 0.207049 │
└────────┴─────┴──────┴─────────────┴──────────┘
```

Zone 0's slope is well determined, so averaging changes little; for the
unseen zone it spreads over the slopes' sd of 0.25.

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

:::{dropdown} The random-slopes script: `examples/regional_slopes.py`
```{literalinclude} ../../examples/regional_slopes.py
:language: python
```
:::
