# Predicting at scale

{download}`Download the script: imputation.py <../../examples/imputation.py>`

The simplest imputation: fit a model of a characteristic on a
survey or administrative extract where it is observed, then predict it, as
probabilities, for a much larger file where it is not. Here the survey is
200,000 simulated people with survey weights, one PUMA in fifty missing
from it, and the file to impute is a million records, in Parquet.

```python
import polars as pl
from logit_stream import mlogit_stream
from logit_stream.simulate import RACES

X = "age10 + female + log_income + owner + urban"
fit = mlogit_stream(f"race ~ {X} | puma", survey, levels=RACES, weights="weight",
                    vcov={"CRV1": "puma"})

(fit.predict(pl.scan_parquet("records.parquet"))    # lazily, batch by batch
    .select("person_id", pl.col("^prob_.*$"))
    .sink_parquet("race_probabilities.parquet"))
```

`predict` adds a `prob_<level>` column per outcome level to `newdata`, in its
own order, without collecting it: each batch's design is evaluated by the
fit's Polars expressions and multiplied by the coefficients in numpy, with the
fixed effects joined from the fit's table (a small table in memory). So
records far larger than memory stream through, at the speed of Polars'
streaming engine.

## What new data must have

Every column the model uses. A categorical covariate's indicators are those
of the fit: a level the fit did not see counts as the reference level, as in
hdfe-stream. A fixed-effect level the fit did not estimate (a PUMA absent
from the training data, or dropped for separation) gives null probabilities,
as does a null covariate; fill those as the application needs. The script
fills them from a model with state indicators instead of PUMA effects,
joined on the record's identifier and `coalesce`d:

```text
┌───────────┬──────────┬────────────┬────────────┬────────────┬────────────┬───────────┬───────────┐
│ person_id ┆ fallback ┆ prob_White ┆ prob_Black ┆ prob_Hispa ┆ prob_Asian ┆ prob_AIAN ┆ prob_Mult │
│           ┆          ┆            ┆            ┆ nic        ┆            ┆           ┆ iracial   │
╞═══════════╪══════════╪════════════╪════════════╪════════════╪════════════╪═══════════╪═══════════╡
│ 0         ┆ false    ┆ 0.687464   ┆ 0.099133   ┆ 0.023139   ┆ 0.168778   ┆ 0.003836  ┆ 0.017649  │
│ 1         ┆ false    ┆ 0.367503   ┆ 0.438244   ┆ 0.10805    ┆ 0.067978   ┆ 0.002396  ┆ 0.015828  │
│ 2         ┆ false    ┆ 0.620984   ┆ 0.227803   ┆ 0.052191   ┆ 0.072578   ┆ 0.008925  ┆ 0.01752   │
│ 3         ┆ false    ┆ 0.711477   ┆ 0.08682    ┆ 0.086328   ┆ 0.105802   ┆ 0.0       ┆ 0.009573  │
└───────────┴──────────┴────────────┴────────────┴────────────┴────────────┴───────────┴───────────┘
13,034 records in PUMAs the fit did not see
```

## Survey weights

Pass the survey weights as `weights=` (analytic weights, the default type):
the estimates are then those of the weighted pseudo-likelihood, and with
`vcov="hetero"` or a clustered vcov their standard errors are the usual
design-based ones (without strata, which the ACS's replicate weights handle
differently).

## Fixed or random effects

Record 3 above has a probability of exactly 0 of being AIAN: its PUMA's
survey sample has nobody AIAN, so the fixed effect for that level is minus
infinity. That is the maximum-likelihood estimate, but a poor prediction
when the PUMA's sample is small, and a record in that PUMA who is AIAN has
probability 0 for their true level. [Random intercepts](../examples/random_intercepts.md)
avoid both problems: small PUMAs' effects are shrunk toward the mean
instead of estimated on their own, and a PUMA the fit did not see is
predicted at the mean effect, 0, so no fallback model is needed:

```python
re = mlogit_stream(f"race ~ {X}", survey, levels=RACES, weights="weight", random="puma",
                   vcov={"CRV1": "puma"})
re.predict(pl.scan_parquet("records.parquet")).sink_parquet("race_probabilities_re.parquet")
```

Against the records' true race (which the simulation knows):

```text
┌───────────────────┬──────────────────────┬────────────────────────┬───────────────┐
│ model             ┆ mean log probability ┆ most probable is right ┆ probability 0 │
╞═══════════════════╪══════════════════════╪════════════════════════╪═══════════════╡
│ fixed effects     ┆ -inf                 ┆ 0.65268                ┆ 0.006592      │
│ random intercepts ┆ -1.063676            ┆ 0.655409               ┆ 0.0           │
└───────────────────┴──────────────────────┴────────────────────────┴───────────────┘
```

0.7% of the records have probability 0 for their true race under the fixed
effects. Both models are calibrated overall (their mean probabilities are
close to the shares of each level).

### The mean effect, or the average over the effects

By default `predict` puts each group's predicted effect into its rows'
probabilities: for a group the fit saw, its conditional mode, and for a
group it did not see, the mean, 0. A probability at the mean effect is not
the average probability over the effects' distribution. The logistic curve
is S-shaped, so averaging over the effects pulls probabilities toward 0.5:
the probability at the mean effect is too low for a rare outcome and too
high for a common one. For one row with linear predictor x'b and a group
effect with sd τ:

| x'b | at the mean effect | averaged, τ = 0.8 | averaged, τ = 1.5 |
|---|---|---|---|
| -3 | 0.047 | 0.061 | 0.097 |
| -2 | 0.119 | 0.143 | 0.190 |
| 0 | 0.500 | 0.500 | 0.500 |
| 1 | 0.731 | 0.707 | 0.671 |

The averaged probability is the expected share of such rows across groups
like the row's. So a total built from probabilities, such as the number of
people in a rare category in places the fit did not see, needs the averaged
ones: at the mean effect it would be undercounted, by almost a quarter at τ = 0.8
and x'b = -3, and by half at τ = 1.5.

:::{admonition} Experimental
:class: warning
`effects="average"` is checked against brute-force integration only, and
warns with `logit_stream.ExperimentalWarning`.
:::

`effects="average"` averages instead:

```python
re.predict(records, effects="average")     # prob_<level>, averaged over the effects
```

For a group the fit did not see, the average is over the prior, N(0,
$\Sigma$). For a group the fit saw, it is over the group's conditional
distribution: normal, with its mode as the mean and its conditional
covariance (whose diagonal `ranef` reports as `sd_<grouping>`). That
matters for small groups, whose conditional sd is large. The average is
by Gauss-Hermite quadrature: 32 nodes for the binary and ordered logit,
and for the multinomial logit 5 per dimension of its J - 1 intercepts
(3,125 points for six levels). That last costs about 16 seconds per
million rows on 16 CPUs, against a fraction of a second for the
probabilities at the mode. The tests check the averages against
brute-force integration, and on 20,000 simulated new groups with a rare
outcome, the mean averaged probability matches the share of 1s while the
probabilities at the mean effect undercount it.

Here only 1.3% of the records are in PUMAs the fit did not see, so the
choice barely matters for the totals. Draws are different: for
[multiple imputation](multiple_imputation.md), `draw_parameters(ranef=True,
newdata=...)` gives each group one draw of its effect, shared by its rows,
which keeps the within-group correlation that no single probability can.


The same script continues on the next page, [combining the predictions
with other sources](combining.md).

:::{dropdown} The full script: `examples/imputation.py`
```{literalinclude} ../../examples/imputation.py
:language: python
```
:::
