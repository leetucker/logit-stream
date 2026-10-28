# Imputation at scale

{download}`Download the script: imputation.py <../../examples/imputation.py>`

The use the package was written for: fit a model of a characteristic on a
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
probability 0 for their true level. [Random intercepts](random_intercepts.md)
avoid both problems: small PUMAs' effects are shrunk toward the mean
instead of estimated on their own, and a PUMA the fit did not see gets the
mean effect, so no fallback model is needed:

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

## Combining with other information

The probabilities from `predict` are the model's distribution of the
outcome given the covariates. Information from another source that bears on
the outcome can be combined with them by Bayes' rule: if the other source
gives evidence $s$, with known probabilities $P(s \mid y = j)$ for each level
$j$, then

$$
P(y = j \mid x, s) = \frac{P(y = j \mid x)\, P(s \mid y = j)}
                          {\sum_l P(y = l \mid x)\, P(s \mid y = l)} ,
$$

the model's prediction as the prior and the other source as the
likelihood. Equivalently, the posterior log-odds of level $j$ against the
base are the prior log-odds (`predict(type="link")`, the `xb_<level>`
columns) plus the log-likelihood ratio $\log P(s \mid j) - \log P(s \mid
\text{base})$.

This pattern -- a model-based prior updated with evidence the model did not
use -- is old and widespread. Some examples:

- **Diagnostic testing.** A clinical prediction model, often a logistic
  regression of risk factors, gives a patient's pre-test probability of a
  condition; a test result's likelihood ratio, estimated in validation
  studies of the test, converts it to the post-test probability
  (post-test odds = pre-test odds x likelihood ratio; Deeks and Altman
  2004).
- **Individual tastes from choice histories.** A choice model estimated on
  a population (a mixed logit, say) gives the distribution of tastes, the
  prior for any one person; that person's own observed choices are the
  likelihood, and the posterior is their individual-level taste parameters,
  used for targeting and for forecasting their future choices (Rossi,
  McCulloch and Allenby 1996; Train 2009, ch. 11).
- **Demographic characteristics from names.** Imputations of race and
  ethnicity have combined a prior from a person's place of residence, and
  sometimes their age, sex and other characteristics, with likelihoods
  $P(\text{name} \mid \text{group})$ tabulated from surname or first-name
  lists (Elliott et al. 2009; Imai and Khanna 2016; Voicu 2018).

The script simulates a generic version: each record gets a signal (a
category A to D) from a second source, with known probabilities
$P(s \mid \text{race})$, and the posterior is a join and a normalization
in Polars:

```python
lik = LIKELIHOOD.unpivot(index="signal", variable_name="race", value_name="lik")
posterior = (imputed.unpivot(index="person_id", on=probs, variable_name="race",
                             value_name="prior")
             .with_columns(pl.col("race").str.strip_prefix("prob_"))
             .join(evidence, on="person_id")              # each record's signal
             .join(lik, on=["signal", "race"])             # P(signal | race)
             .with_columns(post=pl.col("prior") * pl.col("lik"))
             .with_columns(pl.col("post") / pl.col("post").sum().over("person_id")))
```

```text
┌───────────────────┬──────────────────────┬────────────────────────┬───────────────┐
│ model             ┆ mean log probability ┆ most probable is right ┆ probability 0 │
╞═══════════════════╪══════════════════════╪════════════════════════╪═══════════════╡
│ random intercepts ┆ -1.063676            ┆ 0.655409               ┆ 0.0           │
│ and the signal    ┆ -0.809572            ┆ 0.706914               ┆ 0.0           │
└───────────────────┴──────────────────────┴────────────────────────┴───────────────┘
```

The posterior is sharper, and remains calibrated (its mean probabilities
are still the shares). What makes that work, and what to check in practice:

- **Conditional independence.** The rule assumes the evidence is
  independent of the model's covariates given the outcome. If it is not --
  if the signal also varies with income or place within each level -- the
  information they share is counted twice, and the posterior is
  overconfident.
- **If the evidence is in the training data, put it in the model.** A
  signal observed alongside the outcome in the survey is better entered as
  a covariate (with its interactions), which estimates its relation to the
  outcome and to the other covariates jointly. Bayes' rule is for evidence
  that is available only separately, as a table of $P(s \mid y)$.
- **The same population.** $P(s \mid y)$ must hold for the records being
  imputed, not only where it was tabulated.
- **No zeros.** A prior of exactly 0 stays 0 whatever the evidence, which
  is another reason to prefer random intercepts (or a ridge penalty, `l2=`)
  to fixed effects for priors.
- **Check calibration** on records whose outcome is known, if any are: the
  mean posterior probability of each level against its share, overall and
  in groups.

## References

- Deeks, J. J. and D. G. Altman (2004). Diagnostic tests 4: likelihood
  ratios. *BMJ* 329(7458), 168--169.
- Elliott, M. N., P. A. Morrison, A. Fremont, D. F. McCaffrey, P. Pantoja
  and N. Lurie (2009). Using the Census Bureau's surname list to improve
  estimates of race/ethnicity and associated disparities. *Health Services
  and Outcomes Research Methodology* 9(2), 69--83.
- Imai, K. and K. Khanna (2016). Improving ecological inference by
  predicting individual ethnicity from voter registration records.
  *Political Analysis* 24(2), 263--272.
- Rossi, P. E., R. E. McCulloch and G. M. Allenby (1996). The value of
  purchase history data in target marketing. *Marketing Science* 15(4),
  321--340.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation*, 2nd ed.
  Cambridge University Press. Chapter 11, "Individual-level parameters."
- Voicu, I. (2018). Using first name information to improve race and
  ethnicity classification. *Statistics and Public Policy* 5(1), 1--13.

:::{dropdown} The full script: `examples/imputation.py`
```{literalinclude} ../../examples/imputation.py
:language: python
```
:::
