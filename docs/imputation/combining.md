# Combining with other sources

{download}`Download the script: imputation.py <../../examples/imputation.py>`

This page continues the example of [predicting at scale](predicting.md),
with its random-intercepts model of race and ethnicity.

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
