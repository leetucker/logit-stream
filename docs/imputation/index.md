# Imputation

A common use of a fitted logit model is imputation: filling in a
characteristic for records that lack it, from a model fitted where it is
observed. logit-stream's estimators are built for data that do not fit in
memory, and the methods in this section carry that over to imputation.
Predictions, draws and assignments are lazy Polars frames, computed batch
by batch, so a file of any size streams through.

The pages follow the steps of an imputation, from the simplest to the most
complete:

| | |
|---|---|
| [Predicting at scale](predicting.md) | the model's probabilities for a large file: what new data must have, survey weights, fixed or random effects, groups the fit did not see |
| [Combining with other sources](combining.md) | updating the model's probabilities with evidence from another source by Bayes' rule |
| [Multiple imputation](multiple_imputation.md) | drawing the parameters (`draw_parameters`) and each group's intercept (`draw_ranef`), for imputations that carry the estimates' uncertainty |
| [Choices with known totals](known_totals.md) | imputing a choice among alternatives whose totals are known: calibrating the alternatives' constants to the totals, and drawing assignments that match them exactly or approximately (experimental) |

## Which do I need?

- **Probabilities only**, e.g. for weighting, or to estimate a total as
  the sum of each record's probability: `predict`, on the
  [first page](predicting.md).
- **Imputed values** that analyses of the completed data can treat as
  data, with standard errors that include the imputation's uncertainty:
  [multiple imputation](multiple_imputation.md). Each imputation draws the
  parameters (`draw_parameters`), then the values
  (`predict(type="draw")`); `pool` and `pool_matrix` combine the analyses.
- **Imputed choices that must add up** to known totals, such as each
  alternative's number of choosers:
  [calibration and constrained assignment](known_totals.md).

The models, data and settings on these pages are generic examples, made
up to show the methods. They do not describe any particular application
in use.

```{toctree}
:hidden:

predicting
combining
multiple_imputation
known_totals
```
