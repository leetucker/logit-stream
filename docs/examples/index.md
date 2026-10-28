# Examples

Each page here follows one of the scripts in
[`examples/`](https://github.com/leetucker/logit-stream/tree/main/examples) of
the repository: it explains what the script shows, with its output, and
ends with the script itself, which you can download from the link at the top
of the page. The scripts run on simulated data from `logit_stream.simulate`,
so there is nothing else to download and nothing to configure. Each script
stands alone, so a downloaded one runs anywhere logit-stream is installed:

```bash
pip install logit-stream
python quickstart.py
```

or, from a clone of the repository, `python examples/quickstart.py`. Most
take a few seconds; the random-intercepts and mixed-logit examples take
about a minute, since they fit several models each. The scripts that write
files put them in an `output/` directory next to the script, which you can
delete afterwards. The outputs shown on these pages are from those runs,
abridged; your numbers may differ in the last digits.

| | |
|---|---|
| [Quickstart](quickstart.md) | every estimator once, and the reporting and prediction methods they share. Start here. |
| [Binary outcomes](binary.md) | the logit and probit with fixed effects, and Firth's penalized likelihood for separation |
| [Multinomial logit](multinomial.md) | parameterization, standard errors (robust, clustered, multi-way, jackknife), weights, ridge |
| [Ordered logit and probit](ordered.md) | cutpoints, levels, residuals and predictions |
| [Conditional logit](conditional.md) | long-format choice data, alternative-specific constants, policy scenarios |
| [Nested logit](nested.md) | nests of alternatives, the dissimilarity parameters, substitution, deeper trees |
| [Mixed logit](mixed.md) | random coefficients, panels of repeated choices, simulation draws |
| [Fixed effects](fixed_effects.md) | absorbing a fixed effect in the multinomial and ordered logit, and several |
| [Random intercepts](random_intercepts.md) | group intercepts with an estimated covariance, nested groupings, shrinkage |
| [Incidental parameters](incidental_parameters.md) | the fixed-effects logit's bias in short panels, the bias correction, and Chamberlain's conditional logit |
| [Imputation at scale](imputation.md) | predicting a characteristic for a large file, and combining the predictions with other information |

## The simulated data

`logit_stream.simulate` ships with the package, so you can try the
estimators, or check your own code, on data whose answer you know (see
[the reference](../reference/utilities.md)):

- `simulate_people` is a population shaped like the demographic imputation
  problem: people in PUMAs nested in states, with age, sex, income and
  household characteristics, and three outcomes drawn from logit models of
  them -- Hispanic origin (binary), race and ethnicity (six categories) and
  educational attainment (five ordered levels) -- each with PUMA effects.
  `truth_people` returns the coefficients it draws from.
- `simulate_choices` is long-format mode-choice data (train, metro, car),
  in the shape of the Swissmetro survey, drawn from the conditional logit;
  with `nests=` from the nested logit, and with `tasks=` and `sd_time=`
  from the mixed logit with a panel of repeated choices.
- `simulate_panel` is a panel with a binary outcome and unit effects
  correlated with a covariate, for the fixed-effects logit.

```{toctree}
:hidden:

quickstart
binary
multinomial
ordered
conditional
nested
mixed
fixed_effects
random_intercepts
incidental_parameters
imputation
```
