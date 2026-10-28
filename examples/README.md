# Examples

One script for each page of the
[guide](https://leetucker.github.io/logit-stream/examples/index.html), which
explains what the script shows, with its output. Every script runs on
simulated data from `logit_stream.simulate` and stands alone, so there is
nothing to download or configure:

```bash
pip install logit-stream
python examples/quickstart.py
```

Scripts that write files put them in `examples/output/`, which nothing
needs afterwards.

| File | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | Every estimator once, and the reporting and prediction methods they share. Start here. |
| [`binary.py`](binary.py) | The logit and probit with fixed effects, separation, and Firth's penalized likelihood. |
| [`multinomial.py`](multinomial.py) | Parameterization, robust, clustered, multi-way and jackknife standard errors, weights, several models and a table, ridge, collinearity. |
| [`ordered.py`](ordered.py) | The ordered logit and probit: cutpoints, levels, residuals, predictions, a fixed effect. |
| [`conditional.py`](conditional.py) | Long-format choice data, alternative-specific constants, unsorted rows, clustering, a policy scenario, and the multinomial logit as a conditional logit. |
| [`nested.py`](nested.py) | The nested logit: dissimilarity parameters, the test against the conditional logit, substitution, a three-level tree, clustering and weights. |
| [`mixed.py`](mixed.py) | The mixed logit: random coefficients, panels, the number and kind of draws, an sd on the boundary, lognormal coefficients. |
| [`fixed_effects.py`](fixed_effects.py) | Absorbed fixed effects: the same fit as indicators, the effects, separation, and several fixed effects. |
| [`random_intercepts.py`](random_intercepts.py) | Random intercepts: the covariance, shrinkage, unseen groups, nested groupings, within a fixed effect, the ordered logit's. About a minute. |
| [`incidental_parameters.py`](incidental_parameters.py) | The fixed-effects logit's bias in short panels, the analytic bias correction, Chamberlain's conditional logit, correlated random effects, and the ordered logit's blow-up-and-cluster estimator. |
| [`imputation.py`](imputation.py) | Predicting race for a million records from a survey, fixed against random effects, and combining the predictions with another source of information by Bayes' rule. |
