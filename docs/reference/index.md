# Function reference

Everything here is importable from `logit_stream`, except the simulators,
which are in `logit_stream.simulate`.

| | |
|---|---|
| **[Estimation](estimation.md)** | |
| `logit_stream`, `probit_stream` | binary logit and probit, with any number of fixed effects, fitted by hdfe-stream's `feglm_stream` |
| `mlogit_stream` | multinomial logit, with absorbed fixed effects or random intercepts |
| `ologit_stream`, `oprobit_stream` | ordered logit and probit, with an absorbed fixed effect (or, for the logit, random intercepts) |
| `clogit_stream` | conditional (McFadden) logit on long-format choice data |
| `cfelogit_stream` | Chamberlain's conditional fixed-effects logit for panels |
| `nlogit_stream`, `mixlogit_stream` | nested logit; mixed logit (experimental) |
| **[Results](results.md)** | |
| `BinaryLogitResult`, `MultinomialLogitResult`, `OrderedLogitResult`, `OrderedProbitResult`, `ConditionalLogitResult`, `ConditionalFELogitResult`, `NestedLogitResult`, `MixedLogitResult` | one model: coefficients, standard errors, fit statistics, predictions, residuals, fixed or random effects |
| `LogitResult` | what they share |
| `LogitMulti` | a formula that expands to several models |
| **[Utilities](utilities.md)** | |
| `etable` | regression tables, through pyfixest (hdfe-stream's) |
| `bonferroni`, `holm` | multiple-testing corrections across models (hdfe-stream's) |
| `simulate_people`, `simulate_choices`, `simulate_panel` | simulated data with known coefficients |
| `lr_test` | the likelihood-ratio test of nested models, with the boundary mixture for a variance |
| `pool`, `pool_matrix` | combining multiply-imputed analyses by Rubin's rules, with joint tests |
| **[Building on the passes](core.md)** | |
| `logit_stream.core`: `logit_design`, `LogitDesign`, `NewtonMoments` | the multinomial and ordered models' stored design and Newton pass, for packages that run their own iteration (penalized-stream) |
| `logit_stream.assign` | constrained assignment to known totals: `sample_assignments`, the samplers `SwapMCMC`, `ExactSmall`, `Balanced`, and `GaussianPenalty` (experimental) |

```{toctree}
:hidden:

estimation
results
utilities
core
```
