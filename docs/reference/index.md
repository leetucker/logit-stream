# Function reference

Everything here is importable from `logit_stream`, except the simulators,
which are in `logit_stream.simulate`.

| | |
|---|---|
| **[Estimation](estimation.md)** | |
| `logit_stream`, `probit_stream` | binary logit and probit, with any number of fixed effects (hdfe-stream's `feglm_stream`) |
| `mlogit_stream` | multinomial logit, with absorbed fixed effects or random intercepts |
| `ologit_stream`, `oprobit_stream` | ordered logit and probit, with an absorbed fixed effect (or, for the logit, random intercepts) |
| `clogit_stream` | conditional (McFadden) logit on long-format choice data |
| `cfelogit_stream` | Chamberlain's conditional fixed-effects logit for panels |
| `nlogit_stream`, `mixlogit_stream` | nested and mixed logit (experimental) |
| **[Results](results.md)** | |
| `MultinomialLogitResult`, `OrderedLogitResult`, `OrderedProbitResult`, `ConditionalLogitResult`, `ConditionalFELogitResult`, `NestedLogitResult`, `MixedLogitResult` | one model: coefficients, standard errors, fit statistics, predictions, residuals, fixed or random effects |
| `LogitResult` | what they share |
| `LogitMulti` | a formula that expands to several models |
| **[Utilities](utilities.md)** | |
| `etable` | regression tables, through pyfixest (hdfe-stream's) |
| `bonferroni`, `holm` | multiple-testing corrections across models (hdfe-stream's) |
| `simulate_people`, `simulate_choices`, `simulate_panel` | simulated data with known coefficients |

```{toctree}
:hidden:

estimation
results
utilities
```
