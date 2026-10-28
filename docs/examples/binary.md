# Binary outcomes

{download}`Download the script: binary.py <../../examples/binary.py>`

$$
P(y_i = 1 \mid x_i) = F(x_i'\beta + \alpha_{g(i)}),
$$

with $F$ the logistic distribution function (`logit_stream`) or the
standard normal's (`probit_stream`), and $\alpha$ the fixed effects, if
any. Both are hdfe-stream's `feglm_stream`, so they absorb any number of
fixed effects, and their options and result (an `hdfe_stream.HDFEResult`)
are hdfe-stream's ([its guide](https://leetucker.github.io/hdfe-stream/examples/glm.html)).

## Logit and probit

```python
from logit_stream import logit_stream, mlogit_stream, probit_stream
from logit_stream.simulate import simulate_people

people = simulate_people(200_000, seed=1)
X = "age10 + female + log_income + owner + urban"
logit = logit_stream(f"hispanic ~ {X} | puma", people, vcov={"CRV1": "puma"})
probit = probit_stream(f"hispanic ~ {X} | puma", people, vcov={"CRV1": "puma"})
```

```text
┌─────────────┬───────────┬───────────┬──────────┐
│ coefficient ┆ logit     ┆ probit    ┆ ratio    │
╞═════════════╪═══════════╪═══════════╪══════════╡
│ age10       ┆ -0.203589 ┆ -0.112242 ┆ 1.813834 │
│ female      ┆ -0.004303 ┆ -0.0013   ┆ 3.311033 │
│ log_income  ┆ -0.307143 ┆ -0.171274 ┆ 1.79328  │
│ owner       ┆ -0.2886   ┆ -0.151423 ┆ 1.905916 │
│ urban       ┆ 0.507099  ┆ 0.28051   ┆ 1.807777 │
└─────────────┴───────────┴───────────┴──────────┘
log-likelihood: logit -83303.8, probit -83326.9
pseudo R2: 0.0742, within the PUMAs: 0.0346
```

The two fit the data almost equally well, and their coefficients differ
mostly by the scale of the error, about 1.6 to 1.8 (the coefficient on
`female` is indistinguishable from zero in both, so its ratio means
nothing). The results carry the log-likelihood, its null, McFadden's pseudo
R-squared and the within pseudo R-squared (what the covariates add beyond
the PUMA effects), AIC and BIC.

## Separation, fixed effects and predictions

A PUMA where nobody, or everybody, has the outcome has an infinite effect.
hdfe-stream drops it before the fit, with a warning, and `sample()` says
which rows it dropped and why:

```python
logit.sample().group_by("dropped_because").len().collect()
```

```text
┌─────────────────┬────────┐
│ dropped_because ┆ len    │
╞═════════════════╪════════╡
│ null            ┆ 199928 │
│ separation      ┆ 72     │
└─────────────────┴────────┘
```

`fixef()` returns the estimated effects, and `predict()` predicts for new
data, lazily: the linear predictor by default, the probability with
`type="response"`. A PUMA without an estimated effect gets a null.

```python
logit.fixef("puma").collect()
logit.predict(new, type="response").collect()
```

## The same model as a two-level multinomial logit

`mlogit_stream` with two levels fits the same model:

```python
two = mlogit_stream(f"hispanic ~ {X} | puma", people, levels=[0, 1], vcov={"CRV1": "puma"})
```

The estimates agree to 6e-11. The differences are conventions: hdfe-stream
judges convergence by the relative change in deviance and gives the iid
variance pyfixest's factor $(N - 1) / (N - K)$; `mlogit_stream` stops on
the Newton decrement and uses Stata's conventions for maximum likelihood
(no factor for the iid variance). `mlogit_stream` absorbs one fixed effect
directly (two or more through a Poisson regression; see [fixed
effects](fixed_effects.md)) and offers what hdfe-stream's binary models do
not: random intercepts, the ridge penalty `l2=`, the cluster jackknife, and
Firth's penalized likelihood.

## Separated covariates and Firth's penalized likelihood

A covariate can separate the outcome too: an indicator whose rows all have
the outcome 0 has a maximum-likelihood estimate of minus infinity. Newton's
method then drifts toward it, and the fit warns about the very large
coefficient and standard error. Here a rare indicator (3% of 400 rows) never
has the outcome:

```python
ml = mlogit_stream("hispanic ~ age10 + log_income + rare", small, levels=[0, 1])
firth = mlogit_stream("hispanic ~ age10 + log_income + rare", small, levels=[0, 1],
                      firth=True)
```

```text
UserWarning: hispanic ~ age10 + log_income + rare: very large coefficients and standard
errors on ['1:rare']: the outcome is probably (quasi-)separated by these covariates, ...
┌──────────────┬───────────┬──────────────┬───────────┬──────────┐
│ coefficient  ┆ ML        ┆ ML se        ┆ Firth     ┆ Firth se │
╞══════════════╪═══════════╪══════════════╪═══════════╪══════════╡
│ 1:Intercept  ┆ 0.884063  ┆ 1.708678     ┆ 0.848602  ┆ 1.689232 │
│ 1:age10      ┆ 0.043522  ┆ 0.091527     ┆ 0.045333  ┆ 0.09052  │
│ 1:log_income ┆ -0.257645 ┆ 0.159553     ┆ -0.253437 ┆ 0.157817 │
│ 1:rare       ┆ -21.36714 ┆ 16367.462665 ┆ -1.692791 ┆ 1.500492 │
└──────────────┴───────────┴──────────────┴───────────┴──────────┘
```

`firth=True` maximizes Firth's (1993) penalized likelihood, the
log-likelihood plus half the log-determinant of the information matrix (a
Jeffreys prior). Its estimates are finite when a covariate separates the
outcome, and their $O(1/n)$ bias is removed, which matters for rare
outcomes and small samples generally. The standard errors are from the
information at the estimates; they agree with R's brglm2 (and, for two
levels, logistf). It is available for the multinomial logit, and so the
binary logit as two levels, without fixed effects, random intercepts or
`l2`. The other cures are to merge the level, drop the covariate, or add a
ridge penalty (`l2=`, which shrinks every coefficient).

The bias that fixed effects estimated from few rows each cause is a
different problem, with its own corrections: see [incidental
parameters](incidental_parameters.md).

:::{dropdown} The full script: `examples/binary.py`
```{literalinclude} ../../examples/binary.py
:language: python
```
:::
