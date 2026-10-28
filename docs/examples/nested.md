# Nested logit

{download}`Download the script: nested.py <../../examples/nested.py>`

The conditional logit's substitution pattern is rigid: when an alternative
gets worse, its choosers go to the others in proportion to their shares (the
independence of irrelevant alternatives). The nested logit (experimental)
groups alternatives that are closer substitutes into nests, within which
their unobserved utilities are correlated, so that choosers who leave one
alternative go disproportionately to its nest-mates. `nlogit_stream` takes
the same long-format data as `clogit_stream`, a column naming the
alternatives, and the tree of nests.

```python
import polars as pl
from logit_stream import clogit_stream, nlogit_stream
from logit_stream.simulate import simulate_choices

nests = {"rail": ["train", "metro"], "road": ["car"]}
data = (simulate_choices(50_000, nests=nests, lambdas=0.5, seed=1)
        .with_columns(income_car=pl.col("income") * (pl.col("alt") == "car")))
FML = "chosen ~ cost + time + C(alt) + income_car"
fit = nlogit_stream(FML, data, case="case", alt="alt", nests=nests, vcov="hetero")
fit.summary()
```

```text
### chosen ~ cost + time + C(alt) + income_car
Nested logit (logit_stream, Newton)   vcov: hetero
choice sets: 50,000   alternatives (rows): 134,988
nests (alt): rail: train, metro; road: car
log-likelihood of the conditional logit: -34,936.848
log-likelihood: -34111.627   null: -48517.828   pseudo R2: 0.296926   AIC: 68235.255   BIC: 68288.173
┌─────────────────┬───────────┬────────────┬────────────┬─────────────┐
│ Coefficient     ┆ Estimate  ┆ Std. Error ┆ t value    ┆ Pr(>|t|)    │
╞═════════════════╪═══════════╪════════════╪════════════╪═════════════╡
│ cost            ┆ -1.112621 ┆ 0.016045   ┆ -69.343914 ┆ 0.0         │
│ time            ┆ -1.328957 ┆ 0.017765   ┆ -74.807139 ┆ 0.0         │
│ C(alt)[T.metro] ┆ 0.703156  ┆ 0.01266    ┆ 55.54271   ┆ 0.0         │
│ C(alt)[T.train] ┆ 0.293196  ┆ 0.013829   ┆ 21.201982  ┆ 9.1558e-100 │
│ income_car      ┆ 0.292545  ┆ 0.011669   ┆ 25.070388  ┆ 1.0466e-138 │
│ lambda.rail     ┆ 0.514243  ┆ 0.00864    ┆ 59.515987  ┆ 0.0         │
└─────────────────┴───────────┴────────────┴────────────┴─────────────┘
```

The data are drawn from a nested logit with train and metro in a nest with
$\lambda = 0.5$, and the car alone. The estimates recover it: -1.1 on cost,
-1.3 on time, 0.7 and 0.3 for metro and train against the car, 0.3 on
income for the car, and $\lambda_\text{rail} = 0.514$.

The nested and mixed logit refuse collinear covariates rather than drop
them, as the conditional logit does, so a chooser's characteristic enters
as the columns it should have: `income_car` here, instead of
`income:C(alt)` with one of its levels dropped.

## The model

It uses the utility-consistent normalization (RU2, Stata's `nlogit` default
and R mlogit's `unscaled = FALSE`). With two levels, alternative $r$ of nest
$m$ has probability

$$
\frac{e^{V_r/\lambda_m}}{\sum_{s \in m} e^{V_s/\lambda_m}} \cdot
\frac{e^{\lambda_m I_m}}{\sum_l e^{\lambda_l I_l}}, \qquad
I_m = \log \sum_{s \in m} e^{V_s/\lambda_m},
$$

with one $\lambda$ per nest of two or more children (a nest of one has
none). $1 - \lambda_m^2$ is the correlation of the unobserved utilities
within nest $m$. With every $\lambda = 1$ it is the conditional logit, which
is where the fit starts; estimates outside $(0, 1]$, or above the parent
nest's, where the model is not consistent with utility maximization
everywhere, get a warning.

The likelihood is not concave, so the fit is the mixed logit's: each read
evaluates the log-likelihood and its analytic gradient at several points,
the information is their central differences, and between them the steps
are quasi-Newton. The test suite checks it against R's mlogit: the same
estimates, and the BHHH standard errors mlogit reports to six digits (the
iid variance here is the inverse of the observed information, as Stata's),
and a three-level fit against the likelihood written out recursively.

## Against the conditional logit

The conditional logit is $\lambda = 1$, which the likelihood-ratio test and
the Wald test both reject decisively here:

```python
cl = clogit_stream(FML, data, case="case")
lr = 2 * (fit.loglik - cl.loglik)                  # 1650.4, chi-squared(1)
fit.wald_test(keep="lambda.rail", q=1.0)           # statistic 3160.6
```

Fitted to these data, the conditional logit's coefficients are wrong too,
the cost and time coefficients by about a third:

```text
┌─────────────────┬─────────────┬───────────┐
│ coefficient     ┆ conditional ┆ nested    │
╞═════════════════╪═════════════╪═══════════╡
│ cost            ┆ -1.443622   ┆ -1.112621 │
│ time            ┆ -1.796145   ┆ -1.328957 │
│ C(alt)[T.metro] ┆ 0.680443    ┆ 0.703156  │
│ C(alt)[T.train] ┆ 0.087421    ┆ 0.293196  │
│ income_car      ┆ 0.327711    ┆ 0.292545  │
└─────────────────┴─────────────┴───────────┘
```

## Substitution

Raising metro fares by half, where do the metro's lost riders go? Predicting
the shares before and after with each model:

```python
dearer = data.with_columns(
    cost=pl.when(pl.col("alt") == "metro").then(1.5 * pl.col("cost")).otherwise("cost"))
fit.predict(dearer)
```

```text
share of the metro's loss going to each alternative
┌─────────────┬──────────┬──────────┐
│ model       ┆ train    ┆ car      │
╞═════════════╪══════════╪══════════╡
│ conditional ┆ 0.546189 ┆ 0.453811 │
│ nested      ┆ 0.65264  ┆ 0.34736  │
└─────────────┴──────────┴──────────┘
```

The conditional logit splits them roughly by the train's and car's shares;
the nested logit sends more of them to the train, the metro's nest-mate.
`predict()` and `resid()` work as for the conditional logit; the
probabilities need each whole choice set, so Polars evaluates them in
memory.

## Deeper trees

A nest of nests is a dict within the dict. With four alternatives, public
transport {rail {train, metro}, bus} against the car:

```python
tree = {"public": {"rail": ["train", "metro"], "bus": ["bus"]}, "private": ["car"]}
four = simulate_choices(50_000, alternatives=("train", "metro", "bus", "car"), nests=tree,
                        lambdas={"public": 0.7, "rail": 0.4}, seed=2)
deep = nlogit_stream(FML, four, case="case", alt="alt", nests=tree)
```

```text
┌─────────────────┬───────────┬────────────┐
│ Coefficient     ┆ Estimate  ┆ Std. Error │
╞═════════════════╪═══════════╪════════════╡
│ cost            ┆ -1.098435 ┆ 0.016694   │
│ time            ┆ -1.28752  ┆ 0.019098   │
│ C(alt)[T.car]   ┆ -0.346578 ┆ 0.016252   │
│ C(alt)[T.metro] ┆ 0.37017   ┆ 0.011904   │
│ C(alt)[T.train] ┆ -0.044541 ┆ 0.011175   │
│ income_car      ┆ 0.317571  ┆ 0.012138   │
│ lambda.public   ┆ 0.698161  ┆ 0.012858   │
│ lambda.rail     ┆ 0.386346  ┆ 0.007534   │
└─────────────────┴───────────┴────────────┘
log-likelihoods: two levels -44793.3, three -44247.6: LR 1091.4, p = 2.5e-239
```

Deeper trees repeat the two-level step: a nest's value in its parent is
$\lambda_m I_m$, and an alternative's probability is the product of the
conditional probabilities along its path. The two-level tree with all
public transport in one nest is the three-level one with
$\lambda_\text{rail} = \lambda_\text{public}$, rejected here. The bus is a
nest of one, with no $\lambda$ of its own. Every alternative must be in
exactly one nest, a nest holds alternatives or nests but not both, and the
top level needs at least two nests.

## Covariates of a nest

Stata's `nlogit` takes variables at each level of the tree; here every
covariate enters the alternatives' utilities, and under RU2 that is the same
model. A term $z_m \gamma$ that is the same for every alternative of nest
$m$ leaves each conditional probability within the nest unchanged (it
cancels from $V_r/\lambda_m - I_m$) and adds exactly $z_m \gamma$ to the
nest's value $\lambda_m I_m$ -- which is what a nest-level variable with
coefficient $\gamma$ does. So a nest-level variable is a column constant
within each set's nest, entered as usual: a nest-specific constant is an
indicator of the nest, and a chooser's characteristic with an effect on the
nest is its product with that indicator (from a column naming each row's
nest; not `<alt>_nest`, the name the fit uses for its own). (Under RU1, the
other normalization, the two differ by the factor $\lambda_m$.)

## Inference, weights and offsets

`vcov=` is `"iid"` (the inverse of the observed information), `"hetero"`
(the sandwich with the choice sets' scores) or clustered, `{"CRV1": var}`
(multi-way `"a+b"`), with the clusters constant within choice sets.
Weights must be constant within sets too. With five choice sets per person
and weights by person:

```python
nlogit_stream(FML, panel, case="case", alt="alt", nests=nests,
              vcov={"CRV1": "person"}, weights="w")
```

`offset=` adds a known term to each alternative's utility, as for the
conditional logit. `device="cuda"` runs the passes on an NVIDIA GPU
(experimental).

:::{dropdown} The full script: `examples/nested.py`
```{literalinclude} ../../examples/nested.py
:language: python
```
:::
