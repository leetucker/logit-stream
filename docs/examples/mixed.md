# Mixed logit

{download}`Download the script: mixed.py <../../examples/mixed.py>`

The mixed logit (experimental) is the conditional logit with coefficients
that vary across choosers: each chooser $n$ has their own $\beta_n$, drawn
from a distribution whose parameters are estimated,

$$
P_n = \int \prod_{t} \frac{\exp(x_{n t a_{nt}}'\beta)}{\sum_{b \in C_{nt}}
\exp(x_{n t b}'\beta)} \, f(\beta \mid \mu, \sigma) \, d\beta ,
$$

the probability of all of chooser $n$'s choices $a_{nt}$ in their choice
sets $t$, with one draw of $\beta$ for all of them. Besides tastes that
vary, this relaxes the independence of irrelevant alternatives, and lets a
person's choices be correlated over time. `mixlogit_stream` takes the
conditional logit's long-format data, the random coefficients and their
distributions, and the column identifying the chooser.

```python
import polars as pl
from logit_stream import clogit_stream, mixlogit_stream
from logit_stream.simulate import simulate_choices

# 4,000 people make five choices each, with a time coefficient of their own
data = (simulate_choices(20_000, tasks=5, sd_time=1.0, seed=3)
        .with_columns(income_car=pl.col("income") * (pl.col("alt") == "car")))
FML = "chosen ~ cost + time + C(alt) + income_car"
fit = mixlogit_stream(FML, data, case="case", panel="person", random={"time": "normal"},
                      draws=500, vcov={"CRV1": "person"})
fit.summary()
```

```text
### chosen ~ cost + time + C(alt) + income_car
Mixed logit (logit_stream, Newton)   vcov: CRV1:person
choosers: 4,000   choice sets: 20,000   alternatives (rows): 54,024
random: time normal   simulation: 500 Halton draws per chooser
log-likelihood of the conditional logit: -16,499.469
log-likelihood: -16244.957   null: -19421.455   pseudo R2: 0.163556   AIC: 32501.914   BIC: 32539.678
┌─────────────────┬───────────┬────────────┬────────────┬─────────────┐
│ Coefficient     ┆ Estimate  ┆ Std. Error ┆ t value    ┆ Pr(>|t|)    │
╞═════════════════╪═══════════╪════════════╪════════════╪═════════════╡
│ cost            ┆ -1.069592 ┆ 0.02471    ┆ -43.285032 ┆ 0.0         │
│ time            ┆ -1.276978 ┆ 0.03105    ┆ -41.126598 ┆ 0.0         │
│ C(alt)[T.metro] ┆ 0.675224  ┆ 0.023196   ┆ 29.109767  ┆ 2.7007e-186 │
│ C(alt)[T.train] ┆ 0.264453  ┆ 0.024763   ┆ 10.67914   ┆ 1.2745e-26  │
│ income_car      ┆ 0.280275  ┆ 0.020141   ┆ 13.915817  ┆ 5.0776e-44  │
│ sd.time         ┆ 1.004795  ┆ 0.035558   ┆ 28.258033  ┆ 1.1340e-175 │
└─────────────────┴───────────┴────────────┴────────────┴─────────────┘
```

The simulation draws each person's time coefficient from a normal
distribution with mean -1.3 and sd 1; `time` is the mean and `sd.time` the
sd. The share of people for whom more travel time is a good thing is
$\Phi(\mu / \sigma)$ = 0.102 (0.097 in the simulation).

## Random coefficients

`random=` maps covariates to `"normal"` ($\mu + \sigma z$) or `"lognormal"`
($\exp(\mu + \sigma z)$), $z$ standard normal; the coefficients are
uncorrelated. A coefficient's mean parameter is reported under its name and
its sd as `sd.<name>`. The sign of an sd is not identified (the
distribution of $\sigma z$ is that of $-\sigma z$), so it is reported as
its absolute value, with the same standard error; the signed estimate is in
`diagnostics["sd_signed"]`.

A lognormal coefficient is positive, so for a cost, whose coefficient should
be negative for everyone, use the negated column:

```python
neg = data.with_columns(neg_cost=-pl.col("cost"))
mixlogit_stream("chosen ~ neg_cost + time + C(alt) + income_car", neg, case="case",
                panel="person", random={"time": "normal", "neg_cost": "lognormal"},
                draws=500)
```

Here `neg_cost` is $\mu$ and `sd.neg_cost` is $\sigma$ of the log of the
coefficient: a median cost coefficient of $-e^{\mu}$ = -1.068 and a mean of
$-e^{\mu + \sigma^2/2}$ = -1.072 (the simulation's is -1.1 for everyone).

## Against the conditional logit

The conditional logit is the mixed logit with every sd zero:

```python
cl = clogit_stream(FML, data, case="case", vcov={"CRV1": "person"})
lr = 2 * (fit.loglik - cl.loglik)          # 509.0
```

```text
┌─────────────────┬─────────────┬───────────┐
│ coefficient     ┆ conditional ┆ mixed     │
╞═════════════════╪═════════════╪═══════════╡
│ cost            ┆ -0.98576    ┆ -1.069592 │
│ time            ┆ -1.024514   ┆ -1.276978 │
│ C(alt)[T.metro] ┆ 0.604385    ┆ 0.675224  │
│ C(alt)[T.train] ┆ 0.233627    ┆ 0.264453  │
│ income_car      ┆ 0.246765    ┆ 0.280275  │
└─────────────────┴─────────────┴───────────┘
```

The conditional logit's coefficients are smaller: the variation in tastes it
leaves in the error term inflates the error's scale, and the coefficients
are relative to that scale.

An sd of zero is on the boundary of the parameter space, so the
likelihood-ratio statistic's distribution under the null is half a
chi-squared(1) and half a point mass at zero. That matters when the test is
close. It also means that **an sd's standard error and t-statistic mean
little when the sd is near zero**: there the likelihood is flat in the sd to
first order, so its estimate converges only at the rate $n^{-1/4}$. Adding
a random cost coefficient, whose sd is zero in the simulation:

```python
both = mixlogit_stream(FML, data, case="case", panel="person",
                       random={"time": "normal", "cost": "normal"}, draws=500)
```

```text
│ sd.cost         ┆ 0.091868  ┆ 0.216394   │
LR for sd.cost = 0: 0.08, p = 0.39
```

In other simulated samples of this size the estimate ranged from 0 to about
0.3, with "standard errors" as variable, some making it look significant.
Use the likelihood-ratio test, against half a chi-squared(1).

## Panels

`panel=` names the column(s) identifying the chooser, whose choice sets
share one draw of the coefficients. Without it, each choice set is its own
chooser. Here the two give similar estimates, but they are different
models: with a panel the five choices of a person are correlated through
their shared taste, and the likelihood uses that.

The variance is the inverse Hessian (`"iid"`), or the sandwich with the
choosers' scores, which is the variance clustered by chooser:
`{"CRV1": panel}` with a panel (with G / (G - 1)), `"hetero"` without one.
Both are computed at fit time.

## Simulation

The integral over $\beta$ is simulated: each chooser's probability is the
average over `draws` draws of the coefficients (500 by default). Halton
draws (the default) are evenly spread quasi-random points, a prime base per
random coefficient, the first `drop=100` of each sequence discarded, and
consecutive points for consecutive choosers, as xlogit and jaxlogit draw
them; `halton=False` gives pseudo-random normal draws (`seed=`). Halton
draws need far fewer points for the same accuracy:

```text
┌───────┬────────┬───────────┬──────────┬───────────────┬─────────┐
│ draws ┆ halton ┆ time      ┆ sd.time  ┆ loglik        ┆ seconds │
╞═══════╪════════╪═══════════╪══════════╪═══════════════╪═════════╡
│ 50    ┆ true   ┆ -1.276351 ┆ 1.008308 ┆ -16245.569043 ┆ 3.89    │
│ 50    ┆ false  ┆ -1.259164 ┆ 0.977691 ┆ -16272.157087 ┆ 1.71    │
│ 200   ┆ true   ┆ -1.276477 ┆ 1.004917 ┆ -16245.308398 ┆ 2.57    │
│ 200   ┆ false  ┆ -1.27447  ┆ 1.003797 ┆ -16243.728855 ┆ 3.2     │
│ 500   ┆ true   ┆ -1.276978 ┆ 1.004795 ┆ -16244.957049 ┆ 4.92    │
│ 500   ┆ false  ┆ -1.278302 ┆ 1.002477 ┆ -16245.316171 ┆ 5.07    │
└───────┴────────┴───────────┴──────────┴───────────────┴─────────┘
```

With 50 Halton draws the estimates are within 0.004 of 500's; with 50
pseudo-random ones the sd is off by 0.03 and the log-likelihood by 27.
(The times, on 8 logical CPUs, vary from run to run at this size.)

The draws are kept in memory when they fit (`draws_cache_mb=512`), and
otherwise regenerated in each read of the data rather than stored. The
benchmark against xlogit and jaxlogit, and against the exact likelihood by
quadrature, is [here](../benchmarks/mixed.md).

## How it is fitted

The simulated likelihood is not concave. Each read of the data evaluates
it and its analytic gradient at one or several points. A Newton step uses
the Hessian from central differences of the gradient (2P + 1 gradients in
one read), where it is negative definite; elsewhere the step is BHHH's,
with the information estimated by the sum of the choosers' score outer
products. After a Newton step the Hessian is updated by BFGS from the
change in the gradient (quasi-Newton steps, one gradient each), and
recomputed to confirm convergence, so the final Hessian, and the iid
variance, are at the estimates. Each step is followed by a line search.
`switch=` takes BHHH steps first, until their decrement falls below it.
`device="cuda"` runs the passes on an NVIDIA GPU
(experimental), with the likelihood vectorized over rows and draws and the
draws generated on the device.

## Not in the prototype

Correlated random coefficients, distributions other than the normal and
lognormal, weights, predictions and residuals, and clustering beyond the
chooser.

:::{dropdown} The full script: `examples/mixed.py`
```{literalinclude} ../../examples/mixed.py
:language: python
```
:::
