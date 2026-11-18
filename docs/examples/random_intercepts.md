# Random intercepts

{download}`Download the script: random_intercepts.py <../../examples/random_intercepts.py>`

`random=` fits the groups' intercepts as random effects instead of fixed
ones (*experimental*): each group's intercepts are drawn from a normal
distribution with mean zero and a covariance $\Sigma$ that is estimated, one
intercept per non-base outcome level in the multinomial logit, one shift in
the ordered logit.

```python
from logit_stream import mlogit_stream, ologit_stream
from logit_stream.simulate import RACES, simulate_people

# 200 PUMAs in 20 states, PUMA effects with sd 0.5 in every race equation
people = simulate_people(100_000, n_states=20, pumas_per_state=10, seed=1)
X = "age10 + female + log_income + owner + urban"
re = mlogit_stream(f"race ~ {X}", people, levels=RACES, random="puma",
                   vcov={"CRV1": "puma"})
re.summary()
```

```text
### race ~ age10 + female + log_income + owner + urban
Multinomial logit (logit_stream, Newton)   vcov: CRV1:puma
obs: 100,000   outcome race: 6 levels, base 'White'
log-likelihood: -101232.01   null: -107880.18   pseudo R2: 0.061625   AIC: 202554.02   BIC: 202982.1
Newton steps: 28 (converged, decrement 2.75e-09)
random intercepts; Laplace approximation
  by puma (200 groups, unstructured covariance), sd (se): Black: 0.4717 (0.0266)   Hispanic: 0.4937 (0.0293)   Asian: 0.5179 (0.0309)   AIAN: 0.4943 (0.0500)   Multiracial: 0.5295 (0.0493)
    correlations: Black~Hispanic: +0.017 (0.0805)   Black~Asian: -0.235 (0.0775)   ...
```

The sds are close to the simulation's 0.5, and the correlations to its
zero. The same, as a table with 95% intervals, and $\Sigma$ itself:

```python
re.varcorr()
re.random_effects["puma"]["Sigma"]       # and "Sigma_se", its elements' standard errors
re.ranef("puma").collect()               # each PUMA's predicted effects and their sds
```

```text
┌───────┬────────────────────────────┬───────────┬───────────┬───────────┬──────────┐
│ group ┆ term                       ┆ estimate  ┆ std_error ┆ ci_lower  ┆ ci_upper │
╞═══════╪════════════════════════════╪═══════════╪═══════════╪═══════════╪══════════╡
│ puma  ┆ sd(Black)                  ┆ 0.4717    ┆ 0.026588  ┆ 0.422364  ┆ 0.526798 │
│ puma  ┆ sd(Hispanic)               ┆ 0.493689  ┆ 0.029261  ┆ 0.439544  ┆ 0.554504 │
│ puma  ┆ sd(Asian)                  ┆ 0.517923  ┆ 0.030877  ┆ 0.460807  ┆ 0.582119 │
│ puma  ┆ sd(AIAN)                   ┆ 0.494259  ┆ 0.050025  ┆ 0.405325  ┆ 0.602707 │
│ puma  ┆ sd(Multiracial)            ┆ 0.529464  ┆ 0.049342  ┆ 0.441073  ┆ 0.635568 │
│ …     ┆ …                          ┆ …         ┆ …         ┆ …         ┆ …        │
│ puma  ┆ corr(AIAN,Multiracial)     ┆ -0.122162 ┆ 0.147453  ┆ -0.393687 ┆ 0.16897  │
└───────┴────────────────────────────┴───────────┴───────────┴───────────┴──────────┘
```

## Compared with the fixed effect

- **The covariates' coefficients** are those of the model in which group
  effects are independent of the covariates -- the random-effects
  assumption, which the fixed effect does not need. Here the simulation
  satisfies it, and the two are close:

  ```text
  ┌──────────────────┬───────────┬───────────┐
  │ coefficient      ┆ fixed     ┆ random    │
  ╞══════════════════╪═══════════╪═══════════╡
  │ Black:age10      ┆ -0.107702 ┆ -0.107548 │
  │ Black:female     ┆ 0.060783  ┆ 0.061247  │
  │ Black:log_income ┆ -0.354355 ┆ -0.351943 │
  │ Black:owner      ┆ -0.456235 ┆ -0.455614 │
  │ Black:urban      ┆ 0.939993  ┆ 0.931802  │
  └──────────────────┴───────────┴───────────┘
  ```

  When the effects are correlated with the covariates, they are not (see
  [incidental parameters](incidental_parameters.md)).
- **Small groups are shrunk toward the mean** instead of being estimated on
  their own, and no group or level is dropped for separation. The
  smallest PUMAs' predicted Black effects against their fixed effects
  (centred), and the largest PUMAs':

  ```text
  ┌──────┬───────┬───────────┬───────────┐
  │ puma ┆ n_obs ┆ random    ┆ fixed     │
  ╞══════╪═══════╪═══════════╪═══════════╡
  │ 114  ┆ 188   ┆ 0.250216  ┆ 0.370947  │
  │ 53   ┆ 209   ┆ -0.575928 ┆ -0.673262 │
  │ 132  ┆ 211   ┆ 0.712613  ┆ 0.752513  │
  │ …    ┆       ┆           ┆           │
  │ 188  ┆ 2835  ┆ -0.453223 ┆ -0.45614  │
  │ 124  ┆ 3536  ┆ -0.472092 ┆ -0.473511 │
  │ 161  ┆ 5376  ┆ -0.204338 ┆ -0.204552 │
  └──────┴───────┴───────────┴───────────┘
  ```
- **Groups the fit did not see** can be predicted, with their mean effect,
  zero; `predict()` adds each seen group's predicted effect. Here the first
  three records are given a PUMA that does not exist:

  ```text
  ┌───────────┬──────┬────────────┬────────────┐
  │ person_id ┆ puma ┆ prob_White ┆ prob_Black │
  ╞═══════════╪══════╪════════════╪════════════╡
  │ 0         ┆ 9999 ┆ 0.611307   ┆ 0.139666   │
  │ 1         ┆ 9999 ┆ 0.55707    ┆ 0.206087   │
  │ 2         ┆ 9999 ┆ 0.83131    ┆ 0.05592    │
  │ 3         ┆ 21   ┆ 0.779629   ┆ 0.075558   │
  └───────────┴──────┴────────────┴────────────┘
  ```
- **The fixed intercepts stay in the model.**

## Several groupings

A list gives several groupings, each with its own $\Sigma$. They can nest,
to any depth -- PUMAs within states:

```python
ns = mlogit_stream(f"race ~ {X}", people, levels=RACES, random=["state", "puma"],
                   vcov={"CRV1": "state"})
ns.ranef("state").collect()        # each state's effects
ns.ranef("puma").collect()         # each PUMA's, around its state's
```

```text
UserWarning: race ~ age10 + female + log_income + owner + urban: the random intercepts by
['state'] have an estimated variance of zero in some direction (held at the floor, an sd of
0.001; a boundary or 'singular' fit): those groups differ no more than chance allows, beyond
the other terms; consider dropping the grouping. ...
┌───────┬─────────────────┬──────────┬───────────┬────────────┬──────────────┐
│ group ┆ term            ┆ estimate ┆ std_error ┆ ci_lower   ┆ ci_upper     │
╞═══════╪═════════════════╪══════════╪═══════════╪════════════╪══════════════╡
│ puma  ┆ sd(Black)       ┆ 0.469311 ┆ 0.027704  ┆ 0.418036   ┆ 0.526875     │
│ ...
│ state ┆ sd(Black)       ┆ 0.048062 ┆ 0.097851  ┆ 0.000889   ┆ 2.59882      │
│ state ┆ sd(Hispanic)    ┆ 0.018442 ┆ 0.400947  ┆ 5.7463e-21 ┆ 5.9185e16    │
│ ...
```

The simulation has no state effects beyond the PUMAs', and the fit finds
none: the states' $\Sigma$ is singular, and the warning says so.

Two groupings can also cross, as PUMAs and occupations do
(`random=["puma", "occupation"]`). Nested groupings are solved by
elimination from the finest up, at the cost of the fixed effect's; crossed
ones need the smaller grouping's effects solved together, densely, so it may
have at most 6,000 (1,200 occupations of a five-level outcome). The order of
the list does not matter; groupings that neither nest nor cross are refused,
as are more than two that cross.

## Within a fixed effect

The formula can absorb a fixed effect that the random groups nest in --
random PUMA intercepts within fixed state effects:

```python
within = mlogit_stream(f"race ~ {X} | state", people, levels=RACES, random="puma",
                       vcov={"CRV1": "state"})
within.fixef().collect()             # each state's effects
within.ranef("puma").collect()       # each PUMA's, around its state's
```

This is the model with an indicator for every state, and gives the same
estimates and standard errors (checked in the test suite), without the
indicator columns. Each PUMA must lie within one state; nested groupings
(states and PUMAs within regions) must all lie within the fixed effect's
groups, and crossed ones cannot have one.

## The ordered logit

One random shift of the latent index per group:

```python
educ = ologit_stream(f"education ~ {X}", people, random="puma")
educ.varcorr()
```

```text
┌───────┬──────────────┬──────────┬───────────┬──────────┬──────────┐
│ group ┆ term         ┆ estimate ┆ std_error ┆ ci_lower ┆ ci_upper │
╞═══════╪══════════════╪══════════╪═══════════╪══════════╪══════════╡
│ puma  ┆ sd(location) ┆ 0.496957 ┆ 0.025773  ┆ 0.448925 ┆ 0.550127 │
└───────┴──────────────┴──────────┴───────────┴──────────┴──────────┘
```

## Small groups: adaptive quadrature

**Laplace's approximation is biased when groups are small.** It replaces
each group's integral over its effect by a normal curve around the mode,
and with a binary or ordinal outcome and only a few rows per group the
integrand is far from normal. The variance is then underestimated, and
by more than its standard error. The bias is a property of the
approximation, not of this package: lme4's and `clmm`'s Laplace fits
share it. On simulated binary data -- 30,000 firms within 50 cells
(absorbed, `y ~ 1 | cell`), firm effects with variance 0.64, one
simulation each:

| rows per firm (mean) | Laplace | adaptive, 9 nodes | its standard error |
|---|---|---|---|
| 3.0 | 0.494 (-23%) | 0.644 | 0.025 |
| 4.3 | 0.569 (-11%) | 0.664 | 0.019 |
| 11 | 0.611 (-5%) | 0.643 | 0.011 |

With about 80 rows per group the two agree to a few hundredths of a
standard error (the 60-group data of the test suite's lme4 comparison).
The coefficients are biased less than the variance, toward zero: with 8
covariates added to the same simulation, Laplace's slopes were about 2%
smaller than the maximum-likelihood ones at 3 rows per firm (up to one
standard error) and 0.1% at 11.

`random_method="adaptive"` integrates each group's effect by adaptive
Gauss-Hermite quadrature: `random_nodes` points (default 9) centred at the
group's conditional mode and scaled by its conditional sd, as lme4's
`nAGQ > 1` does. It is the maximum-likelihood fit to the quadrature's
accuracy (the tests check it against brute-force integration). It is
available for **one random grouping with a scalar intercept**: the binary
logit (`logit_stream`) and the ordered logit, optionally within an absorbed
fixed effect that the groups nest in:

```python
for method in ["laplace", "adaptive"]:
    sm = logit_stream("y ~ 1 | cell", firms, random="firm", random_method=method)
```

```text
laplace : variance 0.461 (sd 0.679, se 0.013); 11.4 s
adaptive: variance 0.615 (sd 0.784, se 0.016); 5.0 s
```

(The simulation's variance is 0.64.) It starts from the joint-mode fit and
takes full Newton steps in the coefficients, the fixed effect and the log
sd, with each group's nodes held during a step and moved between steps.
Each step is one read of the data, at about nine times a Laplace read's
work; in the simulations above it was nonetheless faster than the Laplace
fit, whose updates of $\Sigma$ slow down with small groups (with the 8
covariates: 5.1 s against 13.3 s at 3 rows per firm, 8.8 s against 14.5 s
at 11). To give a
group's rows together, the first pass writes them grouped by the random
grouping, as it does for an absorbed fixed effect (a group must fit in a
block; `block_mb` raises the size). Restrictions, for now: the iid variance
only (no clustered one), the CPU only, and at least 5 nodes (with fewer,
holding the nodes during a step biases the fit; one node is not Laplace's
fit here, which `random_method="laplace"` gives). Nine nodes are within
1e-5 of the 25-node log-likelihood with groups of about 3 rows; raise
`random_nodes` if the estimates move when you do. For a binary outcome use
`logit_stream`, not a two-level `mlogit_stream`.

### The multinomial logit

With a multinomial outcome Laplace's bias is larger, and not only
downward. On simulated data (J = 5, 20,000 firms within 50 absorbed cells,
firm intercepts with sds 0.8, 0.8, 0.6 and 0.6, correlation 0.3, level E
about 4% of the rows), the bias in each level's variance over 5
simulations, with "small" firms of 0 to 10 known workers (10% with none,
30% with one, mean 3):

| firms | method | B | C | D | E (rare) | mean correlation |
|---|---|---|---|---|---|---|
| small (mean 3 rows) | Laplace | -17% | -16% | -23% | +109% | 0.17 |
| | adaptive | +1% | 0% | -12% | -14% | 0.32 |
| 0 to 10 rows (mean 5) | Laplace | -13% | -14% | -12% | +43% | 0.22 |
| | adaptive | -1% | -2% | -6% | 0% | 0.31 |
| 5 to 10 rows (mean 7.5) | Laplace | -10% | -10% | -8% | +15% | 0.24 |
| | adaptive | 0% | 0% | -4% | -9% | 0.31 |

The slopes were within 3% under both. The rare level's sd varies by
about 0.03 across simulations, so its adaptive figures are within noise.
With 7 points per dimension the estimates agreed with 5 to 0.001; with 3
they moved back toward Laplace's (the rare level's sd 0.66 to 0.76),
which is why at least 5 are required.

```python
fit = mlogit_stream("y ~ x | cell", df, random="firm", random_method="adaptive")
```

The rule is the product of `random_nodes` points per dimension (default
5) over the J - 1 intercepts: 625 points per group for J = 5, centred at
the group's conditional mode and scaled by its conditional covariance.
Groups of 30 rows or more, whose integrands are close to normal, take 3
points per dimension (81). The fit took about 1.6 times the Laplace fit's
time in the simulation above (84 s against 51 s for 60,000 rows). The
tests check it against brute-force integration for J = 3, with and without
a fixed effect: the estimates to 3e-3, the coefficients' and $\Sigma$'s
standard errors to 3%. `draw_parameters` and `draw_ranef` work with it
(see [Multiple imputation](multiple_imputation.md)). See
`logit_stream.random_effects.adaptive_mnl`.

**Conditional sds.** `ranef()` gives each group's effect (its conditional
mode) and, in `sd_<grouping>` (`sd_<grouping>:<level>` for the multinomial
logit, 0 for the base level), its conditional standard deviation: the
square root of the inverse curvature of the group's log posterior at the
mode, given the coefficients, the fixed effect and $\Sigma$ (lme4's
`condVar`). With few rows a group's sd is close to the prior's,
$\sqrt{\Sigma}$:

```text
┌──────┬───────────┬──────────┬───────┐
│ firm ┆ re_firm   ┆ sd_firm  ┆ n_obs │
╞══════╪═══════════╪══════════╪═══════╡
│ 0    ┆ -0.32618  ┆ 0.704382 ┆ 2     │
│ 1    ┆ -0.059486 ┆ 0.68662  ┆ 2     │
│ 4    ┆ 0.038911  ┆ 0.686187 ┆ 2     │
└──────┴───────────┴──────────┴───────┘
```

A normal with that mean and sd is the group effect's approximate posterior,
e.g. for drawing a group's effect in multiple imputation. With very few
rows, or rows all in one outcome level, the posterior is skewed and the
normal is only approximate. The mode and sd do not include the
coefficients' or $\Sigma$'s uncertainty.

## How it is fitted

The likelihood integrates the group effects out. It is approximated by
Laplace's method (`random_method="laplace"`, the default), the approximation
lme4's `glmer` uses with `nAGQ = 1` and ordinal's `clmm` by default. With one
grouping, the coefficients and standard deviations agree with both to about
1e-5, and the log-likelihoods to 1e-4 (checked in the test suite against
their values for the same data); with two, nested or crossed, the
coefficients are within 0.0005 standard errors of theirs and the standard
deviations within 1e-5. On simulated data with about 100 rows per group, the
coefficients were within 0.001 standard errors of the exact
maximum-likelihood estimates (by quadrature). With small groups it is
biased; see [adaptive quadrature](#small-groups-adaptive-quadrature).

`random_method="joint-mode"` instead holds the coefficients and the effects at
their joint mode (lme4's `nAGQ = 0`): faster, and close to the Laplace fit
when groups have many rows each, but with about 100 rows per group its
coefficients can be a few tenths of a standard error away. Here, with 500
rows per PUMA:

```text
joint mode vs Laplace: largest difference 0.183 standard errors; 2.0 s vs 6.9 s
```

Each step is one read of the data, as with the fixed effect: for a given
$\Sigma$, the coefficients and effects solve the fixed effect's problem with
$\Sigma^{-1}$ added to each group's block; $\Sigma$ is updated between reads,
on the data's quadratic approximation in memory; and the Laplace
approximation's extra terms (from the third derivatives of the likelihood)
are added to the gradient in the same read.

**Tiny groups.** The updates of $\Sigma$ converge at the rate of the
fraction of missing information, which approaches 1 when groups have only a
few rows each. When they slow, the rounds switch to quasi-Newton steps in
$\Sigma$, with the coefficients and effects re-solved for each (the
coupling between them is what slows the plain updates). On the ACS
multinomial at 10,000 rows in 2,382 PUMAs (about 4 rows each) the fit takes
52 s rather than 130 s; with larger groups nothing changes.

**A variance of zero.** When groups differ no more than chance allows, the
estimated variance is zero in some direction -- a boundary or "singular"
fit, as lme4 calls it, like the states' above. Its sd is then held at a
floor of 0.001, the fit stops when the approximation no longer rises, and a
warning suggests dropping the grouping. Such fits converge more slowly,
since the updates of $\Sigma$ approach the boundary slowly.

## Inference

The variance of the coefficients is `"iid"` (the inverse of the information
for the coefficients net of the effects) or clustered by a variable that
contains whole groups of the coarsest random grouping; `"hetero"` is not
available, since rows of a group are not independent, and crossed groupings
allow no clustering. Either way, the coefficients' standard errors include
the uncertainty in $\Sigma$, as lme4's and `clmm`'s do; they agree with
theirs to within about 0.4% on the reference data.

$\Sigma$'s standard errors come from the curvature of the Laplace
approximation in $\Sigma$ at the estimates, and the standard deviations' and
correlations' follow by the delta method; they agree with `clmm`'s. The
intervals in `varcorr()` and the summary are Wald intervals on the scales
that respect the bounds -- the log of each sd and Fisher's $z$ of each
correlation -- so they stay in range. They can still be poor when a variance
is near zero, where profile-likelihood intervals would be better (not
computed); at the floor, an sd's interval reaches to infinity, as the
states' do above. `random_cov="diagonal"` estimates uncorrelated effects.

:::{dropdown} The full script: `examples/random_intercepts.py`
```{literalinclude} ../../examples/random_intercepts.py
:language: python
```
:::
