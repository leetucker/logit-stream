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
re.ranef("puma").collect()               # each PUMA's predicted effects
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
maximum-likelihood estimates (by quadrature).

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
