# Choices with known totals

{download}`Download the script: constrained_assignment.py <../../examples/constrained_assignment.py>`

The [conditional logit](../examples/conditional.md) models a choice among
alternatives: each worker choosing among their firm's establishments, say,
by commuting distance and each establishment's attractiveness. This page is
about imputing such a choice where it is not observed but each
alternative's total is: how many workers each establishment employs. There
are two steps:

1. **Calibration**: each alternative's constant is solved so that its
   *expected* number of choosers equals its total. This is
   `predict(calibrate=)`, and it gives each chooser's probabilities.
2. **Constrained assignment** (experimental): all of a group's choices are
   drawn at once, so that the counts equal the totals (exact constraints)
   or stay close to them (soft constraints, for totals measured with
   error). This is `sample_assignments`.

## Calibrating the constants

Fit the model with the alternatives' constants absorbed (see
[absorbed alternative constants](../examples/conditional.md#absorbed-alternative-constants)),
where the choices are known. Then, for alternatives no chooser of the fit
saw but whose totals are known, solve their constants from the totals:

```python
fit = clogit_stream("chosen ~ dist | estab", train, case="worker")
p = fit.predict(newdata, calibrate={"estab": totals})     # prob, lazily
```

Each establishment's constant is the one at which its predicted choices
over `newdata`'s sets equal its total, given the coefficients. Each draw
from `draw_parameters` calibrates under its own coefficients. Within each
firm the totals are first scaled to its number of choice sets, with a
warning if they differ by more than 1%.

:::{dropdown} How the calibration streams
The solve needs all of a group's rows at once: a firm, or more generally
the levels that `newdata`'s choice sets link together. So `newdata` is
streamed through disk a batch of whole groups at a time, about 2 million
rows per batch, or the largest group if that is bigger. The groups are
found by label propagation over the (set, establishment) pairs on disk.
Each batch's constants come from contraction rounds, delta += log(total /
predicted), which converge in a few rounds when every worker in a firm
faces all of its establishments. When they slow down, as on long chains of
linked levels, Newton's method finishes the solve. The predictions come
back as a lazy scan of the batches' files, in `newdata`'s order, so they
can be sunk to disk without ever being in memory. The files are removed
when Python exits. With 4.1 million rows (a 99 MB frame) the calibration
took 4 seconds. Peak memory rose by 0.5 GB with the default batches and by
0.36 GB with batches of 500,000 rows, including the collected result.
:::

**When the coefficients vary by region.** If the distance coefficient
differs between regions, and the training data cover only some regions,
[random slopes by region](../examples/conditional.md#random-slopes-by-region)
estimate how much it varies. Each multiple imputation then draws a slope
for every region it imputes, including regions with no training data,
from N(0, $\Sigma$). The imputations then carry the uncertainty about how
commuting differs in places the training data do not cover, instead of
assuming the training regions' coefficient everywhere. That assumption
remains: the variation among the training regions stands for the variation
among all regions.

**Why estimate the constants, rather than use a proxy for them.** A
simpler model takes the log of each establishment's share of its firm's
workers as its constant, an offset both in the fit and at imputation. Tucker
et al. (2024) show that any such proxy, not estimated jointly with the
coefficients, biases the distance coefficient toward zero, and so yields
too many long commutes (their Appendix B). They also find that estimating
the constants improves the match of the imputed establishments' sizes. A
simulation replicates those findings on synthetic data: 3,000 firms of 2 to
10 establishments and 5 to 150 workers in the fit, and 1,000 new firms to
impute, with a true distance coefficient of -1.2. The script is
[`benchmarks/known_totals_offset.py`](https://github.com/leetucker/logit-stream/blob/main/benchmarks/known_totals_offset.py).
The last column is the sum over establishments of the gap between the
expected and the true count, as a share of the workers.

| | distance coefficient | mean commute (truth 1.797) | P(the true establishment) | workers placed against the totals |
|---|---|---|---|---|
| log-share offset | -1.012 | 1.805 | 0.445 | 18% misallocated |
| constants + calibration | -1.231 | 1.800 | 0.470 | exact (in expectation) |

The offset attenuates the coefficient by a sixth. The aggregate commutes
barely change, but the establishments' totals do not hold. The absorbed
constants' estimate is 2.5% too large: the maximum-likelihood estimate's
own bias with about 13 workers per establishment, as with other fixed
effects estimated from few observations each.

## Drawing assignments that keep the totals

:::{admonition} Experimental
:class: warning
`logit_stream.assign` is new and has not yet been validated beyond the
tests described here. Its samplers and their interface may change. Each
call warns with `logit_stream.ExperimentalWarning`, a subclass of
`hdfe_stream.ExperimentalWarning`, which you can silence with
`warnings.filterwarnings("ignore", category=logit_stream.ExperimentalWarning)`.
:::

An imputation that draws each worker's establishment independently from
the calibrated probabilities matches the totals only on average: in any one
imputation, an establishment with a total of 40 might get 36 or 45.
`sample_assignments` instead draws all of a firm's workers at once, so that
the counts equal the totals or stay close to them.

| | use when |
|---|---|
| `SwapMCMC()` | the totals are exact, and you want the model's distribution given the counts |
| `SwapMCMC(penalty=GaussianPenalty(sd, rel))` | the totals are measured with error |
| `Balanced()` | the totals are exact, and each worker's probabilities must stay the calibrated ones |
| `ExactSmall()` | checking the others on small firms |

### What is drawn

The model's utilities u (x'β, with any other absorbed constants) give
an assignment a of workers to establishments the weight
∏ᵢ exp(u[i, a(i)]). The samplers draw from one of three targets:

- **exact**: that distribution given that the counts nⱼ equal integer
  targets Nⱼ. The establishments' own constants then cancel, since each
  appears Nⱼ times in every allowed assignment, so no calibration is
  needed. Each worker's chance of an establishment is no longer the
  independent logit probability: workers compete for the places.
- **soft**: the same weights, with the calibrated constants, times a
  penalty on the counts, ∏ⱼ gⱼ(nⱼ). With a Gaussian penalty around the
  total, a weak penalty gives the calibrated model's independent
  draws, and a sharp one the exact draw.
- **balanced**: each worker's probability of each establishment is
  exactly the calibrated model's, and the counts are exactly the targets.
  This is the cube method of balanced sampling from survey statistics
  (Deville and Tillé 2004). It is a different target from the exact one:
  it keeps the calibrated probabilities, rather than the model's
  distribution given the counts.

The targets are each group's totals scaled to its number of workers,
since every assignment's counts sum to that. A *group* is a firm, or more
generally the establishments that the choice sets link together. For the
exact samplers, the scaled totals are rounded at random keeping the
group's sum, each rounding up with probability equal to its fractional
part. An establishment with a total of 0 is never chosen.

### The samplers

| | target | how |
|---|---|---|
| `SwapMCMC()` | exact | Metropolis–Hastings from an assignment that meets the targets: two workers swap establishments, which keeps every count |
| `SwapMCMC(penalty=GaussianPenalty(sd, rel))` | soft | swaps, and moves of one worker, which change two counts by one and are accepted with the penalty's change |
| `ExactSmall()` | exact or soft | exact recursion over the workers, with the counts so far as the state; for small firms, with `fallback=` for the others |
| `Balanced()` | balanced | dependent rounding of the workers × establishments table of calibrated probabilities, along cycles of its fractional cells |

`GaussianPenalty(sd, rel)` is log g(n) = −(n − N)² / (2s²), with
s² = sd² + (rel·N)²: an absolute and a relative discrepancy between the
totals and the counts in the records being imputed.

### The example

The script simulates firms of 2 to 10 establishments and 5 to 150
workers, who choose by commuting distance (true coefficient −1.2) and
each establishment's attractiveness. It fits the model on 2,000 firms
whose workers' establishments are known, and imputes 500 others in which
only each establishment's total is known. Each imputation draws the
parameters (`draw_parameters`), then the workers' establishments:

```python
from logit_stream.assign import Balanced, GaussianPenalty, SwapMCMC

d = fit.draw_parameters(5, seed=1)[m]
assigned = d.sample_assignments(newdata, {"estab": totals}, sampler=SwapMCMC(),
                                seed=m).collect()
# the independent draws, for comparison: one per worker from the calibrated probabilities
independent = d.predict(newdata, type="draw", seed=m, calibrate={"estab": totals})
```

The result is `newdata`'s rows with a boolean `assigned`. It is computed
twice: once with the totals equal to the true counts, and once with
totals measured with error, off by about 5% plus one worker. Here are the means over 5
imputations of: the share of workers placed at their true establishment;
the mean commute (truth 1.843); and the establishments' mean absolute gap
between imputed employment and, first, the totals, then the true
counts:

```text
totals = the true counts
              P(true estab)  mean commute  |n - total|  |n - true n|
calibrated            0.487         1.845         1.81          1.81
exact                 0.491         1.838         0.00          0.00
soft                  0.490         1.839         0.68          0.68
balanced              0.486         1.845         0.00          0.00

totals measured with error
              P(true estab)  mean commute  |n - total|  |n - true n|
calibrated            0.486         1.848         1.88          2.06
exact                 0.488         1.840         0.43          0.89
soft                  0.487         1.843         0.84          1.13
balanced              0.486         1.848         0.43          0.89
```

The independent calibrated draws miss each establishment's total by
about 1.8 workers. The exact and balanced samplers hit the totals
whenever the totals are consistent with the firm's number of workers.
When the totals are measured with error, they cannot all be met, so the
exact samplers meet them as scaled to each firm's workers. The aggregate
statistics barely change; the gain is in the establishments' counts.

With noisy totals, the soft sampler's counts are further from the truth
than the exact sampler's (1.13 against 0.89). That is what proper
multiple imputation requires. Each soft draw is a draw of the true count
given the noisy total, so it scatters around the total. The exact
sampler instead treats the total as the truth, every time, and so
understates the uncertainty in the establishments' counts. Which is right
depends on how much error the totals carry; the penalty is where
that assumption is stated.

**Pooling.** The script ends by pooling each method's mean commute over
the imputations by [Rubin's rules](multiple_imputation.md#analyzing-the-completed-data-rubins-rules).
The imputed file is taken to be the whole population, so the mean commute
has no sampling variance, and its standard error is the imputation's
alone (`se=None`):

```python
pool(commutes, se=None, by=["totals", "method"])
```

```text
┌────────────────────────────┬────────────┬──────────┬──────────┬──────────┬──────────┐
│ totals                     ┆ method     ┆ estimate ┆ se       ┆ ci_low   ┆ ci_high  │
╞════════════════════════════╪════════════╪══════════╪══════════╪══════════╪══════════╡
│ totals = the true counts   ┆ calibrated ┆ 1.845048 ┆ 0.003168 ┆ 1.836252 ┆ 1.853844 │
│ totals = the true counts   ┆ exact      ┆ 1.837814 ┆ 0.003031 ┆ 1.829399 ┆ 1.846229 │
│ totals = the true counts   ┆ soft       ┆ 1.838639 ┆ 0.002862 ┆ 1.830692 ┆ 1.846586 │
│ totals = the true counts   ┆ balanced   ┆ 1.845355 ┆ 0.002983 ┆ 1.837072 ┆ 1.853637 │
│ totals measured with error ┆ calibrated ┆ 1.848178 ┆ 0.004105 ┆ 1.836781 ┆ 1.859575 │
│ totals measured with error ┆ exact      ┆ 1.839514 ┆ 0.004265 ┆ 1.827673 ┆ 1.851355 │
│ totals measured with error ┆ soft       ┆ 1.842687 ┆ 0.002735 ┆ 1.835093 ┆ 1.850281 │
│ totals measured with error ┆ balanced   ┆ 1.848334 ┆ 0.00392  ┆ 1.83745  ┆ 1.859218 │
└────────────────────────────┴────────────┴──────────┴──────────┴──────────┴──────────┘
```

Every interval covers the true mean commute, 1.843. With 5 imputations
the degrees of freedom are only 4, so the intervals are wide relative to
the standard errors; more imputations narrow them. For a count fixed by
exact constraints, such as an establishment's employment, every
imputation gives the same value, so `between` is 0 and so is the
standard error. That is correct only if the totals are exactly right; the
soft sampler's counts vary, and their pooled standard errors carry the
totals' error.

### Customizing

A sampler is a `ConstrainedSampler`. A subclass overrides its hooks:

- `targets(batch, seed)`: each establishment's target, e.g. a project's
  own reconciliation of the totals with the records;
- `utilities(batch, targets)`: each row's utility;
- `sample(batch, view, seed)`: the draw, a batch of whole groups at a
  time (the built-in samplers' numba kernels run here); or
  `sample_group(group, seed)`, a group at a time, for a sampler written in
  Python.

The arrays they receive are documented with `Batch` and `View` in
[the reference](../reference/utilities.md). The soft constraint can be
any function of (counts, totals, the establishments' ids) that returns
log g, vectorized, passed as `penalty=`. For example, a target from your
own reconciliation:

```python
import numpy as np
from logit_stream.assign import SwapMCMC, round_targets

class MyTargets(SwapMCMC):
    def targets(self, batch, seed):
        x = my_reconciled_totals(batch.level_id)        # per establishment
        return round_targets(batch, x, seed)            # integers, keeping each firm's sum

fit.sample_assignments(newdata, {"estab": totals}, sampler=MyTargets(), seed=m)
```

Hooks run once per batch, not once per proposal. The inner loops are
numba kernels, since a Python call for each swap would be thousands of
times slower.

### Checks and costs

The tests enumerate every assignment of a small firm: six workers, two of
whom cannot reach the third establishment. Copied as 10,000 independent
firms, it gives each sampler's frequencies of each worker at each
establishment, which match:

- for `SwapMCMC` and `ExactSmall`, the exact conditional distribution;
- for their soft versions, the penalized distribution;
- for `Balanced`, the calibrated probabilities.

All match within a few standard errors. The tests also check that a draw
depends only on its seed, not on how `newdata` is split into batches.

`newdata` is streamed through disk a batch of whole groups at a time, as
in the calibration, so memory is bounded by the batch (about 2 million
rows) or the largest group. On 4.1 million rows (324,000 workers in 5,000
firms), the default exact `SwapMCMC` (200 sweeps) took 2.5 seconds; the
soft one 5.8 seconds; `Balanced` 7 seconds; and `ExactSmall` 99 seconds
for the 3,066 firms small enough for it (it is written in Python, and is
mainly a check on the others).

`SwapMCMC`'s `diagnostics` report the proposals and acceptances of each
kind, and how many workers ended at a different establishment from where
the chain started. Each draw starts from the same assignment, so the
draws are independent only if the chain has mixed. As a check, compare
the results with twice the `sweeps`.

## Where these methods have been used

Matching a choice model's predictions to known totals at each alternative
is old in transportation and regional science. Doubly constrained trip
distribution models scale each destination's attraction until the
predicted trips match its total (Wilson 1967). The workplace and school
location models of activity-based travel models adjust each zone's
constant, its "shadow price", until the modeled workers or students match
the zone's employment or enrollment (ActivitySim's shadow pricing is an
open-source implementation). `predict(calibrate=)` is that calibration.

Drawing integer assignments that respect totals has precedents too. In
spatial microsimulation, weights fitted to small-area totals by iterative
proportional fitting are converted to whole individuals by
"integerization" (Lovelace and Ballas 2013). In surveys, balanced random
imputation uses the cube method so that random imputations keep the
estimated totals (Chauvet, Deville and Haziza 2011), as `Balanced` does
for the establishments' counts. Sampling tables with fixed margins by
swaps is standard for exact conditional inference (Diaconis and Sturmfels
1998), and for null models of presence-absence matrices in ecology.

## References

- Chauvet, G., J.-C. Deville and D. Haziza (2011). On balanced random
  imputation in surveys. *Biometrika* 98(2), 459--471.
- Deville, J.-C. and Tillé, Y. (2004). Efficient balanced sampling: the
  cube method. *Biometrika* 91(4), 893–912.
- Diaconis, P. and Sturmfels, B. (1998). Algebraic algorithms for
  sampling from conditional distributions. *Annals of Statistics* 26(1),
  363–397.
- Gandhi, R., Khuller, S., Parthasarathy, S. and Srinivasan, A. (2006).
  Dependent rounding and its applications to approximation algorithms.
  *Journal of the ACM* 53(3), 324–360.
- Lovelace, R. and D. Ballas (2013). 'Truncate, replicate, sample': a
  method for creating integer weights for spatial microsimulation.
  *Computers, Environment and Urban Systems* 41, 1--11.
- Tucker, L., M. Yi, F. Babalievsky, H. P. Janicki, S. R. Tibbets and
  L. Warren (2024). Revisions to the LEHD establishment imputation
  procedure and applications to administrative job frame. CES Working
  Paper 24-51, Center for Economic Studies, U.S. Census Bureau.
- Wilson, A. G. (1967). A statistical theory of spatial distribution
  models. *Transportation Research* 1(3), 253--269.

:::{dropdown} The full script: `examples/constrained_assignment.py`
```{literalinclude} ../../examples/constrained_assignment.py
:language: python
```
:::
