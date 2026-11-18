# Constrained assignment

{download}`Download the script: constrained_assignment.py <../../examples/constrained_assignment.py>`

:::{admonition} Experimental
:class: warning
`logit_stream.assign` is new and has not yet been validated beyond the
tests described here. Its samplers and their interface may change. Each
call warns with `logit_stream.ExperimentalWarning`, a subclass of
`hdfe_stream.ExperimentalWarning`, which you can silence with
`warnings.filterwarnings("ignore", category=logit_stream.ExperimentalWarning)`.
:::

The [conditional logit](conditional.md)'s calibration solves each
establishment's constant so that its *expected* employment equals a known
total. Imputations that draw each worker's establishment independently
from those probabilities then match the totals only on average: in any
one imputation, an establishment with 40 reported employees might get 36
or 45. When the totals are known, an imputation can respect them. That is
what `sample_assignments` does. It draws all of a firm's workers at once,
so that the establishments' counts equal their totals (exact constraints)
or stay close to them (soft constraints, for totals measured with error).

## What is drawn

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
  reported total, a weak penalty gives the calibrated model's independent
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

## The samplers

| | target | how |
|---|---|---|
| `SwapMCMC()` | exact | Metropolis–Hastings from an assignment that meets the targets: two workers swap establishments, which keeps every count |
| `SwapMCMC(penalty=GaussianPenalty(sd, rel))` | soft | swaps, and moves of one worker, which change two counts by one and are accepted with the penalty's change |
| `ExactSmall()` | exact or soft | exact recursion over the workers, with the counts so far as the state; for small firms, with `fallback=` for the others |
| `Balanced()` | balanced | dependent rounding of the workers × establishments table of calibrated probabilities, along cycles of its fractional cells |

`GaussianPenalty(sd, rel)` is log g(n) = −(n − N)² / (2s²), with
s² = sd² + (rel·N)²: an absolute and a relative discrepancy between the
reported totals and the wage records.

## The example

The script simulates firms of 2 to 10 establishments and 5 to 150
workers, who choose by commuting distance (true coefficient −1.2) and
each establishment's attractiveness. It fits the model on 2,000 firms
whose workers' establishments are known, and imputes 500 others in which
only each establishment's total is reported. Each imputation draws the
parameters (`draw_parameters`), then the workers' establishments:

```python
from logit_stream.assign import Balanced, GaussianPenalty, SwapMCMC

d = fit.draw_parameters(5, seed=1)[m]
assigned = d.sample_assignments(newdata, {"estab": totals}, sampler=SwapMCMC(),
                                seed=m).collect()
```

The result is `newdata`'s rows with a boolean `assigned`. It is computed
twice: once with the true counts reported, and once with the reported
totals off by about 5% plus one worker. Here are the means over 5
imputations of: the share of workers placed at their true establishment;
the mean commute (truth 1.843); and the establishments' mean absolute gap
between imputed employment and, first, the reported totals, then the
true counts:

```text
totals = the true counts
              P(true estab)  mean commute  |n - total|  |n - true n|
calibrated            0.488         1.846         1.81          1.81
exact                 0.490         1.836         0.00          0.00
soft                  0.489         1.838         0.68          0.68
balanced              0.486         1.846         0.00          0.00

totals reported with error
              P(true estab)  mean commute  |n - total|  |n - true n|
calibrated            0.486         1.849         1.88          2.03
exact                 0.489         1.843         0.48          0.89
soft                  0.488         1.845         0.87          1.13
balanced              0.487         1.849         0.48          0.89
```

The independent calibrated draws miss each establishment's total by
about 1.8 workers. The exact and balanced samplers hit the totals
whenever the totals are consistent with the firm's number of workers.
When the totals are reported with error, they cannot all be met, so the
exact samplers meet them as scaled to each firm's workers. The aggregate
statistics barely change; the gain is in the establishments' counts.

With noisy totals, the soft sampler's counts are further from the truth
than the exact sampler's (1.13 against 0.89). That is what proper
multiple imputation requires. Each soft draw is a draw of the true count
given the noisy report, so it scatters around the report. The exact
sampler instead treats the report as the truth, every time, and so
understates the uncertainty in the establishments' counts. Which is right
depends on how much error the reported totals carry; the penalty is where
that assumption is stated.

## Customizing

A sampler is a `ConstrainedSampler`. A subclass overrides its hooks:

- `targets(batch, seed)`: each establishment's target, e.g. a project's
  own reconciliation of reported totals with the records;
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

## Checks and costs

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

:::{dropdown} The full script: `examples/constrained_assignment.py`
```{literalinclude} ../../examples/constrained_assignment.py
:language: python
```
:::

## References

- Deville, J.-C. and Tillé, Y. (2004). Efficient balanced sampling: the
  cube method. *Biometrika* 91(4), 893–912.
- Diaconis, P. and Sturmfels, B. (1998). Algebraic algorithms for
  sampling from conditional distributions. *Annals of Statistics* 26(1),
  363–397.
- Gandhi, R., Khuller, S., Parthasarathy, S. and Srinivasan, A. (2006).
  Dependent rounding and its applications to approximation algorithms.
  *Journal of the ACM* 53(3), 324–360.
