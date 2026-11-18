# Utilities

```{eval-rst}
.. currentmodule:: logit_stream

.. autofunction:: etable

.. autofunction:: bonferroni

.. autofunction:: holm

.. currentmodule:: logit_stream.simulate

.. autofunction:: simulate_people

.. autofunction:: truth_people

.. autofunction:: simulate_choices

.. autofunction:: simulate_panel

.. currentmodule:: logit_stream
```

## Constrained assignment (experimental)

`logit_stream.assign` draws each choice set's alternative so that the
alternatives' counts match known totals, exactly or approximately: see
[the example](../examples/constrained_assignment.md). Its functions warn
with `logit_stream.ExperimentalWarning`, a subclass of hdfe-stream's.

```{eval-rst}
.. currentmodule:: logit_stream.assign

.. autofunction:: sample_assignments

.. autoclass:: ConstrainedSampler
   :members: targets, utilities, sample, sample_group

.. autoclass:: SwapMCMC

.. autoclass:: ExactSmall

.. autoclass:: Balanced

.. autoclass:: GaussianPenalty

.. autofunction:: round_targets

.. autoclass:: logit_stream.calibrate.Batch

.. autoclass:: View
   :members: group

.. currentmodule:: logit_stream

.. autoclass:: ExperimentalWarning
```
