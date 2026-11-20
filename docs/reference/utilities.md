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

## The likelihood-ratio test

```{eval-rst}
.. automodule:: logit_stream.lrtest
   :no-members:

.. currentmodule:: logit_stream

.. autofunction:: lr_test
```

## Multiple imputation

`predict(type="draw", seed=m)` draws each row's outcome for imputation (see
each result's `predict`); `pool` and `pool_matrix` combine the analyses of
the completed data sets by Rubin's rules: see
[multiple imputation](../imputation/multiple_imputation.md).

```{eval-rst}
.. automodule:: logit_stream.mi
   :no-members:

.. currentmodule:: logit_stream

.. autofunction:: pool

.. autofunction:: pool_matrix

.. autoclass:: PooledEstimate
   :members: se, tidy, wald_test
```

## Experimental features

Features that are checked only against brute force or their own
definitions (no external reference), that run on a GPU path not yet
optimized for one, or that are far less complete than other tools, warn
with `logit_stream.ExperimentalWarning` when used, once per line of the
calling code. It subclasses `hdfe_stream.ExperimentalWarning`; silence it
with `warnings.filterwarnings("ignore", category=logit_stream.ExperimentalWarning)`.

```{eval-rst}
.. automodule:: logit_stream._experimental
   :no-members:
```

## Constrained assignment (experimental)

`logit_stream.assign` draws each choice set's alternative so that the
alternatives' counts match known totals, exactly or approximately: see
[the example](../imputation/known_totals.md). Its functions warn
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
