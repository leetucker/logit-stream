# Estimation

```{eval-rst}
.. currentmodule:: logit_stream

.. autofunction:: logit_stream

.. autofunction:: probit_stream

.. autofunction:: mlogit_stream

.. autofunction:: ologit_stream

.. autofunction:: oprobit_stream

.. autofunction:: clogit_stream

.. autofunction:: cfelogit_stream
```

## Experimental

The mixed logit is experimental: normal and lognormal coefficients,
uncorrelated, Halton or pseudo-random draws, no weights, prediction or
clustering beyond the chooser. Its interface may change, and each fit warns
with `logit_stream.ExperimentalWarning`. The nested logit (any depth of
nests, all covariates in the alternatives' utilities) is checked against R's
mlogit.

```{eval-rst}
.. currentmodule:: logit_stream

.. autofunction:: mixlogit_stream

.. autofunction:: nlogit_stream
```
