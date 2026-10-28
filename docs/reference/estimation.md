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

The mixed and nested logit are prototypes. The mixed logit: normal and
lognormal coefficients, uncorrelated, Halton or pseudo-random draws, no
weights, prediction or clustering beyond the chooser. The nested logit: any
depth of nests, all covariates in the alternatives' utilities. Their
interfaces may change.

```{eval-rst}
.. currentmodule:: logit_stream

.. autofunction:: mixlogit_stream

.. autofunction:: nlogit_stream
```
