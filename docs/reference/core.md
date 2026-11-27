# Building on the passes (`logit_stream.core`)

For packages that run their own iteration on the multinomial and ordered
models' streamed passes. penalized-stream's penalized multinomial and ordered
logit is one. The names here are a stable API, changed only in a minor
release. The binary logit's equivalent is hdfe-stream's
`hdfe_stream.core.newton`.

```{eval-rst}
.. automodule:: logit_stream.core.design

.. currentmodule:: logit_stream.core

.. autofunction:: logit_design

.. autoclass:: LogitDesign
   :members: newton_pass, start, feasible, rows, cleanup

.. autoclass:: NewtonMoments
   :members: profiled, effects_step
```
