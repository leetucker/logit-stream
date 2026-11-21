# Building on the passes (`logit_stream.core`)

For packages that run their own iteration on this package's streamed passes:
penalized-stream's penalized logit is one. The names here are a stable API,
changed only in a minor release.

```{eval-rst}
.. automodule:: logit_stream.core

.. currentmodule:: logit_stream.core

.. autofunction:: logit_design

.. autoclass:: LogitDesign
   :members: newton_pass, rows, cleanup

.. autoclass:: NewtonMoments
   :members: profiled, effects_step
```
