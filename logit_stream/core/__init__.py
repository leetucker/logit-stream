"""The engine of the multinomial and ordered models: their likelihoods and
fused numba passes (`models`), one absorbed fixed effect (`fe`,
`fe_kernels`), pass 0 into the block store (`store`), the streamed Newton
iteration (`newton`) and the robust variances' kernels (`meat_kernels`).

The stable API for packages built on this one is `logit_design`,
`LogitDesign` and `NewtonMoments` (`logit_stream.core.design`, whose
docstring describes them); the rest is internal.
"""

from .design import LogitDesign, NewtonMoments, logit_design

__all__ = ["logit_design", "LogitDesign", "NewtonMoments"]
