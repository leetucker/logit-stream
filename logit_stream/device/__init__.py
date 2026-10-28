"""Newton passes on a GPU (``device="cuda"``, experimental).

Each pass's log-likelihood, gradient and information are formed on the
device, block by block, and only those sums come back; the Newton step (a
P x P solve, with a fixed effect's Schur complement or the random
intercepts' block elimination) runs on the CPU. That covers the
multinomial, ordered (logit and probit), conditional and conditional
fixed-effects logit, the multinomial and ordered with an absorbed fixed
effect (each level's sums by scatter-adds over the rows' level codes) or
random intercepts (each unit's sums and cross blocks, and the exact Laplace
terms), Firth's adjustment, and the mixed and nested logit (their
likelihoods vectorized: over rows and draws, the draws generated on the
device; over choice sets and nests). The robust and clustered variances'
scores pass runs there too. The blocks are copied to the device once and
kept there when the sample fits in half of its free memory (a design's 0/1
columns as bytes, expanded each pass); otherwise each pass copies them over
again, a block at a time.

The array namespace is hdfe-stream's (`hdfe_stream.core.gpu.cuda_backend`:
CuPy). The functions here are written against it, so the tests run them on a
numpy stand-in; on a GPU the sums are formed in a different order, so the
estimates agree with the CPU's to the convergence tolerance, not bitwise.

What stays on the CPU: pass 0 (unless ``polars_engine="gpu"``), the solves,
and the cluster jackknife (CRV3).

Layout
    store.py           the backend and `DeviceStore`
    _common.py         elementwise pieces, segment sums and maxima, scatters
    models.py          the models' sums (`accumulate`), Firth's adjustment
    fe.py              the absorbed fixed effect's pass
    random_effects.py  the random intercepts' pass
    mixed.py           the mixed logit's pass and draws
    nested.py          the nested logit's pass
    meat.py            the robust variances' scores pass
"""

from .fe import fe_pass
from .meat import fe_meat, meat_sums, plain_meat, re_meat, scores, to_host
from .mixed import mxl_draws, mxl_pass
from .models import accumulate, firth_adjustment
from .nested import nl_pass
from .random_effects import re_pass
from .store import DeviceStore, backend


def resolve(device):
    """device= as the drivers take it: None for the CPU, else the GPU's
    backend (failing now, without CuPy or a GPU)."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    return None if device == "cpu" else backend()


__all__ = ["DeviceStore", "accumulate", "backend", "fe_meat", "fe_pass", "firth_adjustment",
           "meat_sums", "mxl_draws", "mxl_pass", "nl_pass", "plain_meat", "re_meat",
           "re_pass", "resolve", "scores", "to_host"]
