"""logit-stream's experimental features and their warning.

A feature is experimental when it has been checked only against brute force
or its own definitions rather than an external reference, when it runs on a
GPU path that has not been optimized for one, or when it is far less
complete than other tools offer. Each use warns with `ExperimentalWarning`,
attributed to the caller's line, so Python's default filter shows it once
per line of user code:

- `mixlogit_stream` (the mixed logit: no predictions, uncorrelated
  coefficients only);
- ``device="cuda"`` and ``polars_engine="gpu"``;
- random slopes: ``clogit_stream(random={"cz": "dist"})`` and
  ``random={"g": ["1", "x"]}`` for the binary, ordered and multinomial logit;
- the multinomial logit's adaptive quadrature (``random_method="adaptive"``);
- draws of the group effects given their rows (`draw_ranef`,
  ``draw_parameters(ranef=True)``), and parameter draws of an adaptive
  multinomial fit;
- ``predict(effects="average")``;
- constrained assignment (`logit_stream.assign`).
"""

from __future__ import annotations

import os
import sys
import warnings

from hdfe_stream.core.report import ExperimentalWarning as _Experimental

_HERE = os.path.dirname(os.path.abspath(__file__))


class ExperimentalWarning(_Experimental):
    """A logit-stream feature that is experimental (see
    `logit_stream._experimental`): its results may change, and it has been
    validated less than the rest of the package. It subclasses
    ``hdfe_stream.ExperimentalWarning``: filter this class to silence
    logit-stream's experimental warnings, or that one for every package built
    on hdfe-stream's core::

        warnings.filterwarnings("ignore", category=logit_stream.ExperimentalWarning)
    """


def _caller_level():
    """The stack level of the first frame outside this package."""
    f, level = sys._getframe(2), 2
    while f is not None and os.path.abspath(f.f_code.co_filename).startswith(_HERE):
        f, level = f.f_back, level + 1
    return level


def experimental(feature, why=""):
    """Warn that `feature` is experimental (`why`: what is missing or
    unchecked), at the user's line."""
    msg = f"{feature} (experimental)" + (f": {why}" if why else "")
    warnings.warn(msg, ExperimentalWarning, stacklevel=_caller_level())
