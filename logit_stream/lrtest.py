"""The likelihood-ratio test of two nested models.

``lr_test(restricted, full)`` compares two fits of the same outcome on the
same rows, the restricted model a special case of the full one: the
statistic 2 (loglik_full - loglik_restricted), on the difference in their
numbers of estimated parameters (`n_params`, which counts fixed effects'
levels and the random effects' variances and covariances).

When the restriction sets a variance to zero -- a random intercept against
none, a random coefficient of the mixed logit against a fixed one -- the
null value is on the boundary of the parameter space, and the statistic's
null distribution is not chi-squared: for one variance, it is the 50:50
mixture of chi-squared(0) (a point mass at zero) and chi-squared(1) (Self
and Liang 1987), and the plain chi-squared p-value is twice too large.
``boundary`` chooses the reference distribution:

``"auto"`` (default)
    the mixture where the full model adds exactly one variance to the
    restricted one (a random grouping with a scalar intercept, or one
    random coefficient of the mixed logit), the plain chi-squared
    otherwise;
``True``
    Stram and Lee's (1994) 50:50 mixture of chi-squared(df - 1) and
    chi-squared(df): the reference for adding one random effect, with its
    covariances with the effects already in the model;
``False``
    the plain chi-squared(df), as R's anova for lme4's fits (conservative
    when a variance is on the boundary).

Testing several variances at once (a multinomial random intercept, with
its J - 1 variances and their covariances, against none) has a more
complicated mixture; the plain chi-squared is conservative there.

The test needs the likelihood: not a ridge-penalized (`l2`) or Firth fit,
and not analytic (sampling) weights, under which the log-likelihood is a
pseudo-likelihood and its ratio is not chi-squared (use `wald_test` with a
robust variance instead). With random intercepts the log-likelihood is the
fit's approximation of it (Laplace's, or the adaptive quadrature's), so the
two fits should use the same method.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

__all__ = ["lr_test"]


def _variances_added(restricted, full):
    """The number of variance parameters the full model adds, when that
    is all that distinguishes their random parts in a way this can tell:
    random groupings with a scalar intercept, or random coefficients of the
    mixed logit (each with its own sd). None if it cannot tell."""
    rr = getattr(restricted, "random_effects", None) or {}
    rf = getattr(full, "random_effects", None) or {}
    extra = [k for k in rf if k not in rr]
    if extra and all(np.size(np.asarray(rf[k]["Sigma"])) == 1 for k in extra) \
            and set(rr) <= set(rf):
        return len(extra)
    mr = getattr(restricted, "random", None)
    mf = getattr(full, "random", None)
    if isinstance(mf, dict) and mf:
        mr = mr if isinstance(mr, dict) else {}
        if set(mr) <= set(mf):
            return len(set(mf) - set(mr))
    return None


def lr_test(restricted, full, boundary: bool | str = "auto") -> dict:
    """The likelihood-ratio test of `restricted` against `full`, two fitted
    results of nested models of the same outcome on the same rows (see
    `logit_stream.lrtest`).

    Returns ``statistic`` (2 (loglik_full - loglik_restricted)), ``df``
    (the difference in `n_params`), ``p``, ``reference`` (the distribution
    the p-value is from) and ``boundary`` (whether it is a boundary
    mixture). `boundary`: ``"auto"``, ``True`` or ``False``, as there.
    """
    if boundary not in ("auto", True, False):
        raise ValueError(f"boundary must be 'auto', True or False, got {boundary!r}")
    for r in (restricted, full):
        for a in ("loglik", "n_params", "n_obs"):
            if getattr(r, a, None) is None:
                raise TypeError(f"lr_test needs fitted results with {a!r}; got {type(r).__name__}")
        if getattr(r, "l2", 0.0) or getattr(r, "firth", False):
            raise ValueError("lr_test needs the likelihood: a ridge-penalized (l2) or Firth fit's "
                             "objective is penalized")
        if getattr(r, "weights", None) and getattr(r, "weights_type", "") != "fweights":
            raise ValueError("lr_test needs the likelihood: with analytic (sampling) weights it is "
                             "a pseudo-likelihood; use wald_test with a robust variance")
    if getattr(restricted, "depvar", None) != getattr(full, "depvar", None):
        raise ValueError(f"the models have different outcomes ({restricted.depvar!r}, "
                         f"{full.depvar!r})")
    if restricted.n_obs != full.n_obs:
        raise ValueError(f"the models were fitted to different rows ({restricted.n_obs:,} and "
                         f"{full.n_obs:,} observations)")
    df = int(full.n_params) - int(restricted.n_params)
    if df <= 0:
        raise ValueError(f"the full model must have more parameters than the restricted one "
                         f"({full.n_params} and {restricted.n_params}); pass the restricted "
                         "model first")
    ll0, ll1 = float(restricted.loglik), float(full.loglik)
    stat = 2.0 * (ll1 - ll0)
    if stat < -1e-6 * max(1.0, abs(ll1)):
        warnings.warn(f"the full model's log-likelihood is below the restricted one's (by "
                      f"{-stat / 2:.4g}): are they nested, and did both converge?", stacklevel=2)
    stat = max(stat, 0.0)
    if boundary == "auto":
        boundary = df == 1 and _variances_added(restricted, full) == 1
    if boundary:
        lo = stats.chi2.sf(stat, df - 1) if df > 1 else float(stat <= 0)
        p = 0.5 * lo + 0.5 * stats.chi2.sf(stat, df)
        ref = f"0.5 chi2({df - 1}) + 0.5 chi2({df})"
    else:
        p = stats.chi2.sf(stat, df)
        ref = f"chi2({df})"
    return {"statistic": stat, "df": df, "p": float(p), "reference": ref,
            "boundary": bool(boundary), "loglik": (ll0, ll1)}
