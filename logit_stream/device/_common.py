"""Elementwise pieces, segment sums and maxima, and scatters, written against
the array namespace (numpy's reduceat and add.at on the CPU stand-in)."""

from __future__ import annotations

import numpy as np


def _expit(xp, x):
    """1 / (1 + exp(-x)), without overflow."""
    e = xp.exp(-xp.abs(x))
    return xp.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))



def _ndtr(xp, x):
    if xp is np:
        from scipy.special import ndtr
        return ndtr(x)
    from cupyx.scipy.special import ndtr
    return ndtr(x)



def _cdf_parts(xp, link, x):
    """F(x), 1 - F(x), the density f and its derivative f', for the ordered
    model's link (x may hold +-1e300 for the open ends)."""
    if link == "logit":
        F, S = _expit(xp, x), _expit(xp, -x)
        f = F * S
        return F, S, f, f * (1 - 2 * F)
    F, S = _ndtr(xp, x), _ndtr(xp, -x)
    xc = xp.clip(x, -40.0, 40.0)
    f = xp.exp(-0.5 * xc * xc) / np.sqrt(2 * np.pi)
    return F, S, f, -xc * f



def _gram(X, v):
    """X' diag(v) X."""
    return (X * v[:, None]).T @ X



def _segsum(xp, V, starts):
    """Sums of V's rows (or entries) over the segments [starts[i],
    starts[i + 1]), none of them empty: numpy's reduceat on the CPU, the
    cumulative sum's differences on the device."""
    if xp is np:
        return np.add.reduceat(V, starts[:-1], axis=0)
    cs = xp.cumsum(V, axis=0)
    zero = xp.zeros((1,) + V.shape[1:], dtype=V.dtype)
    cs = xp.concatenate([zero, cs], axis=0)
    return cs[starts[1:]] - cs[starts[:-1]]



def _segmax(xp, V, starts):
    """Each segment's maxima (of V's rows, or entries), none empty."""
    if xp is np:
        return np.maximum.reduceat(V, starts[:-1], axis=0)
    import cupyx
    n = len(starts) - 1
    seg = xp.repeat(xp.arange(n), xp.diff(starts))
    out = xp.full((n,) + V.shape[1:], -xp.inf)
    cupyx.scatter_max(out, seg, V)
    return out



def _scatter_add(xp, out, idx, vals):
    """out[idx[i]] += vals[i], repeated indices summed."""
    if xp is np:
        np.add.at(out, idx, vals)
    else:
        import cupyx
        cupyx.scatter_add(out, idx, vals)



def _scatter_max(xp, out, idx, vals):
    """out[idx[i]] = max(out[idx[i]], vals[i])."""
    if xp is np:
        np.maximum.at(out, idx, vals)
    else:
        import cupyx
        cupyx.scatter_max(out, idx, vals)



def _host(xp, a):
    return a if xp is np or a is None else xp.asnumpy(a)
