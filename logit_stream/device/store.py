"""The device and the blocks on it: the backend (CuPy, through hdfe-stream's
`core.gpu`), and `DeviceStore`, the store's blocks copied over once and
kept, or copied each pass when they do not fit."""

from __future__ import annotations

import numpy as np


def backend():
    """The device's array namespace (CuPy), looked up when called so that
    tests can replace hdfe-stream's with a numpy stand-in."""
    from hdfe_stream.core import gpu
    return gpu.cuda_backend()



def _free_bytes(xp):
    try:
        return int(xp.cuda.runtime.memGetInfo()[0])
    except AttributeError:          # the numpy stand-in
        return None



class DeviceStore:
    """The store's blocks on the device: copied once and kept (`resident`)
    when they fit in `budget` bytes (default: half the device's free memory),
    else copied each time they are iterated. A design whose 0/1 columns the
    store keeps as bytes (`BlockStore(binary=)`) goes over and stays that
    way, and is put back together, float64, as each block is iterated: on
    the device that is a cheap copy, and the blocks take less of its memory
    (so more samples stay resident) and of the bus."""

    FIELDS = ("X", "X~b", "y", "w", "off", "starts", "fe", "cl")

    def __init__(self, store, be, budget=None, extra=None):
        """`extra(block) -> {name: array}`: further per-row arrays, computed
        on the host from each block as it is copied (the random intercepts'
        unit codes)."""
        self.store, self.be, self.extra = store, be, extra
        xp = be.xp
        self._cols = (None if store.binary is None else
                      (xp.asarray(np.flatnonzero(~store.binary)),
                       xp.asarray(np.flatnonzero(store.binary)), len(store.binary)))
        need = 0
        for blk in store.blocks:
            src = blk.arrays if store.in_memory else blk.files
            for k in self.FIELDS:
                if k in src:
                    a = src[k]
                    size = a.nbytes if store.in_memory else a.stat().st_size
                    if k == "X" and np.dtype(store.dtype) != np.float64:
                        size *= 8 // np.dtype(store.dtype).itemsize   # float64 there
                    need += size
        if budget is None:
            free = _free_bytes(be.xp)
            budget = None if free is None else free // 2
        self.bytes = need
        self.resident = budget is None or need <= budget
        self._blocks = None

    def _copy(self, blk):
        xp = self.be.xp
        out = {}
        for k in self.FIELDS:
            if k in blk:
                a = blk[k]
                out[k] = xp.asarray(a, dtype=xp.float64 if k in ("X", "w", "off") else None)
        if self.extra is not None:
            out.update({k: xp.asarray(v) for k, v in self.extra(blk).items()})
        return out

    def _whole(self, blk):
        if "X~b" not in blk:
            return blk
        xp = self.be.xp
        dcols, bcols, K = self._cols
        Xd, Xb = blk["X"], blk["X~b"]
        X = xp.empty((len(Xb), K))
        X[:, dcols] = Xd
        X[:, bcols] = Xb
        out = {k: v for k, v in blk.items() if k != "X~b"}
        out["X"] = X
        return out

    def __iter__(self):
        fields = None if self.extra else self.FIELDS
        if self.resident:
            if self._blocks is None:
                self._blocks = [self._copy(b) for b in self.store.iterate(fields, raw=True)]
            for blk in self._blocks:
                yield self._whole(blk)
            return
        for blk in self.store.iterate(fields, raw=True):
            yield self._whole(self._copy(blk))
