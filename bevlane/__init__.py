"""METEOR / BevLane package.

Torch >= 2.6 flipped `torch.load`'s `weights_only` default from False to True.
Every checkpoint this project writes carries a metadata dict alongside the
weights (`ious`, `ade_c`, `args`, `git`), and the IoU values in it are numpy
scalars, so on a newer runtime than the one that wrote the file every load
fails with:

    WeightsUnpickler error: Unsupported global:
    GLOBAL numpy.core.multiarray.scalar was not an allowed global by default

That is not a security question here -- the files are this project's own output,
written by its own training loop -- but it does mean a checkpoint trained on
torch 2.1 cannot be read on torch 2.11 without a change. Rather than pass
`weights_only=False` at each of the ~40 load sites (and miss one), restore the
old default once, for this process.

Verified on both: torch 2.1.1+cu121 (workstation) and torch 2.11.0+cu128 (training server), where
the same checkpoint reproduces ADE 0.659 / ADEc 0.656 to three decimals.
"""
import torch as _torch

if not getattr(_torch.load, "_meteor_compat", False):
    _orig_load = _torch.load

    def _load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    _load._meteor_compat = True
    _torch.load = _load
