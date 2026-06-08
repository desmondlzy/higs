# PyTorch 2.6 changed torch.load's default to weights_only=True, which rejects the
# numpy/config objects stored in nerfstudio checkpoints. nerfstudio 1.1.5 calls
# torch.load without weights_only=False, so loading our own (trusted) checkpoints
# would fail with an UnpicklingError. Restore the pre-2.6 behavior process-wide.
import torch as _torch

if not getattr(_torch.load, "_heatgaussian_weights_only_patch", False):
    _orig_torch_load = _torch.load

    def _torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_torch_load(*args, **kwargs)

    _torch_load._heatgaussian_weights_only_patch = True
    _torch.load = _torch_load
