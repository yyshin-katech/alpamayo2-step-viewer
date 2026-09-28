"""Meta-device construction of Alpamayo2Super that mirrors ``from_pretrained`` (transformers 4.57.1).

``Alpamayo2Super.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")`` does, in order:
config load -> _get_dtype (sets the default dtype to bf16 while the model is built, which is
why init-time buffers such as accel_std end up in bf16) -> construct under
[no_init_weights, init_empty_weights] (parameters on meta, buffers real) -> tie_weights ->
restore default dtype -> move/init the missing keys -> load checkpoint tensors (cast to the
meta parameter's dtype) -> tie_weights -> eval -> (no generation config: not a GenerationMixin)
-> dispatch_model, which for a single device is ``model.to(device)``.

Everything except the tensor loading is reproduced here; the parameters stay on meta and are
streamed in by ``stream.Streamer``.
"""

from __future__ import annotations

import copy

import torch
from transformers.modeling_utils import _get_dtype
from transformers.utils import ContextManagers

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

from stream import CkptIndex


def build_meta_model(snap, name_or_path: str | None = None, dtype=torch.bfloat16):
    cls = Alpamayo2Super
    config, _ = cls.config_class.from_pretrained(str(snap), return_unused_kwargs=True)
    config = copy.deepcopy(config)
    config, dtype, dtype_orig = _get_dtype(cls, dtype, None, config, None, None, True)
    config.name_or_path = name_or_path if name_or_path is not None else str(snap)
    ctx = cls.get_init_context(False, False)
    config = copy.deepcopy(config)
    try:
        with ContextManagers(ctx):
            model = cls(config)
        model.tie_weights()
    finally:
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

    assert model._keep_in_fp32_modules is None or dtype != torch.float16
    assert model._keep_in_fp32_modules_strict is None
    assert not model.can_generate()

    index = CkptIndex(snap)
    ckpt_keys = set(index.locs)
    sd_keys = set(model.state_dict())
    unexpected = sorted(ckpt_keys - sd_keys)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:10]}")
    missing = sorted(sd_keys - ckpt_keys)
    model._move_missing_keys_from_meta_to_cpu(missing, dtype, None)
    model._initialize_missing_keys(missing, False)
    model.tie_weights()
    model.eval()

    buffers_in_ckpt = [k for k, _ in model.named_buffers() if k in ckpt_keys]
    if buffers_in_ckpt:
        raise RuntimeError(f"checkpoint holds buffers (unsupported by streaming): {buffers_in_ckpt[:5]}")
    return model, missing


def move_buffers_to_cuda(model, device="cuda:0") -> None:
    """The buffer half of dispatch_model's ``model.to(device)`` (parameters are streamed)."""
    for mod in model.modules():
        for name, b in list(mod._buffers.items()):
            if b is not None:
                assert b.device.type != "meta", name
                mod._buffers[name] = b.to(device)
    model.hf_device_map = {"": torch.device(device)}
