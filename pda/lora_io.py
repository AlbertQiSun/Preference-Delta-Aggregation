"""Loading, saving and name-mapping utilities for PEFT LoRA adapters."""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch

# A LoRA adapter is represented as {module_name: (B, A)} with
#   B : (d, r)   (lora_B.weight, i.e. out_features x r)
#   A : (r, k)   (lora_A.weight, i.e. r x in_features)
# so that the (unscaled) weight update is dW = B @ A, shape (d, k).
Factors = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def _read_state_dict(path: str) -> Dict[str, torch.Tensor]:
    st = os.path.join(path, "adapter_model.safetensors")
    bn = os.path.join(path, "adapter_model.bin")
    if os.path.exists(st):
        from safetensors.torch import load_file
        return load_file(st)
    if os.path.exists(bn):
        return torch.load(bn, map_location="cpu")
    raise FileNotFoundError(f"no adapter_model.safetensors or adapter_model.bin in {path}")


def load_config(path: str) -> dict:
    with open(os.path.join(path, "adapter_config.json")) as f:
        return json.load(f)


def lora_scaling(cfg: dict) -> float:
    """Scaling factor PEFT applies at inference: dW_effective = scaling * B @ A."""
    r, alpha = cfg["r"], cfg["lora_alpha"]
    if cfg.get("use_rslora", False):
        return alpha / (r ** 0.5)
    return alpha / r


def load_adapter(path: str,
                 dtype: torch.dtype = torch.float64,
                 target_modules: Optional[List[str]] = None) -> Tuple[Factors, dict]:
    """Read one PEFT LoRA adapter directory. Returns (factors, adapter_config)."""
    sd = _read_state_dict(path)
    cfg = load_config(path)
    factors: Factors = {}
    for key, w in sd.items():
        if "lora_A" not in key:
            continue
        name = key.replace(".lora_A.default.weight", "").replace(".lora_A.weight", "")
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in sd:
            print(f"  [warn] no lora_B for {key}, skipping")
            continue
        if target_modules and not any(m in name for m in target_modules):
            continue
        factors[name] = (sd[b_key].to(dtype), w.to(dtype))
    if not factors:
        raise RuntimeError(f"no LoRA (A, B) pairs found in {path}")
    return factors, cfg


def load_adapters(paths: List[str],
                  dtype: torch.dtype = torch.float64,
                  target_modules: Optional[List[str]] = None,
                  verbose: bool = True) -> Tuple[List[Factors], List[dict]]:
    """Load several adapters and restrict them to their common set of modules.

    All adapters must share the same LoRA rank and scaling, because the merge is
    computed on the unscaled factors and the merged adapter reuses the config of
    the first adapter.
    """
    adapters, cfgs = [], []
    for p in paths:
        f, c = load_adapter(p, dtype, target_modules)
        adapters.append(f)
        cfgs.append(c)
        if verbose:
            print(f"  loaded {len(f):4d} adapted matrices  r={c['r']}  "
                  f"alpha={c['lora_alpha']}  <- {p}")

    ranks = {c["r"] for c in cfgs}
    scalings = {round(lora_scaling(c), 10) for c in cfgs}
    if len(ranks) != 1 or len(scalings) != 1:
        raise ValueError(f"adapters must share r and lora_alpha; got r={ranks}, "
                         f"scaling={scalings}")

    common = set(adapters[0])
    for a in adapters[1:]:
        common &= set(a)
    if not common:
        raise RuntimeError("adapters share no module names (check target_modules)")
    if any(len(a) != len(common) for a in adapters) and verbose:
        print(f"  [warn] module sets differ, using the intersection ({len(common)} matrices)")
    names = sorted(common)
    return [{n: a[n] for n in names} for a in adapters], cfgs


def save_adapter(factors: Factors, out_dir: str, reference_adapter: str,
                 save_dtype: Optional[torch.dtype] = None, extra_info: Optional[dict] = None):
    """Write merged factors as a PEFT adapter that loads with PeftModel.from_pretrained.

    adapter_config.json is copied from `reference_adapter`. Since all merged
    adapters share r and lora_alpha, the copied scaling is the correct one.
    """
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    cfg = load_config(reference_adapter)
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if save_dtype is None:
        ref_sd = _read_state_dict(reference_adapter)
        save_dtype = next(iter(ref_sd.values())).dtype

    sd = {}
    for name, (B, A) in factors.items():
        sd[f"{name}.lora_A.weight"] = A.to(save_dtype).contiguous()
        sd[f"{name}.lora_B.weight"] = B.to(save_dtype).contiguous()
    save_file(sd, os.path.join(out_dir, "adapter_model.safetensors"))

    if extra_info is not None:
        with open(os.path.join(out_dir, "merge_info.json"), "w") as f:
            json.dump(extra_info, f, indent=2)


def to_model_param_name(adapter_module_name: str, model_keys) -> Optional[str]:
    """Map a PEFT module name to the base-model parameter name.

    Example: base_model.model.model.layers.0.self_attn.q_proj
             -> model.layers.0.self_attn.q_proj.weight
    """
    base = adapter_module_name
    for prefix in ("base_model.model.", "base_model."):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    for cand in (base + ".weight", base):
        if cand in model_keys:
            return cand
    for k in model_keys:
        if k.endswith(base + ".weight"):
            return k
    return None
