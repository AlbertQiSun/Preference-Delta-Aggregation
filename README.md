# From "Weak" Signals to Strong Models: Preference Delta Aggregation with LoRA Merging

[![arXiv](https://img.shields.io/badge/arXiv-2606.00357-b31b1b.svg)](https://arxiv.org/abs/2606.00357)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://arxiv.org/abs/2606.00357)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**English** | [简体中文](README_zh.md)

Qi Sun\*, Siyue Zhang\*†, Yulin Chen, Yuxiang Xue, Ru Peng, Chen Zhao†

\* Equal contribution &nbsp;&nbsp; † Corresponding authors

Official code for our NeurIPS 2026 paper.

## Overview

High-quality supervision for strong LLMs is scarce. Prior work shows that a preference pair built from two *weaker* models can still teach a stronger model through the quality difference between the two (a *preference delta*). This paper asks whether several such weak signals can be combined.

**Preference Delta Aggregation (PDA)**

1. derives a preference delta from each pair of weak models,
2. trains each delta separately as a LoRA adapter with preference optimization, and
3. merges the adapters in parameter space.

To reduce directional interference during merging we propose **Geometric Alignment Merging (GAM)**, which aligns the adapters' singular subspaces before averaging them.

On knowledge reasoning and agentic search benchmarks, aggregating several deltas beats every single delta, and the gain grows as more deltas are added. With GAM, PDA improves Qwen3-8B by 6.8 points on knowledge reasoning and 7.3 points on agentic search on average. It beats the best single delta by 2.1 and 4.3 points and outperforms other multi-delta baselines.

## Geometric Alignment Merging

For n rank-r updates of the same module, dW_i = B_i A_i:

1. **Decompose**: thin SVD of each update, dW_i = U_i S_i V_iᵀ.
2. **Align**: orthogonal Procrustes on both sides to a reference adapter,
   R_L,i = argmin ‖U_i R − U_ref‖, R_R,i = argmin ‖V_i R − V_ref‖ over R ∈ O(r).
   Then U_i ← U_i R_L,i, S_i ← R_L,iᵀ S_i R_R,i and V_i ← V_i R_R,i (this leaves dW_i unchanged).
3. **Average** the aligned factors separately: Ū, S̄, V̄.
4. **Re-orthonormalize** Ū and V̄ with a thin QR and absorb the triangular factors into S̄.
5. **Reconstruct** dW_GAM = Û Ŝ V̂ᵀ. It has rank ≤ r and is saved exactly as a rank-r LoRA adapter, so the merged model costs the same at inference as a single adapter.

`pda/gam.py` contains two implementations that agree to machine precision (see `tests/test_gam.py`):

| function | what it does | cost per module |
|---|---|---|
| `gam_merge` (default) | Step 1 from the LoRA factors via two thin QRs and an r×r SVD. Never forms the d×k update. Step 4 is skipped because it does not change the product. | O((d + k) r²) |
| `gam_merge_dense` | follows the five steps literally on dense d×k updates (reference implementation) | O(d k min(d, k)) |

`analysis/merge_cost.py` measures the time and memory of both versions and of the baselines.

## Repository structure

```
pda/                          core library
  gam.py                      GAM (fast and reference implementations), averaging baseline
  lora_io.py                  load / save PEFT LoRA adapters
scripts/
  merge_adapters.py           merge preference-delta adapters (GAM, averaging)
  merge_ties_peft.py          TIES-Merging baseline (PEFT add_weighted_adapter)
  build_ablation_models.py    standalone models for the GAM component ablation
analysis/
  geometry_audit.py           subspace overlap, Procrustes statistics, step-by-step ablation ladder
  check_init.py               shared LoRA initialization check, left vs right alignment
  direction_retention.py      how much of each delta's direction survives each merge operator
  delta_magnitude.py          ||dW|| / ||W|| relative to the base weights
  merge_cost.py               wall-clock time and memory of the merge operators
  gam_numerics.py             float32 / float64 accuracy of the dense and factored SVD paths
tests/test_gam.py             numerical tests
third_party/                  git submodules: LLaMA-Factory, OpenCompass, Agentic-Rag
```

## Installation

```bash
git clone --recursive https://github.com/AlbertQiSun/Preference-Delta-Aggregation.git
cd Preference-Delta-Aggregation
pip install -r requirements.txt
```

If you cloned without `--recursive`, run `git submodule update --init --recursive`. The submodules are only needed for training and evaluation. Merging needs only `torch` and `safetensors`.

## Quick start: merge preference-delta adapters

```bash
python scripts/merge_adapters.py \
    --adapters ckpt/delta_1 ckpt/delta_2 ckpt/delta_3 ckpt/delta_4 \
    --method gam \
    --output merged/pda_gam
```

* All adapters must be PEFT LoRA adapters of the same base model with the same `r` and `lora_alpha`.
* The first adapter is the Procrustes reference (change it with `--ref-idx`). In the paper the reference is the single delta with the best validation performance.
* The output is a regular PEFT adapter (`PeftModel.from_pretrained` loads it). Add `--export-full-model --base-model Qwen/Qwen3-8B` to also write a standalone model with the update folded in.
* `--method average` gives the averaging baseline (average of the updates, truncated to rank r). `--method gam_dense` runs the reference implementation.

The TIES baseline:

```bash
python scripts/merge_ties_peft.py \
    --base_model Qwen/Qwen3-8B \
    --loras ckpt/delta_1 ckpt/delta_2 ckpt/delta_3 ckpt/delta_4 \
    --density 0.5 --output merged/ties
```

Python API:

```python
from pda import load_adapters, gam_merge, save_adapter

adapters, cfgs = load_adapters(["ckpt/delta_1", "ckpt/delta_2", "ckpt/delta_3"])
merged = {name: gam_merge([a[name][0] for a in adapters],   # lora_B, (d, r)
                          [a[name][1] for a in adapters])   # lora_A, (r, k)
          for name in adapters[0]}
save_adapter(merged, "merged/pda_gam", reference_adapter="ckpt/delta_1")
```

## Full pipeline

| stage | tool |
|---|---|
| 1. Build preference pairs from each pair of weak models | *coming soon* |
| 2. Train each preference delta as a LoRA adapter with preference optimization | [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) (`third_party/LlamaFactory`) |
| 3. Merge the adapters | `scripts/merge_adapters.py` (this repo) |
| 4a. Agentic search evaluation (multi-hop QA with retrieval over Wikipedia) | [Agentic-Rag](https://github.com/yizhou0409/Agentic-Rag) (`third_party/Agentic-Rag`) |
| 4b. Knowledge reasoning evaluation | [OpenCompass](https://github.com/open-compass/opencompass) (`third_party/opencompass`) |

See the README in each submodule for setup. For Agentic-Rag this includes building the E5 index over the Wikipedia dump.

## Analysis scripts

All analysis scripts read PEFT adapters directly, run on CPU (GPU optional) and never evaluate a model.

```bash
# subspace geometry and the ablation ladder AVG -> AVG-r -> GAM-id -> GAM
python analysis/geometry_audit.py --adapters ckpt/delta_* --out results/geometry
python analysis/check_init.py --adapters ckpt/delta_* --audit-json results/geometry/geometry_audit.json

# direction retention of AVG-r, TIES and GAM as deltas are added
python analysis/direction_retention.py --adapters ckpt/delta_1 ckpt/delta_2 ckpt/delta_3 ckpt/delta_4 --all-layers

# merge cost and numerical accuracy
python analysis/merge_cost.py --real-adapter ckpt/delta_1 --threads 16
python analysis/gam_numerics.py

# size of an update relative to the base weights
python analysis/delta_magnitude.py --base /path/to/Qwen3-8B --adapter merged/pda_gam --apply-scaling

# standalone models for the component ablation (avg, avg_r, gam_id, gam_id_nm, gam)
python scripts/build_ablation_models.py --base /path/to/Qwen3-8B --adapters ckpt/delta_* --out merged/ablation
```

## Implementation notes

* **Procrustes over O(r).** GAM solves the alignment over the orthogonal group, as in the paper, which makes the result independent of the sign convention of the SVD routine. `--procrustes rotation` restricts the solution to rotations (det R = +1). It reproduces our earliest implementation and is kept only for that purpose.
* **Precision.** Merging runs in float64 on CPU by default (`--compute-dtype`). On GPUs, TF32 matmul is disabled in all scripts because its ~1e-3 relative precision is visible in the merged weights.
* **Tests.** `python -m pytest tests/` checks that the fast and reference implementations agree, that the result is invariant to SVD sign flips, and that the averaging baseline matches its dense definition.

## Roadmap

- [x] GAM merging, baselines and analysis code
- [ ] Preference-pair construction and LLaMA-Factory training configs
- [ ] Trained preference-delta adapters

## Citation

```bibtex
@inproceedings{sun2026pda,
  title     = {From ``Weak'' Signals to Strong Models: Preference Delta Aggregation with LoRA Merging},
  author    = {Sun, Qi and Zhang, Siyue and Chen, Yulin and Xue, Yuxiang and Peng, Ru and Zhao, Chen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

This project builds on [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory), [OpenCompass](https://github.com/open-compass/opencompass), [Agentic-Rag](https://github.com/yizhou0409/Agentic-Rag) and [PEFT](https://github.com/huggingface/peft). We thank their authors.

## License

The code in this repository is released under the [MIT License](LICENSE). Third-party submodules keep their own licenses.
