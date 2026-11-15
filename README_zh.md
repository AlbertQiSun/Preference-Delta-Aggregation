# From "Weak" Signals to Strong Models: Preference Delta Aggregation with LoRA Merging

[![arXiv](https://img.shields.io/badge/arXiv-2606.00357-b31b1b.svg)](https://arxiv.org/abs/2606.00357)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://arxiv.org/abs/2606.00357)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | **简体中文**

Qi Sun\*, Siyue Zhang\*†, Yulin Chen, Yuxiang Xue, Ru Peng, Chen Zhao†

\* 同等贡献 &nbsp;&nbsp; † 通讯作者

NeurIPS 2026 论文的官方代码。

## 简介

训练强大的 LLM 需要高质量的监督信号，而这类信号往往稀缺。已有工作表明，由两个**较弱**模型构造的偏好对，可以借助两者之间的质量差（即 *preference delta*）来提升一个更强的模型。本文研究的问题是：多个这样的弱信号能否组合起来使用。

**Preference Delta Aggregation (PDA)** 的流程是：

1. 从每一对弱模型中构造一个 preference delta；
2. 用偏好优化把每个 delta 分别训练成一个 LoRA adapter；
3. 在参数空间中合并这些 adapter。

为了减少合并时的方向干扰，我们提出 **Geometric Alignment Merging (GAM)**：先对齐各个 adapter 的奇异子空间，再做平均。

在知识推理和 agentic search 基准上，聚合多个 delta 的效果优于任何单个 delta，并且加入的 delta 越多，提升越大。使用 GAM 时，PDA 让 Qwen3-8B 在知识推理上平均提升 6.8 分，在 agentic search 上平均提升 7.3 分，比最好的单 delta 分别高 2.1 和 4.3 分，也优于其他多 delta 合并基线。

## Geometric Alignment Merging

对同一模块的 n 个秩为 r 的更新 dW_i = B_i A_i：

1. **分解**：对每个更新做 thin SVD，dW_i = U_i S_i V_iᵀ。
2. **对齐**：两侧各做一次正交 Procrustes，对齐到参考 adapter：
   R_L,i = argmin ‖U_i R − U_ref‖，R_R,i = argmin ‖V_i R − V_ref‖，R ∈ O(r)。
   然后令 U_i ← U_i R_L,i，S_i ← R_L,iᵀ S_i R_R,i，V_i ← V_i R_R,i（这一步不改变 dW_i）。
3. **平均**：分别平均对齐后的因子，得到 Ū、S̄、V̄。
4. **重新正交化**：对 Ū、V̄ 做 thin QR，把三角因子吸收进 S̄。
5. **重构**：dW_GAM = Û Ŝ V̂ᵀ。其秩不超过 r，可以无损保存为秩 r 的 LoRA adapter，因此合并后模型的推理开销与单个 adapter 相同。

`pda/gam.py` 提供两种实现，二者在机器精度内一致（见 `tests/test_gam.py`）：

| 函数 | 做法 | 每个模块的开销 |
|---|---|---|
| `gam_merge`（默认） | 第 1 步直接从 LoRA 因子计算：两次 thin QR 加一次 r×r SVD，全程不构造 d×k 矩阵。第 4 步不改变乘积，因此省略。 | O((d + k) r²) |
| `gam_merge_dense` | 在稠密的 d×k 更新上逐步执行五个步骤（参考实现） | O(d k min(d, k)) |

`analysis/merge_cost.py` 可以测量两种实现以及各个基线的耗时和内存。

## 目录结构

```
pda/                          核心代码
  gam.py                      GAM（快速实现与参考实现）、averaging 基线
  lora_io.py                  读写 PEFT LoRA adapter
scripts/
  merge_adapters.py           合并 preference-delta adapter（GAM、averaging）
  merge_ties_peft.py          TIES-Merging 基线（PEFT add_weighted_adapter）
  build_ablation_models.py    生成 GAM 组件消融用的完整模型
analysis/
  geometry_audit.py           子空间重叠、Procrustes 统计、逐步消融阶梯
  check_init.py               检查 LoRA 初始化是否共享，以及对齐主要作用在哪一侧
  direction_retention.py      各合并算子保留了每个 delta 多少方向
  delta_magnitude.py          更新相对基座权重的大小 ||dW|| / ||W||
  merge_cost.py               各合并算子的耗时与内存
  gam_numerics.py             稠密与因子化 SVD 路径在 float32 / float64 下的精度
tests/test_gam.py             数值测试
third_party/                  git submodule：LLaMA-Factory、OpenCompass、Agentic-Rag
```

## 安装

```bash
git clone --recursive https://github.com/AlbertQiSun/Preference-Delta-Aggregation.git
cd Preference-Delta-Aggregation
pip install -r requirements.txt
```

如果克隆时没有加 `--recursive`，请运行 `git submodule update --init --recursive`。submodule 只在训练和评测时需要，合并本身只依赖 `torch` 和 `safetensors`。

## 快速开始：合并 preference-delta adapter

```bash
python scripts/merge_adapters.py \
    --adapters ckpt/delta_1 ckpt/delta_2 ckpt/delta_3 ckpt/delta_4 \
    --method gam \
    --output merged/pda_gam
```

* 所有 adapter 必须是同一基座模型上的 PEFT LoRA adapter，且 `r` 和 `lora_alpha` 相同。
* 第一个 adapter 是 Procrustes 的参考（可用 `--ref-idx` 修改）。论文中参考 adapter 选的是验证集上表现最好的单个 delta。
* 输出是标准的 PEFT adapter，可以直接用 `PeftModel.from_pretrained` 加载。加上 `--export-full-model --base-model Qwen/Qwen3-8B` 会额外保存一个已合入更新的完整模型。
* `--method average` 是 averaging 基线（对更新取平均后截断到秩 r）。`--method gam_dense` 运行参考实现。

TIES 基线：

```bash
python scripts/merge_ties_peft.py \
    --base_model Qwen/Qwen3-8B \
    --loras ckpt/delta_1 ckpt/delta_2 ckpt/delta_3 ckpt/delta_4 \
    --density 0.5 --output merged/ties
```

Python 接口：

```python
from pda import load_adapters, gam_merge, save_adapter

adapters, cfgs = load_adapters(["ckpt/delta_1", "ckpt/delta_2", "ckpt/delta_3"])
merged = {name: gam_merge([a[name][0] for a in adapters],   # lora_B, (d, r)
                          [a[name][1] for a in adapters])   # lora_A, (r, k)
          for name in adapters[0]}
save_adapter(merged, "merged/pda_gam", reference_adapter="ckpt/delta_1")
```

## 完整流程

| 阶段 | 工具 |
|---|---|
| 1. 从每对弱模型构造偏好对 | *即将发布* |
| 2. 用偏好优化把每个 preference delta 训练成 LoRA adapter | [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)（`third_party/LlamaFactory`） |
| 3. 合并 adapter | `scripts/merge_adapters.py`（本仓库） |
| 4a. Agentic search 评测（基于 Wikipedia 检索的多跳问答） | [Agentic-Rag](https://github.com/yizhou0409/Agentic-Rag)（`third_party/Agentic-Rag`） |
| 4b. 知识推理评测 | [OpenCompass](https://github.com/open-compass/opencompass)（`third_party/opencompass`） |

各 submodule 的配置方法见其自身的 README。Agentic-Rag 需要先在 Wikipedia dump 上构建 E5 索引。

## 分析脚本

所有分析脚本直接读取 PEFT adapter，在 CPU 上即可运行（GPU 可选），不需要评测模型。

```bash
# 子空间几何，以及消融阶梯 AVG -> AVG-r -> GAM-id -> GAM
python analysis/geometry_audit.py --adapters ckpt/delta_* --out results/geometry
python analysis/check_init.py --adapters ckpt/delta_* --audit-json results/geometry/geometry_audit.json

# 随 delta 数量增加，AVG-r、TIES、GAM 的方向保留度
python analysis/direction_retention.py --adapters ckpt/delta_1 ckpt/delta_2 ckpt/delta_3 ckpt/delta_4 --all-layers

# 合并开销与数值精度
python analysis/merge_cost.py --real-adapter ckpt/delta_1 --threads 16
python analysis/gam_numerics.py

# 更新相对基座权重的大小
python analysis/delta_magnitude.py --base /path/to/Qwen3-8B --adapter merged/pda_gam --apply-scaling

# 组件消融用的完整模型（avg, avg_r, gam_id, gam_id_nm, gam）
python scripts/build_ablation_models.py --base /path/to/Qwen3-8B --adapters ckpt/delta_* --out merged/ablation
```

## 实现说明

* **Procrustes 在 O(r) 上求解。** 与论文一致，GAM 在正交群上求解对齐矩阵，因此结果与 SVD 函数的符号约定无关。`--procrustes rotation` 把解限制为旋转矩阵（det R = +1），用于复现我们最早的实现，仅为此保留。
* **精度。** 默认在 CPU 上以 float64 合并（见 `--compute-dtype`）。所有脚本在 GPU 上都关闭了 TF32 matmul，因为它约 1e-3 的相对精度会在合并后的权重中体现出来。
* **测试。** `python -m pytest tests/` 会检查快速实现与参考实现是否一致、结果是否不受 SVD 符号翻转影响，以及 averaging 基线是否与其稠密定义一致。

## 计划

- [x] GAM 合并、基线与分析代码
- [ ] 偏好对构造与 LLaMA-Factory 训练配置
- [ ] 训练好的 preference-delta adapter

## 引用

```bibtex
@inproceedings{sun2026pda,
  title     = {From ``Weak'' Signals to Strong Models: Preference Delta Aggregation with LoRA Merging},
  author    = {Sun, Qi and Zhang, Siyue and Chen, Yulin and Xue, Yuxiang and Peng, Ru and Zhao, Chen},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## 致谢

本项目基于 [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)、[OpenCompass](https://github.com/open-compass/opencompass)、[Agentic-Rag](https://github.com/yizhou0409/Agentic-Rag) 和 [PEFT](https://github.com/huggingface/peft)，感谢这些项目的作者。

## 许可证

本仓库代码以 [MIT License](LICENSE) 发布。第三方 submodule 遵循各自的许可证。
