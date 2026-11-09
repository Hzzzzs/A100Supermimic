# A100Mimic

[English](README.md) · 简体中文

A100Mimic 是基于 [RealMimic](https://github.com/kevinhenryzz/realmimic) 的非官方 SuperMimic 派生实现，目标是让使用单张 A100 40GB 的研究者能够复现 G1 机器人的仿真动作跟踪训练。项目使用可微世界模型，保留上游方法和来源；不声称与论文完全相同的实验设置。

**当前状态：早期发布。** 我们完成了一次单卡全库训练，并在独立环境完成了合成短训练、恢复、真实动作评测入口及文档训练命令的短训练和恢复检查。全库重评和 LAFAN1 外部评测已完成，权重尚未发布；完整的普通用户从零安装、数据下载和训练流程还需要验收。

## 我们的实验

| 内容 | 单次 seed 0 实测 |
| --- | --- |
| 显卡 | 1 × A100-SXM4 40GB |
| 训练数据 | 129,491 条过滤后的 BONES-SEED G1 动作 |
| 训练 | 从零训练 86,000 轮，含 500 轮世界模型预热 |
| 训练进程耗时 | 约 12 小时 59 分 51 秒，含初始化；不含排队与离线评测 |
| 显存 | `nvidia-smi` 每 2 秒采样，最高观察到 21,189 / 40,960 MiB；可能漏掉短暂峰值 |
| 训练库重评 | 129,491 条中 6 条提前终止，平均归一化回报 0.9544，仿真标志 0 条 |
| 训练库困难子集 | 232 条中 4 条提前终止，平均归一化回报 0.7833 |
| LAFAN1 外部测试 | 40 条中 13 条提前终止，平均归一化回报 0.6768，仿真标志 0 条 |

全库和困难子集均为训练动作；LAFAN1 留作外部评测，但不能排除跨来源的相似动作。上述评测采用主求解器 32、线搜索 40、`njmax=500` 的候选设置，与训练设置不同，也不声称与论文协议完全一致。此前主求解器 10、线搜索 40、`njmax=250` 的全库原始评测有 8 条仿真标志，仍为无效记录，详见[实验记录](experiments/a100_40gb_seed0/summary.json)。目前仅有一个训练种子。

训练日志还在 58 个记录周期中报告过约束容量溢出，最终累计记录 1,135 次关节速度异常。这些是训练过程的诊断记录，不等同于评测失败；它们对最终模型的影响尚未确定。

## 环境与数据

已实际测试的训练设备为 Linux x86-64、A100-SXM4 40GB、NVIDIA 驱动 570.124.06、Python 3.12、PyTorch 2.14.1+cu126 和 Warp 1.18.0+cu12。完整下载、解压和转换数据建议预留约 170GB 磁盘；我们的集群任务申请了 16 CPU 核和 110GB 内存，这不是最低资源要求。

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，自行申请 [BONES-SEED 访问](https://huggingface.co/datasets/bones-studio/seed)并遵守数据协议。在项目目录执行：

```bash
uv sync --frozen
uv run hf auth login
tools/download_data.sh
tools/prepare_data.sh
uv run python scripts/a100_make_sets.py data/npz/bones_seed/manifest_filtered.csv data/a100_sets
```

数据脚本使用固定版本下载 BONES-SEED 与 LAFAN1；数据不包含在代码仓库内。最后一条命令在本地生成我们的开发集和困难集：我们的困难集包含躺卧、休息和起身动作，与上游 `data/bench/hard.txt` 不同。使用我们实验的数据时，生成的 `manifest.json` 应显示全库 129,491 条、列表 SHA-256 为 `d05ea3ef335802de8b80d467e5b8dd3af0029a3405cc0770bc26565528e3375d`，困难集 232 条、列表 SHA-256 为 `869caba477075cbd8b96b68406831beed98b75cfa387167c7ae5e2fbe92a5058`。列表保留本地，不上传。LAFAN1 留作外部评测，不参与训练。

## 训练与恢复

```bash
uv run sm-train a100_reference --run-name a100_seed0 \
  --schedule.checkpoint-interval 20000 --schedule.max-hours 19

# 从保存的检查点恢复
uv run sm-train a100_reference --run-name a100_seed0 \
  --resume logs/a100_seed0/ckpt_latest.pt
```

恢复会加载模型、优化器和计数，但会重置仿真环境并重新填充 replay buffer，因此不是逐条轨迹完全一致的恢复。训练时间以实际设备为准。

## 评测

下面使用全库重评通过的候选求解器设置：

```bash
uv run python scripts/eval_benchmark.py \
  --ckpt logs/a100_seed0/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed --clip-list data/bench/full.txt \
  --num-envs 256 --chunk-size 256 --solver-iterations 32 \
  --solver-ls-iterations 40 --njmax 500 --out runs/full_candidate.json
uv run python scripts/a100_report.py data/bench/full.txt runs/full_candidate.json
```

汇总器会拒绝缺失、重复、非有限值和带仿真标志的结果。完整评测另计时间。模型权重、正式协议及对应下载方式会在核验和许可范围确认后补充。

## 来源与许可

本项目基于 RealMimic 固定版本 `61f2973f6426255a8c844d6ca0c894eff4d8f45c`，其中包含 MIT 许可的 Supermimic_mod 训练核心。请参阅 [UPSTREAM.md](UPSTREAM.md)、[NOTICE](NOTICE)、[LICENSE](LICENSE) 和 [LICENSES](LICENSES/)。[上游 README](docs/REALMIMIC_UPSTREAM_README.md) 中的成绩和素材属于原作者，不是本项目的成绩。

Training data includes Motion Data by [Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the [BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).

BONES-SEED、LAFAN1、第三方资产以及权重有各自的使用条款；代码许可证不自动授予数据或权重的再分发权限。
