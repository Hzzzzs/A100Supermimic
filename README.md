# A100Mimic

English · [简体中文](README_zh-CN.md)

An unofficial, A100 40 GB-focused engineering derivative of [RealMimic](https://github.com/kevinhenryzz/realmimic), an open implementation of the [SuperMimic](https://supermimic.github.io/) world-model motion-tracking method. One policy controls a simulated Unitree G1 while tracking reference motions. This project is independent of the paper's authors and does not claim an exact reproduction of their evaluation protocol.

**Status: early release.** The full-library training run completed on one A100 40 GB. A complete re-evaluation and external LAFAN1 test passed our strict simulation-flag checks using the candidate solver settings reported below. A separate environment passed installation, a synthetic GPU fresh/resume check and an eight-motion evaluation entry-point check. The documented training CLI passed a synthetic fresh-and-warm-resume GPU check; the complete fresh-environment data download and full-training workflow have not been independently repeated. No A100Mimic weights have been published yet.

## Our experiment

| Item | Our single-seed run |
| --- | --- |
| GPU | 1 × NVIDIA A100-SXM4 40 GB |
| Training motions | 129,491 filtered BONES-SEED G1 clips |
| Training | fresh seed 0; 86,000 iterations, including 500 world-model-only warm-up iterations |
| Training process time | 12 h 59 min 51 s, including initialization; excludes offline evaluation and queueing |
| GPU memory | 21,189 / 40,960 MiB peak observed by `nvidia-smi` sampled every 2 seconds; brief peaks may be missed |
| Training-library evaluation | 6 / 129,491 early terminations; mean normalized return 0.9544; 0 simulator-flagged clips |
| Hard training subset | 4 / 232 early terminations; mean normalized return 0.7833 |
| External LAFAN1 | 13 / 40 early terminations; mean normalized return 0.6768; 0 simulator-flagged clips |

The accepted evaluations used solver iterations 32, line-search iterations 40 and per-world constraint capacity 500, which differ from training and are not claimed to match the paper's protocol. The 129,491 clips are training motions. LAFAN1 was reserved for external evaluation, though cross-source overlap has not been ruled out. An earlier evaluation using solver 10/40 and capacity 250 recorded eight simulator-flagged clips and remains invalid; its raw values are retained in the [experiment record](experiments/a100_40gb_seed0/summary.json). These results are from one seed and do not establish run-to-run stability.

During training, the log recorded 58 reporting intervals with a constraint-capacity overflow warning and a final cumulative count of 1,135 joint-velocity anomalies. These training diagnostics are not evaluation failures; their effect on the learned policy has not been established.

## Requirements

- Linux x86-64, a single NVIDIA A100 40 GB, and a compatible NVIDIA driver. Our run used driver 570.124.06, Python 3.12, PyTorch 2.14.1+cu126, and Warp 1.18.0+cu12.
- Around 170 GB of free disk for the full downloaded, extracted, and converted dataset, plus space for logs and checkpoints. The measured cluster job requested 16 CPU cores and 110 GB RAM; those are requests, not proven minimums.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) and access to [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed). Each user must obtain access and follow its terms. Data and model weights are not included in the code repository.

## Install and prepare data

Clone this repository, then from the project directory:

```bash
git clone https://github.com/Hzzzzs/A100Mimic.git
cd A100Mimic
uv sync --frozen
uv run hf auth login
# Accept the BONES-SEED terms and wait for access before downloading.
tools/download_data.sh
tools/prepare_data.sh
uv run python scripts/a100_make_sets.py data/npz/bones_seed/manifest_filtered.csv data/a100_sets
```

The preparation script converts and filters BONES-SEED and creates `data/library/bones_paper.pt` for training and `data/bench/full.txt` for evaluation. The last command produces our development and hard sets in `data/a100_sets`: our hard set includes lying/resting/standing-up actions and differs from the upstream `data/bench/hard.txt`. For our recorded data, the generated `manifest.json` should report 129,491 full motions (SHA-256 `d05ea3ef335802de8b80d467e5b8dd3af0029a3405cc0770bc26565528e3375d`) and 232 hard motions (SHA-256 `869caba477075cbd8b96b68406831beed98b75cfa387167c7ae5e2fbe92a5058`). These are hashes of the local lists; do not upload the lists themselves. LAFAN1 is reserved for external evaluation. The complete fresh-install and download workflow is not yet independently verified on a second machine.

## Train and resume

```bash
# Fresh full-library run; do not start a second copy with the same run name.
uv run sm-train a100_reference --run-name a100_seed0 \
  --schedule.checkpoint-interval 20000 --schedule.max-hours 19

# Resume after interruption with the saved models, optimizers, and counters.
uv run sm-train a100_reference --run-name a100_seed0 \
  --resume logs/a100_seed0/ckpt_latest.pt
```

Resume is a **warm resume**: the simulator resets and the replay buffer is refilled, so it does not reproduce the exact same trajectory as an uninterrupted run. Use a time limit suitable for your scheduler; queue time and offline evaluation are not included in the training estimate. The experimental adaptive-horizon code is disabled by default and has not demonstrated a performance gain.

## Evaluate

The following uses the candidate solver settings of the accepted full-library re-evaluation:

```bash
uv run python scripts/eval_benchmark.py \
  --ckpt logs/a100_seed0/ckpt_latest.pt \
  --npz-dir data/npz/bones_seed --clip-list data/bench/full.txt \
  --num-envs 256 --chunk-size 256 --solver-iterations 32 \
  --solver-ls-iterations 40 --njmax 500 --out runs/full_candidate.json

uv run python scripts/a100_report.py data/bench/full.txt runs/full_candidate.json
```

The report refuses incomplete, duplicate, nonfinite or simulation-flagged results. Evaluation can take several hours on a single A100 and is separate from training time. For multiple shards, supply each evaluator JSON to `a100_report.py` with the same checkpoint and solver settings. The evaluation CLI has been checked on a GPU; the complete fresh-install workflow remains to be verified.

## Origin and licenses

A100Mimic derives from RealMimic commit `61f2973f6426255a8c844d6ca0c894eff4d8f45c`, which incorporates MIT-licensed `Supermimic_mod`. See [UPSTREAM.md](UPSTREAM.md), [NOTICE](NOTICE), [LICENSE](LICENSE), and [LICENSES](LICENSES/). The original upstream README and its results are retained in [docs/REALMIMIC_UPSTREAM_README.md](docs/REALMIMIC_UPSTREAM_README.md); those results and media are not A100Mimic experiments.

BONES-SEED, LAFAN1, third-party assets, upstream model weights and any weights trained on restricted data have their own terms; the code license does not grant rights to redistribute them. Our future model release will state its applicable terms separately.

Training data includes Motion Data by [Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the [BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license).
