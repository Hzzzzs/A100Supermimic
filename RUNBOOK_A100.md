# A100Mimic working runbook

The main installation, training and preliminary evaluation instructions are in [README.md](README.md). A100Mimic is an unofficial derivative of RealMimic; see [UPSTREAM.md](UPSTREAM.md), LICENSE and NOTICE.

The one-GPU full BONES-SEED seed-0 run completed 86,000 iterations in 12 h 59 min 51 s of training-process time on an A100-SXM4-40GB. It used 4,096 environments, H32, a 2,048 policy batch, enabled self-collision, and the count-aware sampler inherited from RealMimic. The adaptive-horizon extension was off. Sampled `nvidia-smi` memory peaked at 21,189 MiB (2-second interval). The result is one run, not a multi-seed claim.

The original 129,491-clip full-library evaluation at solver 10/40 and `njmax=250` contained eight simulation-flagged clips and remains invalid. A complete re-evaluation at solver 32/40 and `njmax=500` had zero flagged clips, six early terminations and mean normalized return 0.95439. The reserved 40-clip LAFAN1 evaluation under the same candidate solver settings had zero flagged clips, 13 early terminations and mean normalized return 0.67676. Training used solver 10/20. These settings are not claimed to reproduce the paper's evaluation protocol; see [experiment summary](experiments/a100_40gb_seed0/summary.json).

The checkpoint contains model and optimizer state, sampler state and process RNG. Resume resets the simulator and refills replay; it is a warm resume, not an exact continuation of individual trajectories.

BONES-SEED data is gated and is not distributed with the code. LAFAN1 remains an external evaluation set. Data, upstream assets and weights have terms distinct from the code license. Our final trained checkpoint has not been published.
