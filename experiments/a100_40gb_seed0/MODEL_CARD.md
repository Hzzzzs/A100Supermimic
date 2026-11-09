# A100Supermimic seed 0 model card

This checkpoint is an unofficial RealMimic-derived simulation policy for a Unitree G1 humanoid. Given a reference motion, it produces actions to track that motion. It has not been tested on a physical robot. The checkpoint is not currently distributed.

It was trained from scratch for 86,000 iterations on 129,491 filtered BONES-SEED motions with one NVIDIA A100-SXM4-40GB. The training process took 12 h 59 min 51 s including initialization, excluding offline evaluation and queue time. Its SHA-256 is `faa4108db6f088f4c498a03b23596aa3a2990bbcb85fa6e3d1fbbd5e99deed42`.

Using solver iterations 32, line-search iterations 40, and constraint capacity 500, the complete training library had 6 early terminations out of 129,491 and mean normalized return 0.9544. The reserved external LAFAN1 set had 13 early terminations out of 40 and mean return 0.6768. No simulation flags were recorded in these evaluations. LAFAN1 was not used for this training run; cross-source motion similarity has not been ruled out. The two datasets differ in motion length and composition, so their scores should not be interpreted as a controlled comparison.

The evaluation solver differs from the training solver and is not claimed to match the paper's evaluation protocol. Earlier full-library evaluation at 10/40/250 had simulation flags and is invalid. This is one training seed; training logged constraint overflow warnings and joint velocity anomalies with unknown effects. See [summary.json](summary.json) for the recorded settings and results.

The repository's code license does not grant rights to redistribute training data, derived videos, or checkpoint weights. Any future weight distribution must state applicable terms separately.
