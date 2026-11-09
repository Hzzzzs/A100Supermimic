# A100 40 GB full-library seed 0

`summary.json` records the completed single-seed training run, full-library and held-out LAFAN1 evaluations with candidate solver settings, and the earlier **unaccepted raw diagnostic** evaluation. The earlier evaluation triggered simulation flags on eight clips; the candidate evaluation covered the full library and recorded no simulation flags.

Training used the filtered BONES-SEED G1 motion library and the RealMimic-derived release preset. `process_elapsed_seconds` measures initialization and training in one process, excluding queueing and offline evaluation. The GPU-memory peak is a two-second `nvidia-smi` sample, not a guaranteed instantaneous peak. The checkpoint is identified by its SHA-256; its trained weights have not been released. The full dataset, per-motion source files and pretrained weights are not included here.

LAFAN1 was reserved for external evaluation; cross-source overlap has not been ruled out. This experiment has one seed and does not establish stability across runs or exact parity with the original paper.
