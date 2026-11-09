"""A100 baseline: preserve the upstream release algorithm and physics."""

from supermimic.config import TrainConfig
from supermimic.presets import release


def a100_reference() -> TrainConfig:
    cfg = release()
    cfg.run_name = "a100_reference"
    return cfg
