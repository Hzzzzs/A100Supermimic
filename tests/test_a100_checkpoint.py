import random

import numpy as np
import torch

from a100mimic.checkpoint import atomic_save, restore_rng, rng_state


def test_warm_checkpoint_restores_random_streams(tmp_path):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    checkpoint = tmp_path / "last.pt"
    atomic_save({"rng": rng_state(), "step": 12}, checkpoint)
    expected = (random.random(), np.random.rand(), torch.rand(4))
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restore_rng(loaded["rng"])
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert loaded["step"] == 12
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    assert list(tmp_path.iterdir()) == [checkpoint]
