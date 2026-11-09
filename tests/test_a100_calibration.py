import pytest
import torch

from a100mimic.calibration import score_open_loop


class Spec:
    def error_vector(self, pred, true):
        return (pred - true)[..., :70]


class Model:
    spec = Spec()

    def step(self, state, action, xi):
        out = state.clone()
        out[:, 0] += action[:, 0]
        return out


def test_open_loop_coverage_and_prefix_max():
    states = torch.zeros(33, 4, 71)
    actions = torch.zeros(32, 4, 29)
    actions[0, :, 0] = 1.0
    actions[1, :, 0] = -1.0  # final error returns to zero but prefix error must remain
    valid = torch.ones(32, 4, dtype=torch.bool)
    valid[20:, 0] = False
    actions[20:, 0] = float("nan")  # padding after termination must not poison valid windows
    result = score_open_loop(Model(), states, actions, torch.zeros(4, 33), valid, min_windows=2)
    assert result["valid_16"] == 4 and result["valid_32"] == 3
    assert result["e_16"] == pytest.approx(1 / (3**0.5))
    assert result["e_32"] == pytest.approx(result["e_16"])
    assert result["incomplete_32_fraction"] == pytest.approx(0.25)
    assert score_open_loop(Model(), states, actions, torch.zeros(4, 33), valid, min_windows=4)["e_32"] is None


def test_rejects_windows_that_cross_reset_or_nonfinite_live_actions():
    states = torch.zeros(33, 1, 71)
    actions = torch.zeros(32, 1, 29)
    valid = torch.ones(32, 1, dtype=torch.bool)
    valid[10] = False
    with pytest.raises(ValueError, match="reset or termination"):
        score_open_loop(Model(), states, actions, torch.zeros(1, 33), valid, min_windows=1)
    valid[:] = True
    actions[10, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        score_open_loop(Model(), states, actions, torch.zeros(1, 33), valid, min_windows=1)
