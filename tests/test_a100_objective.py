from types import SimpleNamespace

import pytest
import torch

from supermimic.imagination.objective import gae_targets, policy_objective


@pytest.mark.parametrize("horizon", [16, 32])
def test_objective_bootstrap_keeps_terminal_state_gradient(horizon):
    gamma = 0.9
    terminal_state = torch.tensor([2.0], requires_grad=True)
    rollout = SimpleNamespace(rewards=torch.ones(horizon, 1), survival=torch.ones(horizon + 1, 1))
    objective = policy_objective(rollout, terminal_state, gamma)
    expected = (1.0 - gamma**horizon) / (1.0 - gamma) + 2.0 * gamma**horizon
    assert objective.item() == pytest.approx(expected, rel=1e-6)
    objective.backward()
    assert terminal_state.grad.item() == pytest.approx(gamma**horizon, rel=1e-6)


def test_gae_terminal_does_not_bootstrap_past_termination():
    rewards = torch.ones(2, 1)
    values = torch.full((3, 1), 10.0)
    terminal = torch.tensor([[0.0], [1.0]])
    result = gae_targets(rewards, terminal, values, 0.9, 1.0)
    torch.testing.assert_close(result, torch.tensor([[1.9], [1.0]]))
    assert not result.requires_grad
