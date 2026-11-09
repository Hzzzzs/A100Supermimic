from types import SimpleNamespace

import torch

from a100mimic.collector import extract_first_episodes


def test_extraction_keeps_terminal_target_but_not_reset_state():
    n, capacity = 2, 68
    b = SimpleNamespace(E=n, T=capacity, device=torch.device("cpu"))
    b.s = torch.zeros(n, capacity, 71)
    b.a_exec = torch.zeros(n, capacity, 29)
    b.xi = torch.zeros(n, capacity, 33)
    b.valid = torch.zeros(n, capacity, dtype=torch.bool)
    b.terminal = torch.zeros(n, capacity, dtype=torch.bool)
    b.ep_id = torch.full((n, capacity), -1)
    b.ptr = torch.tensor([5, 34])
    b.valid[0, :5] = True
    b.ep_id[0, :3] = 0
    b.ep_id[0, 3:5] = 1
    b.terminal[0, 2] = True
    b.s[0, 2, 0] = 123.0  # terminal state
    b.s[0, 3, 0] = 999.0  # next episode's reset state
    b.valid[1, :34] = True
    b.ep_id[1, :34] = 0
    states, actions, xi, valid = extract_first_episodes(b)
    assert valid[:, 0].sum().item() == 2
    assert states[2, 0, 0].item() == 123.0
    assert not valid[2:, 0].any()
    assert states[3, 0, 0].item() == 0.0
    assert valid[:, 1].sum().item() == 32
