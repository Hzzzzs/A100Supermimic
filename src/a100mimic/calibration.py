"""Open-loop error scoring for independent calibration trajectories.

The caller is responsible for collecting trajectories separately from training
replay while keeping the policy, model and normalizers frozen. This module does
not collect data or update a horizon controller.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def score_open_loop(world, states: torch.Tensor, actions: torch.Tensor, xi: torch.Tensor,
                    valid: torch.Tensor, min_windows: int = 32) -> dict:
    """Score 16/32-step prefixes using recorded executed actions.

    states: (H+1,N,71); actions: (H,N,29); xi: (N,33); valid: (H,N).
    A terminal transition is valid; its following transitions must be invalid.
    Missing windows return None and never imply trust.
    """
    if actions.ndim != 3 or states.ndim != 3 or valid.ndim != 2 or xi.ndim != 2:
        raise ValueError("invalid calibration ranks")
    horizon, count, action_dim = actions.shape
    if horizon < 32 or action_dim != 29 or states.shape != (horizon + 1, count, 71):
        raise ValueError("expected at least 32 steps and canonical G1 state/action shapes")
    if xi.shape != (count, 33) or valid.shape != (horizon, count) or valid.dtype != torch.bool:
        raise ValueError("invalid context or validity mask")
    if min_windows < 1 or not bool(torch.isfinite(states[0]).all()) or not bool(torch.isfinite(xi).all()):
        raise ValueError("invalid initial state, context or minimum coverage")
    if bool((valid[1:] & ~valid[:-1]).any()):
        raise ValueError("valid trajectory cannot resume after reset or termination")

    state = states[0]
    scores = []
    groups = ((0, 3), (3, 6), (6, 9), (9, 12), (12, 41), (41, 70))
    for step in range(32):
        mask = valid[step]
        if not bool(torch.isfinite(actions[step][mask]).all()) or not bool(torch.isfinite(states[step + 1][mask]).all()):
            raise ValueError("nonfinite valid calibration transition")
        safe_action = torch.where(mask[:, None], actions[step], torch.zeros_like(actions[step]))
        state = world.step(state, safe_action, xi)
        # Do not allow arbitrary padding to contaminate otherwise valid entries.
        target = torch.where(mask[:, None], states[step + 1], state)
        error = world.spec.error_vector(state, target)
        group_rms = torch.stack([error[:, start:end].square().mean(-1).sqrt() for start, end in groups], dim=-1)
        score = group_rms.amax(-1)
        if not bool(torch.isfinite(score[mask]).all()):
            raise ValueError("nonfinite model prediction for a valid transition")
        scores.append(torch.where(mask, score, torch.zeros_like(score)))
    curve = torch.stack(scores).cummax(dim=0).values
    result = {"total_windows": count, "valid_by_step": valid[:32].sum(dim=1).cpu().tolist()}
    for h in (16, 32):
        mask = valid[:h].all(dim=0)
        n = int(mask.sum())
        result[f"valid_{h}"] = n
        result[f"e_{h}"] = float(torch.quantile(curve[h - 1, mask], 0.9)) if n >= min_windows else None
    result["incomplete_32_fraction"] = 1.0 - result["valid_32"] / count if count else 0.0
    return result
