"""Collect isolated calibration trajectories using a separate simulator.

No trajectory enters the training recorder. RNG state is restored on exit;
the calibration environment's reset sampler is initialized independently and
does not inherit the training sampler's accumulated statistics.
"""

from __future__ import annotations

from dataclasses import replace
import random

import numpy as np
import torch

from a100mimic.checkpoint import restore_rng, rng_state


def extract_first_episodes(buffer, steps: int = 32) -> tuple[torch.Tensor, ...]:
    """Extract first-episode prefixes, retaining terminal targets before reset.

    The caller must ensure the ring cannot wrap during collection. For every
    valid transition, consecutive rows belong to the same episode. An invalid
    suffix remains invalid; no reset state is substituted for a terminal target.
    """
    if steps < 1 or buffer.T < 2 * (steps + 2):
        raise ValueError("calibration replay capacity is too small")
    n, device = buffer.E, buffer.device
    states = torch.zeros((steps + 1, n, buffer.s.shape[-1]), device=device)
    states[0, :, 3] = 1.0  # harmless identity quaternion for missing windows
    actions = torch.zeros((steps, n, buffer.a_exec.shape[-1]), device=device)
    xi = torch.zeros((n, buffer.xi.shape[-1]), device=device)
    valid = torch.zeros((steps, n), dtype=torch.bool, device=device)
    for env_id in range(n):
        count = int(buffer.ptr[env_id])
        if count == 0 or not bool(buffer.valid[env_id, 0]):
            continue
        ep = buffer.ep_id[env_id, 0]
        states[0, env_id] = buffer.s[env_id, 0]
        xi[env_id] = buffer.xi[env_id, 0]
        for t in range(min(steps, count - 1)):
            if not (bool(buffer.valid[env_id, t]) and bool(buffer.valid[env_id, t + 1])):
                break
            if bool(buffer.terminal[env_id, t]) or buffer.ep_id[env_id, t] != ep or buffer.ep_id[env_id, t + 1] != ep:
                break
            states[t + 1, env_id] = buffer.s[env_id, t + 1]
            actions[t, env_id] = buffer.a_exec[env_id, t]
            valid[t, env_id] = True
    return states, actions, xi, valid


@torch.no_grad()
def collect_independent(env_args, policy, *, seed: int, num_envs: int = 64, steps: int = 32,
                        device: str = "cuda:0") -> tuple[torch.Tensor, ...]:
    """Collect prefixes under a frozen policy and restore the caller's RNG.

    Collection uses a private replay and no optimizer/normalizer updates. It
    must run synchronously when no other thread consumes global RNG. The caller
    owns advancing/persisting `seed` and must account for calibration cost.
    """
    if num_envs < 1 or steps != 32:
        raise ValueError("expected positive env count and 32 calibration steps")
    from supermimic.envs import make_env

    saved_rng = rng_state()
    env = None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        args = replace(env_args, num_envs=num_envs, seed=seed, replay_capacity=2 * (steps + 2), record_transitions=True)
        env = make_env(args, device=device)
        obs, _ = env.reset()
        for _ in range(steps + 1):
            action = policy(obs["policy"], obs["goal"])
            if not bool(torch.isfinite(action).all()):
                raise ValueError("nonfinite calibration action")
            obs, _, _, _, _ = env.step(action)
        recorder = env.recorder_manager.get_term("replay")
        return extract_first_episodes(recorder.buffer, steps)
    finally:
        if env is not None:
            env.close()
        restore_rng(saved_rng)
