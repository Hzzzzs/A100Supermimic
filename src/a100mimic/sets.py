"""Deterministic dataset subsets derived only from names and categories."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque

from realmimic.eval.motion_groups import HARD_SET_GROUPS, motion_group

HARD_GROUPS = frozenset((*HARD_SET_GROUPS, "lying_resting_standing_up"))

def select_dev(names: list[str], size: int = 2000, hard_count: int = 400,
               frames: dict[str, int] | None = None) -> dict[str, list[str]]:
    if len(set(names)) != len(names) or len(names) < size or not 0 <= hard_count <= size:
        raise ValueError("unique names and sufficient dataset size required")

    def key(name: str) -> tuple[bytes, str]:
        return hashlib.sha256(("a100mimic-v1" + name).encode()).digest(), name

    hard = sorted((n for n in names if motion_group(n) in HARD_GROUPS), key=key)
    other = sorted((n for n in names if motion_group(n) not in HARD_GROUPS), key=key)
    take_hard = min(hard_count, len(hard))
    take_other = min(size - take_hard, len(other))
    # If other clips are exhausted, fill from the unused hard clips.
    chosen = hard[:take_hard] + other[:take_other]
    chosen += hard[take_hard : take_hard + size - len(chosen)]
    # Round-robin category and length terciles, with stable hashes within each cell.
    if frames is not None and any(n not in frames or frames[n] <= 0 for n in chosen):
        raise ValueError("positive frame count required for every selected clip")
    lengths = sorted(frames[n] for n in chosen) if frames is not None else []
    cuts = (lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]) if lengths else (0, 0)
    cells = defaultdict(list)
    for name in chosen:
        length_bin = sum(frames[name] > cut for cut in cuts) if frames is not None else 0
        cells[(motion_group(name), length_bin)].append(name)
    queues = [deque(sorted(cells[cell], key=key)) for cell in sorted(cells)]
    smoke = []
    while len(smoke) < min(64, size):
        for queue in queues:
            if queue and len(smoke) < min(64, size):
                smoke.append(queue.popleft())
    return {"dev": sorted(chosen), "hard": sorted(hard), "smoke": sorted(smoke)}
