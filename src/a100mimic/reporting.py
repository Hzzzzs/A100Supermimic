"""Validate and recompute complete-clip benchmark summaries."""

from __future__ import annotations

import math
from collections import Counter


def summarize_complete(rows: list[dict], expected_names: list[str]) -> dict:
    """Reject incomplete/ambiguous inputs before reporting full-library metrics.

    `norm_return` uses the complete clip length, including the unplayed part
    after termination. Trajectories with simulation flags are rejected until diagnosed.
    """
    if not expected_names or len(set(expected_names)) != len(expected_names):
        raise ValueError("expected clip IDs must be nonempty and unique")
    counts = Counter(row["name"] for row in rows)
    duplicates = sorted(name for name, n in counts.items() if n != 1)
    missing = sorted(set(expected_names) - set(counts))
    extra = sorted(set(counts) - set(expected_names))
    if duplicates or missing or extra:
        raise ValueError(f"invalid coverage: duplicate={duplicates[:5]}, missing={missing[:5]}, extra={extra[:5]}")
    returns, frames = [], []
    failed = 0
    for row in rows:
        n, steps = int(row["frames"]), int(row["steps"])
        value = float(row["norm_return"])
        if n <= 0 or not 0 < steps <= n or not math.isfinite(value):
            raise ValueError(f"invalid result for {row['name']}")
        if int(row.get("overflow_steps", 0)):
            raise ValueError(f"simulation flag for {row['name']} (bits={row.get('overflow_bits', 'unrecorded')})")
        if not row["failed"] and steps != n:
            raise ValueError(f"incomplete nonfailed clip {row['name']}")
        returns.append(value)
        frames.append(n)
        failed += bool(row["failed"])
    return {
        "clips": len(rows),
        "failed": failed,
        "failure_fraction": failed / len(rows),
        "mean_norm_return": sum(returns) / len(rows),
        "frame_weighted_norm_return": sum(r * n for r, n in zip(returns, frames)) / sum(frames),
    }
