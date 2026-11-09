import pytest

from a100mimic.reporting import summarize_complete
from a100mimic.sets import select_dev


def test_complete_clip_scores_and_coverage():
    rows = [
        {"name": "a", "frames": 10, "steps": 5, "norm_return": 0.5, "failed": True},
        {"name": "b", "frames": 20, "steps": 20, "norm_return": 1.0, "failed": False},
    ]
    result = summarize_complete(rows, ["a", "b"])
    assert result["mean_norm_return"] == 0.75
    assert result["frame_weighted_norm_return"] == pytest.approx(25 / 30)
    assert result["failed"] == 1
    with pytest.raises(ValueError, match="coverage"):
        summarize_complete(rows + [rows[0]], ["a", "b"])
    with pytest.raises(ValueError, match="coverage"):
        summarize_complete(rows[:1], ["a", "b"])
    with pytest.raises(ValueError, match="simulation flag"):
        summarize_complete([{**rows[0], "overflow_steps": 1}, rows[1]], ["a", "b"])


def test_selection_is_stable_and_does_not_duplicate():
    names = [f"flip_090_{i}" for i in range(4)] + [f"walk_{i}" for i in range(20)]
    frames = {name: 100 + i * 20 for i, name in enumerate(names)}
    first = select_dev(names, size=10, hard_count=4, frames=frames)
    second = select_dev(list(reversed(names)), size=10, hard_count=4, frames=frames)
    assert first == second
    assert len(first["dev"]) == len(set(first["dev"])) == 10
    assert set(first["hard"]) <= set(first["dev"])
