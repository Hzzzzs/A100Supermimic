"""Atomic checkpoint writes and learning-process RNG state for warm resume."""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("cannot restore CUDA RNG without a CUDA device")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_save(payload: dict, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(payload, name)
        os.replace(name, destination)
    finally:
        if os.path.exists(name):
            os.unlink(name)
