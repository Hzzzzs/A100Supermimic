"""Experimental error-calibrated H16/H32 policy; not enabled by default."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HorizonController:
    tau: float
    horizon: int = 16
    consecutive_upgrade: int = 0
    min_windows: int = 32

    def __post_init__(self):
        if not math.isfinite(self.tau) or self.tau <= 0:
            raise ValueError("tau must be positive and finite")
        if self.horizon not in (16, 32) or self.min_windows < 1:
            raise ValueError("invalid horizon or minimum coverage")

    def update(self, e16: float | None, n16: int, e32: float | None, n32: int) -> dict:
        """Use independent calibration scores; never interpret missing data as trust."""
        valid16 = n16 >= self.min_windows and e16 is not None and math.isfinite(e16)
        valid32 = n32 >= self.min_windows and e32 is not None and math.isfinite(e32)
        if not valid16 or not valid32:
            self.horizon = 16
            self.consecutive_upgrade = 0
            return {"horizon": self.horizon, "status": "coverage_unknown"}
        status = "out_of_tolerance" if e16 > self.tau else "ok"
        if self.horizon == 32:
            if e32 > 1.2 * self.tau:
                self.horizon = 16
                self.consecutive_upgrade = 0
        elif e32 <= 0.8 * self.tau:
            self.consecutive_upgrade += 1
            if self.consecutive_upgrade >= 2:
                self.horizon = 32
                self.consecutive_upgrade = 0
        else:
            self.consecutive_upgrade = 0
        return {"horizon": self.horizon, "status": status}
