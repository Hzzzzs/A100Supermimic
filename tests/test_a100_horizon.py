import math

import pytest

from a100mimic.horizon import HorizonController


def test_hysteresis_and_missing_coverage():
    ctl = HorizonController(tau=1.0)
    assert ctl.update(.4, 64, .7, 64)["horizon"] == 16
    assert ctl.update(.4, 64, .7, 64)["horizon"] == 32
    assert ctl.update(.4, 64, 1.0, 64)["horizon"] == 32
    assert ctl.update(.4, 64, 1.3, 64)["horizon"] == 16
    assert ctl.update(1.1, 64, 1.3, 64)["status"] == "out_of_tolerance"
    assert ctl.update(.4, 64, math.nan, 64)["status"] == "coverage_unknown"
    assert ctl.horizon == 16


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        HorizonController(tau=0)
