"""The spec lidar's firing pattern on Isaac (084b): CARLA's, from the same spec values."""
import os

import numpy as np
import pytest
import yaml

from aib import lidar_profile as lp

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "..", "config")
SPEC = os.path.join(HERE, "..", "..", "..", "deps", "sim_vehicle_spec")


def rig_lidar(car):
    rig = yaml.safe_load(open(os.path.join(CONFIG, f"{car}.rig.yaml")))
    return rig, rig["sensors"]["top"]


def carla_params(car):
    doc = yaml.safe_load(open(os.path.join(SPEC, "generated", car, "carla_sensors.yaml")))
    return doc, doc["sensors"]["top"]["parameters"]


@pytest.mark.parametrize("car", ["tesla_model3", "drift_mustang"])
def test_the_rigs_profile_is_carlas_pattern(car):
    rig, top = rig_lidar(car)
    doc, cp = carla_params(car)
    assert rig["spec_hash"] == doc["spec_hash"]
    p = top["isaac"]["profile"]
    assert p["name"] == f"spec_{rig['spec_hash'][:12]}_top"
    ch, rate = int(cp["channels"]), float(cp["rotation_frequency"])
    assert p["channels"] == ch and p["rate_hz"] == rate and p["range"] == float(cp["range"])
    assert p["columns"] * ch * rate == float(cp["points_per_second"])
    # CARLA's rings: even from upper_fov down to lower_fov, one a channel
    up, lo = float(cp["upper_fov"]), float(cp["lower_fov"])
    carla = [up - i * (up - lo) / (ch - 1) for i in range(ch)]
    assert np.allclose(lp.elevations(p), carla)


def test_carlas_lidar_has_no_drop_off_and_neither_has_the_profile():
    _, cp = carla_params("tesla_model3")
    for k in ("dropoff_general_rate", "dropoff_zero_intensity", "noise_stddev",
              "atmosphere_attenuation_rate"):
        assert float(cp.get(k, 0.0)) == 0.0
    _, top = rig_lidar("tesla_model3")
    a = lp.attributes(top["isaac"]["profile"])
    assert a[lp.CORE + "minReflectance"] == 0.0
    assert a[lp.CORE + "maxReturns"] == 1
    assert a[lp.CORE + "azimuthErrorStd"] == a[lp.CORE + "elevationErrorStd"] == 0.0
    assert a[lp.CORE + "rangeAccuracyM"] == 0.0


def test_attributes_fire_every_ring_once_a_column():
    p = lp.profile({"channels": "4", "rotation_frequency": "10.0", "points_per_second": "4000",
                    "upper_fov": "3.0", "lower_fov": "-3.0", "range": "50.0"}, "x")
    assert p["columns"] == 100
    a = lp.attributes(p)
    assert a[lp.STATE + "elevationDeg"] == [3.0, 1.0, -1.0, -3.0]
    assert a[lp.STATE + "azimuthDeg"] == [0.0] * 4
    assert a[lp.STATE + "channelId"] == [1, 2, 3, 4]
    assert a[lp.CORE + "reportRateBaseHz"] == a[lp.CORE + "patternFiringRateHz"] == 1000
    fire = a[lp.STATE + "fireTimeNs"]
    assert fire == sorted(fire) and fire[-1] < 1e9 / 1000       # within one column
    for k, v in a.items():
        if k.startswith(lp.STATE):
            assert len(v) == 4, k
    assert a["omni:sensor:modelName"] == "x"


def test_a_pattern_needs_whole_columns_and_a_rate():
    with pytest.raises(ValueError, match="whole number"):
        lp.profile({"channels": "3", "rotation_frequency": "10", "points_per_second": "1000",
                    "upper_fov": "1", "lower_fov": "-1", "range": "9"}, "x")
    assert lp.profile({"channels": "3", "rotation_frequency": "10", "upper_fov": "1",
                       "lower_fov": "-1", "range": "9"}, "x") is None


P = {"channels": 128, "upper_fov": 15.0, "lower_fov": -25.0, "columns": 1024, "rate_hz": 20.0,
     "range": 200.0, "name": "t"}


def test_rings_count_from_the_lowest_whatever_the_scan_has():
    step = 40.0 / 127
    e = [-25.0, -25.0 + step, 15.0, -25.0 + 64 * step + 0.05]
    assert list(lp.rings(e, P)) == [0, 1, 127, 64]
    # a scan whose top rings all see sky keeps its numbers
    assert list(lp.rings([-25.0, 0.0], P)) == [0, round(25 / step)]


def test_the_first_scan_is_checked_against_the_profile():
    rings = np.array(lp.elevations(P))
    assert lp.check(np.repeat(rings[60:], 3), P) is None
    os1 = np.linspace(20.93, -21.48, 128)                     # the stock OS1's rings
    assert "off the profile's rings" in lp.check(os1, P) or "lowest point" in lp.check(os1, P)
    assert "lowest point" in lp.check(rings[:100], P)         # nothing below -12 deg
    assert lp.check([], P) == "no points"
