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


def test_a_scan_on_the_profiles_rings_passes():
    rings = np.array(lp.elevations(P))
    assert lp.check(np.repeat(rings, 3), P) is None
    assert lp.check(rings[rings < 2.0], P) is None            # upper rings in the sky
    assert lp.check(rings + 0.05, P) is None                   # a little pitch in the sweep
    assert lp.check([], P) == "no points"


def test_a_sparser_pattern_over_the_same_field_fails_off_the_rings():
    e = np.linspace(15.0, -25.0, 64)
    assert "off the profile's rings" in lp.check(e, P)


def test_a_sparser_pattern_on_alternate_rings_fails_on_coverage():
    rings = np.array(lp.elevations(P))
    assert "rings at or below -1 deg have a point" in lp.check(rings[::2], P)


def test_the_stock_os1_fails_off_the_rings():
    os1 = np.linspace(20.93, -21.48, 128)                     # 0.34 deg apart, +-21 deg
    assert "off the profile's rings" in lp.check(os1, P)


def test_a_scan_missing_its_lowest_rings_fails_on_coverage():
    rings = np.array(lp.elevations(P))
    assert "rings at or below -1 deg have a point" in lp.check(rings[rings > -21.5], P)


def test_phase_jitter_is_opt_in_seeded_and_within_a_column():
    assert lp.PhaseJitter.of({"columns": 1024}) is None
    assert lp.PhaseJitter.of(None) is None
    a = lp.PhaseJitter.of({"columns": 1024, "phase_jitter": {"seed": 3}})
    b = lp.PhaseJitter.of({"columns": 1024, "phase_jitter": {"seed": 3}})
    da, db = [a.next() for _ in range(500)], [b.next() for _ in range(500)]
    assert da == db                                   # a seed repeats the run
    assert 0.0 <= min(da) and max(da) < 360.0 / 1024
    assert max(da) - min(da) > 0.9 * 360.0 / 1024     # spreads over the column
    assert lp.PhaseJitter.of({"columns": 1024, "phase_jitter": None}).seed == 0
    c = lp.PhaseJitter.of({"columns": 1024, "phase_jitter": {"seed": 4}})
    assert [c.next() for _ in range(5)] != da[:5]


def test_unturn_puts_a_turned_lidars_points_back():
    # a point straight ahead of the world, seen by a lidar turned +10 deg about z, lies at
    # -10 deg in the turned lidar's frame; unturn returns it to straight ahead
    d = 10.0
    seen = np.array([[np.cos(np.radians(-d)) * 5, np.sin(np.radians(-d)) * 5, 1.0]])
    np.testing.assert_allclose(lp.unturn(seen, d), [[5.0, 0.0, 1.0]], atol=1e-9)
    np.testing.assert_allclose(lp.unturn(seen, 0.0), seen)


def test_phase_jitter_streams_differ_per_lidar_and_need_columns():
    p = {"columns": 1024, "phase_jitter": {"seed": 1}}
    a, b = lp.PhaseJitter.of(p, "top"), lp.PhaseJitter.of(p, "left")
    assert [a.next() for _ in range(5)] != [b.next() for _ in range(5)]
    c = lp.PhaseJitter.of(p, "top")
    assert c.next() == lp.PhaseJitter.of(p, "top").next()      # stable: crc32, not hash()
    with pytest.raises(ValueError, match="columns"):
        lp.PhaseJitter.of({"name": "x", "phase_jitter": {}}, "top")


def test_a_planar_profile_has_one_ring():
    """091 W19: a one-ring profile (UST-10LX class) checks and numbers its points."""
    import numpy as np

    from aib import lidar_profile as lp

    p = {"channels": 1, "upper_fov": 0.0, "lower_fov": 0.0, "columns": 1440, "rate_hz": 40.0,
         "range": 10.0}
    assert lp.check(np.zeros(1000), p) is None
    assert "off the plane" in lp.check(np.r_[np.zeros(900), np.full(100, 0.5)], p)
    assert lp.rings(np.zeros(5), p).tolist() == [0] * 5


def test_the_lidar_ticks_at_its_scan_rate():
    """Issue 099: the stock variant's 20 Hz tick is replaced by the profile's scan rate."""
    from aib.lidar_profile import tick_rate
    assert tick_rate({"rate_hz": 40.0}, 20.0) == 40.0
    assert tick_rate({"rate_hz": 10.0}, 20.0) == 10.0
    assert tick_rate({"rate_hz": 10.0}, 10.0) is None
    assert tick_rate({"rate_hz": 10.0}, None) == 10.0
