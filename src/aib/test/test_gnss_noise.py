"""The GNSS receiver's noise (scensim issue 195 A9): RTK-like by default, seeded, off on request.
Pure (numpy)."""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from _paths import RIG
from aib.control import State
from aib.rig import EgoConfig
from aib.sensing import (GNSS_NOISE_DEFAULTS, GnssNoise, gnss_noise_settings, gnss_reading)


def fixes(settings, n):
    noise = GnssNoise(gnss_noise_settings(settings))
    return np.array([noise.offset() for _ in range(n)]), noise


class SettingsTest(unittest.TestCase):
    def test_the_defaults_are_rtk_like(self):
        s = gnss_noise_settings({})
        self.assertEqual(s, GNSS_NOISE_DEFAULTS)
        self.assertEqual((s["enabled"], s["horizontal_stddev_m"], s["vertical_stddev_m"],
                          s["bias_stddev_m"], s["seed"]), (True, 0.02, 0.04, 0.0, None))
        self.assertEqual(gnss_noise_settings(None), GNSS_NOISE_DEFAULTS)

    def test_unknown_keys_and_bad_values_are_refused(self):
        with self.assertRaisesRegex(ValueError, r"unknown key\(s\) \['noise_m'\]"):
            gnss_noise_settings({"noise_m": 0.02})
        for k in ("horizontal_stddev_m", "vertical_stddev_m", "bias_stddev_m"):
            with self.assertRaisesRegex(ValueError, f"{k} must be >= 0"):
                gnss_noise_settings({k: -0.1})
        with self.assertRaisesRegex(ValueError, "seed"):
            gnss_noise_settings({"seed": 1.5})
        with self.assertRaisesRegex(ValueError, "seed"):
            gnss_noise_settings({"seed": -1})
        with self.assertRaisesRegex(ValueError, "enabled"):
            gnss_noise_settings({"enabled": "yes"})


class NoiseTest(unittest.TestCase):
    def test_white_noise_has_the_configured_spread_and_no_mean(self):
        d, _ = fixes({"seed": 7}, 20000)
        sd = d.std(axis=0)
        # 20000 samples: the sample sd is within 2 % of sigma (about 4 standard errors)
        self.assertTrue(abs(sd[0] - 0.02) < 0.0004 and abs(sd[1] - 0.02) < 0.0004, sd)
        self.assertTrue(abs(sd[2] - 0.04) < 0.0008, sd)
        se = np.array([0.02, 0.02, 0.04]) / math.sqrt(20000)
        self.assertTrue(np.all(np.abs(d.mean(axis=0)) < 4 * se))
        # independent axes
        c = np.corrcoef(d.T)
        self.assertTrue(np.all(np.abs(c[np.triu_indices(3, 1)]) < 0.05), c)

    def test_a_bias_is_drawn_once_per_receiver(self):
        d, noise = fixes({"seed": 3, "horizontal_stddev_m": 0.0, "vertical_stddev_m": 0.0,
                          "bias_stddev_m": 0.5}, 100)
        self.assertTrue(np.all(d == d[0]))  # constant over the run
        self.assertGreater(np.abs(d[0]).max(), 0.0)
        self.assertEqual(tuple(d[0]), tuple(noise.bias))
        biases = np.array([GnssNoise(gnss_noise_settings({"seed": s, "bias_stddev_m": 0.5})).bias
                           for s in range(2000)])
        self.assertTrue(np.all(np.abs(biases.std(axis=0) - 0.5) < 0.05), biases.std(axis=0))

    def test_the_same_seed_gives_the_same_fixes(self):
        a, _ = fixes({"seed": 42, "bias_stddev_m": 0.1}, 50)
        b, _ = fixes({"seed": 42, "bias_stddev_m": 0.1}, 50)
        c, _ = fixes({"seed": 43, "bias_stddev_m": 0.1}, 50)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_no_seed_draws_a_fresh_one_and_names_it(self):
        _, n1 = fixes({}, 1)
        _, n2 = fixes({}, 1)
        self.assertIsInstance(n1.seed, int)
        self.assertNotEqual(n1.seed, n2.seed)
        # the named seed replays the run
        r1 = GnssNoise(gnss_noise_settings({"seed": n1.seed}))
        r2 = GnssNoise(gnss_noise_settings({"seed": n1.seed}))
        self.assertEqual([r1.offset() for _ in range(5)], [r2.offset() for _ in range(5)])

    def test_disabled_is_the_truth(self):
        d, noise = fixes({"enabled": False, "seed": 1, "bias_stddev_m": 1.0}, 10)
        self.assertTrue(np.all(d == 0.0))
        self.assertEqual(noise.covariance(), ([0.0] * 9, 0))

    def test_the_covariance_is_diagonal_and_known(self):
        cov, kind = GnssNoise(gnss_noise_settings({"seed": 0, "bias_stddev_m": 0.01})).covariance()
        self.assertEqual(kind, 2)  # NavSatFix COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.assertAlmostEqual(cov[0], 0.02 ** 2 + 0.01 ** 2)
        self.assertAlmostEqual(cov[4], 0.02 ** 2 + 0.01 ** 2)
        self.assertAlmostEqual(cov[8], 0.04 ** 2 + 0.01 ** 2)
        self.assertEqual([cov[i] for i in (1, 2, 3, 5, 6, 7)], [0.0] * 6)


class ReadingTest(unittest.TestCase):
    def test_the_offset_moves_the_fix_east_north_up(self):
        s = State(position=(100.0, 200.0, 5.0))
        lat0, lon0, alt0 = gnss_reading(s, (0.0, 0.0, 0.0))
        lat1, lon1, alt1 = gnss_reading(s, (0.0, 0.0, 0.0), offset=(1.0, 0.0, 0.0))
        self.assertAlmostEqual(lat1, lat0, places=12)
        self.assertGreater(lon1, lon0)
        lat2, lon2, alt2 = gnss_reading(s, (0.0, 0.0, 0.0), offset=(0.0, 1.0, 0.5))
        self.assertGreater(lat2, lat0)
        self.assertAlmostEqual(alt2, alt0 + 0.5)
        same = gnss_reading(s, (0.0, 0.0, 0.0), offset=(0.0, 0.0, 0.0))
        self.assertEqual(same, (lat0, lon0, alt0))


class EgoConfigTest(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.yaml"
            p.write_text(f"rig: {RIG}\n" + text)
            return EgoConfig.load(p)

    def test_absent_is_the_default_receiver(self):
        cfg = self.load("")
        self.assertEqual(cfg.gnss, GNSS_NOISE_DEFAULTS)
        self.assertEqual(cfg.load_rig().sensors["gnss"].noise, GNSS_NOISE_DEFAULTS)

    def test_a_project_overrides_it(self):
        cfg = self.load("gnss: {horizontal_stddev_m: 0.05, seed: 9}\n")
        want = {**GNSS_NOISE_DEFAULTS, "horizontal_stddev_m": 0.05, "seed": 9}
        self.assertEqual(cfg.gnss, want)
        self.assertEqual(cfg.load_rig().sensors["gnss"].noise, want)

    def test_bad_keys_are_refused_with_the_file(self):
        with self.assertRaisesRegex(ValueError, r"e\.yaml: gnss: unknown key"):
            self.load("gnss: {noise_m: 0.02}\n")


# --- L1b (scensim issue 195): the receiver's heading and attitude (GnssInsOrientationStamped) ---

from types import SimpleNamespace  # noqa: E402

from aib import frames  # noqa: E402
from aib.sensing import gnss_ins_orientation  # noqa: E402

DEG = math.pi / 180


def errors(settings, n):
    noise = GnssNoise(gnss_noise_settings(settings))
    out = []
    for _ in range(n):
        noise.offset()
        out.append(noise.orientation_error())
    return np.array(out), noise


class InsSettingsTest(unittest.TestCase):
    def test_the_defaults_are_a_dual_antenna_receivers(self):
        s = gnss_noise_settings({})
        self.assertEqual((s["heading_stddev_deg"], s["attitude_stddev_deg"]), (0.2, 0.05))

    def test_bad_values_are_refused(self):
        for k in ("heading_stddev_deg", "attitude_stddev_deg"):
            with self.assertRaisesRegex(ValueError, f"{k} must be >= 0"):
                gnss_noise_settings({k: -1.0})


class InsNoiseTest(unittest.TestCase):
    def test_a_still_cars_errors_have_the_configured_spread(self):
        d, _ = errors({"seed": 11}, 20000)
        sd, mean = d.std(axis=0), d.mean(axis=0)
        for axis, sigma in ((0, 0.05 * DEG), (1, 0.05 * DEG), (2, 0.2 * DEG)):
            self.assertLess(abs(sd[axis] - sigma), 0.02 * sigma, (axis, sd))
            self.assertLess(abs(mean[axis]), 4 * sigma / math.sqrt(20000), (axis, mean))

    def test_the_same_seed_gives_the_same_headings(self):
        a, _ = errors({"seed": 5}, 30)
        b, _ = errors({"seed": 5}, 30)
        c, _ = errors({"seed": 6}, 30)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_disabled_is_the_exact_orientation(self):
        d, noise = errors({"enabled": False, "seed": 1}, 10)
        self.assertTrue(np.all(d == 0.0))
        self.assertEqual(noise.rmse_rotation(), (0.0, 0.0, 0.0))

    def test_the_rmse_fields_carry_the_stddevs_in_rad(self):
        r = GnssNoise(gnss_noise_settings({"seed": 0})).rmse_rotation()
        for got, want in zip(r, (0.05 * DEG, 0.05 * DEG, 0.2 * DEG)):
            self.assertAlmostEqual(got, want)

    def test_the_orientation_is_gnss_links_in_the_map_turned_by_the_error(self):
        q_base = frames.rpy_quat(0.0, 0.0, 0.7)
        q_mount = frames.rpy_quat(0.0, 0.0075 * 2, -0.0364)
        exact = gnss_ins_orientation(q_base, q_mount, (0.0, 0.0, 0.0))
        self.assertEqual(exact, frames.qmul(q_base, q_mount))
        level = frames.rpy_quat(0, 0, 0.7)
        turned = gnss_ins_orientation(level, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.01))
        want = frames.rpy_quat(0, 0, 0.71)
        self.assertTrue(all(abs(a - b) < 1e-12 for a, b in zip(turned, want)), (turned, want))


class _Msg(SimpleNamespace):
    pass


def _nav():
    return _Msg(header=_Msg(stamp=None, frame_id=""), status=_Msg(status=0, service=0),
                latitude=0.0, longitude=0.0, altitude=0.0, position_covariance=None,
                position_covariance_type=0)


def _ins():
    return _Msg(header=_Msg(stamp=None, frame_id=""),
                orientation=_Msg(orientation=_Msg(w=1.0, x=0.0, y=0.0, z=0.0),
                                 rmse_rotation_x=0.0, rmse_rotation_y=0.0, rmse_rotation_z=0.0))


class _Pub:
    def __init__(self):
        self.sent = []

    def publish(self, m):
        self.sent.append(m)


class GnssSensorTest(unittest.TestCase):
    """aib.isaac.sensors.Gnss with a stand-in for the ROS side: the INS message beside the fix."""

    def test_the_ins_message_has_the_fixs_stamp_and_frame(self):
        from aib.isaac.sensors import Gnss
        from aib.rig import Rig

        rig = Rig.load(RIG).sensors["gnss"]
        rig.noise = {"seed": 3}
        nav, ins = _Pub(), _Pub()
        ros = SimpleNamespace(m={"NavSatFix": _nav, "GnssInsOrientationStamped": _ins},
                              sensor_publisher=lambda s: (nav, ins))
        g = Gnss(rig, ros)
        state = State(position=(10.0, 2.0, 0.0), quat=frames.rpy_quat(0, 0, 0.3))
        for k in range(3):  # L1c: the first fix goes out on the third tick
            g.publish(state, 0.1 * k, f"STAMP{k}")
        self.assertEqual(len(nav.sent), 1)
        self.assertEqual(len(ins.sent), 3)
        fix, o = nav.sent[0], ins.sent[2]
        self.assertEqual(o.header.stamp, fix.header.stamp)
        self.assertEqual(o.header.frame_id, fix.header.frame_id)
        self.assertEqual(o.header.frame_id, "gnss_link")
        q = o.orientation.orientation
        exact = frames.qmul(state.quat, rig.quat)
        # within a few stddevs of gnss_link's exact orientation
        dq = frames.qmul(frames.qconj(exact), (q.w, q.x, q.y, q.z))
        angle = 2 * math.atan2(math.sqrt(dq[1] ** 2 + dq[2] ** 2 + dq[3] ** 2), abs(dq[0]))
        self.assertLess(angle, 6 * 0.2 * DEG)
        self.assertAlmostEqual(o.orientation.rmse_rotation_z, 0.2 * DEG)

    def test_disabled_publishes_the_exact_orientation(self):
        from aib.isaac.sensors import Gnss
        from aib.rig import Rig

        rig = Rig.load(RIG).sensors["gnss"]
        rig.noise = {"enabled": False}
        nav, ins = _Pub(), _Pub()
        ros = SimpleNamespace(m={"NavSatFix": _nav, "GnssInsOrientationStamped": _ins},
                              sensor_publisher=lambda s: (nav, ins))
        g = Gnss(rig, ros)
        state = State(position=(10.0, 2.0, 0.0), quat=frames.rpy_quat(0.01, -0.02, 0.3))
        g.publish(state, 0.0, "STAMP")
        q = ins.sent[0].orientation.orientation
        self.assertEqual((q.w, q.x, q.y, q.z), frames.qmul(state.quat, rig.quat))
        self.assertEqual(ins.sent[0].orientation.rmse_rotation_x, 0.0)


class InsStreamTest(unittest.TestCase):
    def test_the_heading_leaves_a_seeds_position_errors_as_they_were(self):
        """243: the orientation draws come from a second generator, so seed 0's position errors
        are those L1 produced."""
        plain = GnssNoise(gnss_noise_settings({"seed": 0, "bias_stddev_m": 0.1}))
        with_ins = GnssNoise(gnss_noise_settings({"seed": 0, "bias_stddev_m": 0.1}))
        a = [plain.offset() for _ in range(20)]
        b = []
        for _ in range(20):
            b.append(with_ins.offset())
            with_ins.orientation_error()
        self.assertEqual(a, b)
        # and they are L1's: numpy's default_rng(seed), the bias first, then each fix's draw
        rng = np.random.default_rng(0)
        bias = rng.normal(0.0, 0.1, 3)
        first = bias + rng.normal(0.0, np.array([0.02, 0.02, 0.04]))
        self.assertTrue(np.allclose(a[0], first, rtol=0, atol=1e-15))


# --- L1c (scensim issue 195): the orientation before the fix, the fix held for 2 GNSS periods ---

class _Seq:
    """Both publishers in one, recording the order the messages went out."""

    def __init__(self):
        self.sent = []

    def pub(self, name):
        seq = self

        class P:
            def publish(self, m):
                seq.sent.append((name, m))
        return P()


class GnssHoldTest(unittest.TestCase):
    def gnss(self, noise=None):
        from aib.isaac.sensors import Gnss
        from aib.rig import Rig

        rig = Rig.load(RIG).sensors["gnss"]
        rig.noise = noise or {"seed": 2}
        seq = _Seq()
        ros = SimpleNamespace(m={"NavSatFix": _nav, "GnssInsOrientationStamped": _ins},
                              sensor_publisher=lambda s: (seq.pub("fix"), seq.pub("ins")))
        return Gnss(rig, ros), seq, rig.period

    def test_the_first_fix_only_after_two_orientation_periods(self):
        g, seq, dt = self.gnss()
        state = State(position=(5.0, 1.0, 0.0), quat=frames.rpy_quat(0, 0, 0.2))
        for k in range(5):
            g.publish(state, k * dt, f"S{k}")
        names = [n for n, _ in seq.sent]
        self.assertEqual(names, ["ins", "ins", "ins", "fix", "ins", "fix", "ins", "fix"])
        first_ins = seq.sent[0][1].header.stamp
        first_fix = next(m for n, m in seq.sent if n == "fix").header.stamp
        self.assertEqual((first_ins, first_fix), ("S0", "S2"))  # 2 periods apart

    def test_within_a_frame_the_orientation_goes_first_with_the_same_stamp(self):
        g, seq, dt = self.gnss()
        state = State(position=(5.0, 1.0, 0.0), quat=frames.rpy_quat(0, 0, 0.2))
        for k in range(6):
            g.publish(state, k * dt, f"S{k}")
        for (n0, m0), (n1, m1) in zip(seq.sent, seq.sent[1:]):
            if n1 == "fix":
                self.assertEqual(n0, "ins")
                self.assertEqual(m0.header.stamp, m1.header.stamp)
                self.assertEqual(m0.header.frame_id, m1.header.frame_id)

    def test_a_teleport_does_not_hold_the_fix_again(self):
        g, seq, dt = self.gnss()
        here = State(position=(5.0, 1.0, 0.0), quat=frames.rpy_quat(0, 0, 0.2))
        there = State(position=(400.0, -3.0, 0.0), quat=frames.rpy_quat(0, 0, 3.0))
        for k in range(3):
            g.publish(here, k * dt, f"S{k}")
        before = sum(1 for n, _ in seq.sent if n == "fix")
        g.publish(there, 3 * dt, "S3")  # the scenario's reset moved the car
        self.assertEqual(sum(1 for n, _ in seq.sent if n == "fix"), before + 1)

    def test_the_hold_counts_published_orientations_not_calls(self):
        """A call between GNSS periods publishes nothing and does not shorten the hold."""
        g, seq, dt = self.gnss()
        state = State(position=(5.0, 1.0, 0.0))
        for t in (0.0, dt / 2, dt, 1.5 * dt, 2 * dt):
            g.publish(state, t, f"T{t:.3f}")
        self.assertEqual([n for n, _ in seq.sent], ["ins", "ins", "ins", "fix"])
