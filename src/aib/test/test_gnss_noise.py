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
