"""The rig and ego config as the Isaac side loads them, and the sensor kit launch against
the spec's topics. Pure."""
import re
import tempfile
import unittest
from pathlib import Path

from _paths import EGO, REPO, RIG
from aib.rig import EgoConfig, Rig

KIT = REPO / "deps/autoware_sim_vehicle/cars/tesla_model3/tesla_model3_sensor_kit_launch/launch"


class RigTest(unittest.TestCase):
    def test_the_shipped_rig(self):
        r = Rig.load(RIG)
        self.assertEqual(r.name, "tesla_model3")
        self.assertEqual(r.vehicle.mass, 1760.0)
        self.assertEqual(r.ecu.motor_peak_power, 194000.0)  # the onboard computers', from the spec
        self.assertFalse(hasattr(r.vehicle, "motor_peak_power"))
        self.assertEqual([s.name for s in r.of_type("lidar")], ["top"])
        self.assertEqual(r.of_type("camera"), [])
        self.assertEqual(len(r.of_type("camera", enabled_only=False)), 1)
        self.assertAlmostEqual(r.sensors["gnss"].period, 0.1)
        self.assertEqual(r.sensors["imu"].period, 0.0)
        self.assertIn("tires", r.assumptions)

    def test_overrides(self):
        r = Rig.load(RIG, {"tire_friction": 0.7}, {"camera6": {"enabled": True},
                                                   "top": {"isaac": {"variant": "x"}}})
        self.assertEqual(r.vehicle.tire_friction, 0.7)
        self.assertTrue(r.sensors["camera6"].enabled)
        self.assertEqual(r.sensors["top"].isaac, {"config": "OS1", "variant": "x"})
        with self.assertRaises(ValueError):
            Rig.load(RIG, None, {"radar": {"enabled": True}})
        with self.assertRaises(ValueError):
            Rig.load(RIG, None, {"top": {"translation": [0, 0, 0]}})
        r = Rig.load(RIG, None, None, {"esc_enabled": True})
        self.assertTrue(r.ecu.esc_enabled)
        with self.assertRaises(ValueError):  # a calibration field is not a PhysX one
            Rig.load(RIG, {"esc_enabled": True})

    def test_a_stale_rig_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.yaml"
            p.write_text(RIG.read_text().replace("spec_hash: ", "spec_hash: 0", 1))
            with self.assertRaisesRegex(ValueError, "just generate"):
                Rig.load(p)

    def test_the_shipped_ego_config(self):
        cfg = EgoConfig.load(EGO)
        self.assertEqual(Path(cfg.rig), RIG)
        self.assertEqual(cfg.domain, 1)
        self.assertEqual(cfg.vehicle, {})
        self.assertEqual(cfg.ecu, {})
        self.assertEqual(cfg.load_rig().name, "tesla_model3")

    def test_clock_offset_is_refused_by_name(self):
        # Roadmap 051: stamps are sim time + BASE_NS; the key is gone, not ignored.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.yaml"
            p.write_text(f"rig: {RIG}\nclock_offset: wall\n")
            with self.assertRaisesRegex(ValueError, "clock_offset is gone"):
                EgoConfig.load(p)

    def test_an_ego_config_needs_a_rig_and_known_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.yaml"
            p.write_text("domain: 2\n")
            with self.assertRaises(ValueError):
                EgoConfig.load(p)
            p.write_text("rig: x.yaml\nlidar: {}\n")
            with self.assertRaises(ValueError):
                EgoConfig.load(p)


class SensorKitLaunchTest(unittest.TestCase):
    """Autoware's sensing launch consumes the topics the rig publishes."""

    def test_topics_match(self):
        r = Rig.load(RIG)
        text = {p.name: p.read_text() for p in KIT.iterdir()}
        lidar = r.sensors["top"].topic
        self.assertIn(f'("input", "{lidar}")', text["pointcloud_preprocessor.launch.py"])
        imu = r.sensors["imu"].topic
        self.assertTrue(imu.startswith("/sensing/imu/"))
        rel = imu[len("/sensing/imu/"):]
        # The topics are launch arguments (Autoware's sample kit names them imu_raw_name);
        # their defaults are the rig's, and every consumer reads the argument.
        self.assertEqual(re.findall(r'name="imu_raw_name" default="([^"]+)"', text["imu.launch.xml"]), [rel])
        self.assertEqual(re.findall(r'name="input_(?:topic|imu_raw)" value="([^"]+)"',
                                    text["imu.launch.xml"]), ["$(var imu_raw_name)"] * 2)
        gnss = r.sensors["gnss"].topic
        rel = gnss[len("/sensing/gnss/"):]
        self.assertEqual(re.findall(r'name="gnss_fix_name" default="([^"]+)"', text["gnss.launch.xml"]), [rel])
        self.assertIn('<remap from="fix" to="$(var gnss_fix_name)"/>', text["gnss.launch.xml"])


if __name__ == "__main__":
    unittest.main()
