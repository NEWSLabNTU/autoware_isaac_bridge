"""The rig and ego config as the Isaac side loads them, and the sensor kit launch against
the spec's topics. Pure."""
import re
import tempfile
import unittest
from pathlib import Path

from _paths import EGO, REPO, RIG
from aib.rig import EgoConfig, Rig

KIT = REPO / "src/aib_sensor_kit_launch/aib_sensor_kit_launch/launch"


class RigTest(unittest.TestCase):
    def test_the_shipped_rig(self):
        r = Rig.load(RIG)
        self.assertEqual(r.name, "tesla_model3")
        self.assertEqual(r.vehicle.mass, 1845.0)
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

    def test_the_shipped_ego_config(self):
        cfg = EgoConfig.load(EGO)
        self.assertEqual(Path(cfg.rig), RIG)
        self.assertEqual(cfg.domain, 1)
        self.assertEqual(cfg.clock_offset, "wall")
        self.assertEqual(cfg.vehicle, {})
        self.assertEqual(cfg.load_rig().name, "tesla_model3")

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
        self.assertEqual(re.findall(r'name="input_topic" value="([^"]+)"', text["imu.launch.xml"]), [rel])
        gnss = r.sensors["gnss"].topic
        rel = gnss[len("/sensing/gnss/"):]
        self.assertIn(f'<remap from="fix" to="{rel}"/>', text["gnss.launch.xml"])


if __name__ == "__main__":
    unittest.main()
