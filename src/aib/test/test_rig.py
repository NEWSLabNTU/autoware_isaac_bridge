"""The rig and ego config as the Isaac side loads them, and the sensor kit launch against
the spec's topics. Pure."""
import math
import re
import tempfile
import unittest
from pathlib import Path

from _paths import EGO, PKG, REPO, RIG
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

    def test_the_mustangs_zed_left(self):
        r = Rig.load(PKG / "config/drift_mustang.rig.yaml")
        z = r.sensors["zed_left"]
        self.assertTrue(z.enabled)
        self.assertEqual(z.topic, "/sensing/camera/zed/image_raw")
        self.assertEqual(z.extra["info_topic"], "/sensing/camera/zed/camera_info")
        self.assertEqual((z.extra["width"], z.extra["height"]), (960, 600))
        self.assertAlmostEqual(z.period, 1 / 30)
        self.assertEqual(z.extra["rate_hz"], 30.0)
        fx = 480 / math.tan(math.radians(55))
        for a, b in zip(z.extra["k"], [fx, 0, 480, 0, fx, 300, 0, 0, 1]):
            self.assertAlmostEqual(a, b, places=6)
        self.assertEqual([x.name for x in r.of_type("camera")], ["zed_left"])

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

    def test_a_camera_without_k_gets_its_pinhole(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.yaml"
            text = RIG.read_text()
            text = re.sub(r"\n    k: \[[^\]]*\]", "", text, count=1)
            p.write_text(text)  # its spec: is relative to deps/, as the original's
            r = Rig.load(p, None, {"camera6": {"auto_exposure": True}})
        c = r.sensors["camera6"]
        self.assertTrue(c.extra["auto_exposure"])
        fx = 640 / math.tan(math.radians(74.5 / 2))
        for a, b in zip(c.extra["k"], [fx, 0, 640, 0, fx, 360, 0, 0, 1]):
            self.assertAlmostEqual(a, b, places=6)

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
        self.assertEqual(cfg.control_topic, "/control/command/control_cmd")
        self.assertEqual(cfg.lockstep, {})  # off: the world does not wait for Autoware

    def test_the_drift_cars_ego_config_takes_the_drift_controllers_command(self):
        cfg = EgoConfig.load(EGO.parent / "ego_drift_mustang.yaml")
        self.assertEqual(cfg.load_rig().name, "drift_mustang")
        self.assertEqual(cfg.control_topic, "/control/command/drift/control_cmd")
        # the world waits for NDT's pose each frame (swb roadmap 057)
        self.assertEqual(cfg.lockstep["topic"], "/localization/pose_estimator/pose_with_covariance")

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


class GroundTruthConfigTest(unittest.TestCase):
    """The ego config's ground_truth (swb roadmap 063 e): off by default, defaults filled."""

    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.yaml"
            p.write_text(f"rig: {RIG}\n" + text)
            return EgoConfig.load(p)

    def test_off_by_default(self):
        self.assertEqual(EgoConfig.load(EGO).ground_truth, {})

    def test_on_fills_the_defaults(self):
        gt = self.load("ground_truth: {pose_estimate: true}\n").ground_truth
        self.assertEqual(gt["topic"], "/ground_truth/odom")
        self.assertTrue(gt["pose_estimate"])
        self.assertEqual(gt["covariance"], [0.01, 0.0003])
        gt = self.load("ground_truth: {topic: /gt}\n").ground_truth
        self.assertEqual(gt["topic"], "/gt")
        self.assertFalse(gt["pose_estimate"])

    def test_unknown_keys_are_refused(self):
        with self.assertRaisesRegex(ValueError, "ground_truth: unknown"):
            self.load("ground_truth: {frame: map}\n")

    def test_a_pose_estimate_and_a_lockstep_on_the_same_topic_are_refused(self):
        with self.assertRaisesRegex(ValueError, "would wait on itself"):
            self.load("ground_truth: {pose_estimate: true}\nlockstep: {timeout_s: 1.0}\n")
        # a lockstep on another topic is fine
        cfg = self.load("ground_truth: {pose_estimate: true}\nlockstep: {topic: /x}\n")
        self.assertTrue(cfg.ground_truth["pose_estimate"])


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
