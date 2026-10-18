"""The generator: outputs, headers, and the staleness check. Pure."""
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _paths import REPO, SPEC
from aib import gen
from aib.frames import kit_transform
from aib.params import read_vehicle_info
from aib.rig import Rig


class CheckTest(unittest.TestCase):
    def test_the_repository_is_up_to_date(self):
        # The CI check: fails when someone edits the spec without `just generate`, or edits
        # a generated file by hand.
        self.assertEqual(gen.stale(SPEC, REPO), [])

    def test_a_hand_edit_is_stale_and_generate_fixes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vehicles").mkdir()
            (root / "src").mkdir()
            spec = root / "vehicles" / SPEC.name
            shutil.copy(SPEC, spec)
            outs = gen.outputs(spec, root)
            self.assertEqual(sorted(gen.stale(spec, root)), sorted(outs))
            self.assertEqual(sorted(gen.write(spec, root)), sorted(outs))
            self.assertEqual(gen.stale(spec, root), [])
            info = root / "src/aib_vehicle_launch/aib_vehicle_description/config/vehicle_info.param.yaml"
            info.write_text(info.read_text().replace("3.005", "3.1"))
            self.assertEqual(gen.stale(spec, root), [str(info.relative_to(root))])
            self.assertEqual(gen.main([str(spec), "--root", str(root), "--check"]), 1)
            self.assertEqual(gen.main([str(spec), "--root", str(root)]), 0)
            self.assertEqual(gen.main([str(spec), "--root", str(root), "--check"]), 0)

    def test_every_output_says_where_it_came_from(self):
        for path, text in gen.outputs(SPEC, REPO).items():
            head = "\n".join(text.splitlines()[:3])
            self.assertIn("Generated from vehicles/tesla_model3.yaml", head, path)
            self.assertIn("do not edit", head, path)


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.out = gen.outputs(SPEC, REPO)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for p, text in self.out.items():
            f = self.root / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)

    def tearDown(self):
        self.tmp.cleanup()

    def test_vehicle_info_is_the_geometry(self):
        doc = yaml.safe_load(self.out[
            "src/aib_vehicle_launch/aib_vehicle_description/config/vehicle_info.param.yaml"])
        p = doc["/**"]["ros__parameters"]
        self.assertEqual(p["wheel_base"], 3.005)
        self.assertEqual(p["max_steer_angle"], 0.70)
        self.assertEqual(len(p), 10)
        info = read_vehicle_info(self.root / "src/aib_vehicle_launch/aib_vehicle_description/"
                                 "config/vehicle_info.param.yaml")
        self.assertEqual(info["wheel_tread"], 1.667)

    def test_the_calibration_places_the_lidar_where_csb_did(self):
        d = self.root / "src/aib_sensor_kit_launch/aib_sensor_kit_description/config"
        t, q = kit_transform(d / "sensors_calibration.yaml", d / "sensor_kit_calibration.yaml",
                             "velodyne_top_base_link")
        rig = Rig.load(self.root / "src/aib/config/tesla_model3.rig.yaml")
        for a, b in zip(t, rig.sensors["top"].translation):
            self.assertAlmostEqual(a, b, places=8)
        for a, b in zip(q, rig.sensors["top"].quat):
            self.assertAlmostEqual(a, b, places=8)

    def test_individual_params_is_the_description_calibration(self):
        a = "src/aib_sensor_kit_launch/aib_sensor_kit_description/config/"
        b = "src/aib_individual_params/config/default/aib_sensor_kit/"
        for name in ("sensors_calibration.yaml", "sensor_kit_calibration.yaml"):
            self.assertEqual(self.out[a + name], self.out[b + name])

    def test_the_xacro_names_every_sensor_link(self):
        x = self.out["src/aib_sensor_kit_launch/aib_sensor_kit_description/urdf/sensor_kit.xacro"]
        for link in ("top_base_link", '"top"', "camera6/camera_optical_link", "carla/imu_link",
                     "carla/gnss_link", "velodyne_top_base_link"):
            self.assertIn(link, x)


if __name__ == "__main__":
    unittest.main()
