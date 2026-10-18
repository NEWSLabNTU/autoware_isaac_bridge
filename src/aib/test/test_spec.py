"""The vehicle spec: loading, validation, and what it derives. Pure: no Isaac, no ROS."""
import copy
import unittest

import yaml

from _paths import SPEC
from aib import spec as specmod
from aib.params import VehicleParams


def doc():
    return yaml.safe_load(SPEC.read_text())


class ShippedSpecTest(unittest.TestCase):
    def setUp(self):
        self.spec = specmod.load(SPEC)

    def test_it_is_csbs_model3(self):
        # csb 4aefee8's vehicle_params.py: the plausible real Model 3
        p = specmod.vehicle_params(self.spec)
        self.assertEqual((p.wheel_base, p.wheel_tread, p.wheel_radius, p.wheel_width),
                         (3.005, 1.667, 0.370, 0.235))
        self.assertEqual(p.max_steer_angle, 0.70)
        self.assertEqual(p.mass, 1845.0)
        self.assertAlmostEqual(p.com[0], 0.48 * 3.005)
        self.assertEqual(p.com[1:], (0.0, 0.46))
        self.assertEqual(p.tire_friction, 1.0)
        self.assertEqual(p.max_brake_torque, 3000.0)
        self.assertEqual(p.brake_distribution, "dynamic_axle_load")
        self.assertEqual((p.suspension_frequency, p.suspension_damping_ratio), (1.6, 0.45))
        self.assertEqual(p.physx_substeps, 8)
        self.assertEqual(p.min_active_slip_denominator, 4.0)
        self.assertEqual(p.powertrain, "ideal")

    def test_the_library_defaults_are_the_spec(self):
        # VehicleParams() must not drift from the shipped spec it documents.
        p = specmod.vehicle_params(self.spec)
        d = VehicleParams()
        for k, v in p.to_dict().items():
            if k == "model":  # the body: the spec's, not a default
                continue
            ref = d.to_dict()[k]
            if isinstance(v, float):
                self.assertAlmostEqual(v, ref, msg=k)
            else:
                self.assertEqual(v, ref, msg=k)

    def test_the_assumptions_are_marked(self):
        a = specmod.assumptions(self.spec)
        for key in ("inertia.com_front_fraction", "inertia.com_height", "tires"):
            self.assertIn(key, a)
        self.assertNotIn("inertia.mass", a)

    def test_the_box_is_vehicle_infos_footprint(self):
        (cx, cy, cz), (lx, ly, lz) = specmod.box(self.spec)
        self.assertAlmostEqual(lx, 4.792)
        self.assertAlmostEqual(ly, 2.163)
        self.assertAlmostEqual(lz, 1.488)
        self.assertAlmostEqual(cx, (3.005 + 0.923 - 0.864) / 2)
        self.assertEqual(cy, 0.0)

    def test_sensors_are_in_base_link(self):
        s = specmod.sensors(self.spec)
        (t, _), = [s["top"]["base_link"]]
        for a, b in zip(t, (0.9, 0.0, 2.0)):
            self.assertAlmostEqual(a, b)
        self.assertTrue(s["top"]["enabled"])
        self.assertFalse(s["camera6"]["enabled"])
        self.assertEqual(s["top"]["key"], "velodyne_top_base_link")

    def test_camera_intrinsics(self):
        k = specmod.camera_intrinsics(1280, 720, 90.0)
        self.assertAlmostEqual(k[0], 640.0)
        self.assertEqual((k[2], k[5]), (640.0, 360.0))


class ValidationTest(unittest.TestCase):
    def refused(self, d, text):
        with self.assertRaises(specmod.SpecError) as e:
            specmod.validate(d, "x.yaml")
        self.assertIn(text, str(e.exception))

    def test_the_shipped_one_is_valid(self):
        specmod.validate(doc())

    def test_unknown_keys_are_refused(self):
        d = doc()
        d["inertia"]["masss"] = 1.0
        self.refused(d, "inertia: unknown key(s) ['masss']")
        d = doc()
        d["colour"] = "red"
        self.refused(d, "unknown key(s) ['colour']")

    def test_missing_keys_are_refused(self):
        d = doc()
        del d["geometry"]["wheel_base"]
        self.refused(d, "geometry: missing ['wheel_base']")

    def test_a_powertrain_must_be_known(self):
        d = doc()
        d["powertrain"]["model"] = "steam"
        self.refused(d, "powertrain.model 'steam'")

    def test_a_mount_needs_all_six(self):
        d = doc()
        del d["sensors"]["imu"]["mount"]["yaw"]
        self.refused(d, "sensors.imu.mount")

    def test_sensor_keys_by_type(self):
        d = doc()
        d["sensors"]["imu"]["channels"] = 4
        self.refused(d, "unknown key(s) ['channels'] for a imu")
        d = doc()
        del d["sensors"]["camera6"]["width"]
        self.refused(d, "a camera needs width")

    def test_links_are_unique(self):
        d = doc()
        d["sensors"]["imu2"] = copy.deepcopy(d["sensors"]["imu"])
        self.refused(d, "used twice")

    def test_com_fraction_in_range(self):
        d = doc()
        d["inertia"]["com_front_fraction"] = 1.2
        self.refused(d, "com_front_fraction")

    def test_provenance_names_a_section(self):
        d = doc()
        d["provenance"]["nonsense.x"] = {"basis": "?"}
        self.refused(d, "provenance.nonsense.x")


if __name__ == "__main__":
    unittest.main()
