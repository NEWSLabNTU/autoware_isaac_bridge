"""The vehicle spec as aib reads it (sim_vehicle_spec's, deps/sim_vehicle_spec): its PhysX
vehicle (aib.gen.vehicle_params). The spec's own loading and validation are tested in
sim_vehicle_spec. Pure: no Isaac, no ROS."""
import unittest

import yaml

from _paths import SPEC
from aib import gen, powertrain
from aib.gen import specmod, vehicle_params
from aib.params import VehicleParams


def doc():
    return yaml.safe_load(SPEC.read_text())


class ShippedSpecTest(unittest.TestCase):
    def setUp(self):
        self.spec = specmod.load(SPEC)

    def test_it_is_csbs_model3_as_the_rwd(self):
        # csb 4aefee8's vehicle_params.py (the plausible real Model 3), with the RWD's mass,
        # weight split and drive unit
        p = vehicle_params(self.spec)
        self.assertEqual((p.wheel_base, p.wheel_tread, p.wheel_radius, p.wheel_width),
                         (3.005, 1.667, 0.370, 0.235))
        self.assertEqual(p.max_steer_angle, 0.70)
        self.assertEqual(p.mass, 1760.0)
        self.assertAlmostEqual(p.com[0], 0.47 * 3.005)
        self.assertEqual(p.com[1:], (0.0, 0.46))
        self.assertEqual(p.tire_friction, 1.0)
        self.assertEqual(p.max_brake_torque, 3000.0)
        self.assertEqual(p.brake_distribution, "dynamic_axle_load")
        self.assertEqual((p.suspension_frequency, p.suspension_damping_ratio), (1.6, 0.45))
        self.assertEqual(p.physx_substeps, 8)
        self.assertEqual(p.min_active_slip_denominator, 4.0)
        self.assertEqual(p.powertrain, "ev")
        self.assertEqual(p.friction_vs_slip, ((0.0, 1.0), (0.1, 1.0), (1.0, 0.8)))
        self.assertEqual((p.drive_axle, p.gear_ratio, p.motor_peak_torque, p.motor_peak_power),
                         ("rear", 10.0, 340.0, 194000.0))
        self.assertEqual((p.regen_fade_low, p.regen_fade_high), (1.0, 3.0))
        self.assertTrue(p.abs_enabled and p.tc_enabled)
        self.assertEqual((p.drag_coefficient, p.frontal_area), (0.23, 2.2))

    def test_the_library_defaults_are_the_spec(self):
        # VehicleParams() must not drift from the shipped spec it documents.
        p = vehicle_params(self.spec)
        d = VehicleParams()
        for k, v in p.to_dict().items():
            if k == "model":  # the body: the spec's, not a default
                continue
            ref = d.to_dict()[k]
            if isinstance(v, float):
                self.assertAlmostEqual(v, ref, msg=k)
            else:
                self.assertEqual(v, ref, msg=k)

    def test_physx_settings_come_from_the_isaac_section(self):
        d = doc()
        d["isaac"]["physx"]["substeps"] = 3
        p = vehicle_params(specmod.Spec(SPEC, d))
        self.assertEqual(p.physx_substeps, 3)

    def test_a_spec_without_physx_takes_the_defaults(self):
        d = doc()
        del d["isaac"]["physx"]
        p = vehicle_params(specmod.Spec(SPEC, d))
        self.assertEqual(p.physx_substeps, VehicleParams().physx_substeps)


class SpecVocabularyTest(unittest.TestCase):
    """What the spec may say, aib can build."""

    def test_every_powertrain_model_is_one_aib_has(self):
        self.assertEqual(set(specmod.MODELS), set(powertrain.MODELS))

    def test_every_drive_axle_is_one_aib_has(self):
        for axle in specmod.DRIVE_AXLES:
            d = doc()
            d["powertrain"]["drive_axle"] = axle
            specmod.validate(d)
            self.assertEqual(vehicle_params(specmod.Spec(SPEC, d)).drive_axle, axle)

    def test_it_reads_the_submodule(self):
        self.assertEqual(gen.SPECS, SPEC.parent)


class DerivedTest(unittest.TestCase):
    def test_an_ev_spec_is_an_ev(self):
        d = doc()
        d["powertrain"]["model"] = "ev"
        specmod.validate(d)
        p = vehicle_params(specmod.Spec(SPEC, d))
        self.assertEqual(p.powertrain, "ev")

    def test_no_resistance_section_is_no_road_load(self):
        d = doc()
        del d["resistance"]
        p = vehicle_params(specmod.Spec(SPEC, d))
        self.assertEqual((p.rolling_coefficient, p.drag_coefficient, p.frontal_area),
                         (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
