"""Vehicle parameters, the road load and the steering linkage: the maths the PhysX vehicle
applies every substep. Pure. Ported from swb's test_vehicle_params.py where it applies. The
onboard computers' (powertrain, brakes, steering actuator) are tested in sim_vehicle_ecu."""
import math
import unittest

_paths = __import__("_paths")  # sys.path, for direct runs
from aib.control import GEAR_DRIVE, GEAR_PARK, ackermann, gear_from_command
from aib.params import (G, VehicleMismatch, VehicleParams, bicycle_limit, check_geometry,
                        road_load)


class ParamsTest(unittest.TestCase):
    def test_from_dict_overrides_win_and_unknown_ones_are_refused(self):
        p = VehicleParams.from_dict({"mass": 1800.0}, {"mass": 1900.0, "com": [1.45, 0.0, 0.55]})
        self.assertEqual(p.mass, 1900.0)
        self.assertEqual(p.com, (1.45, 0.0, 0.55))
        with self.assertRaises(ValueError):
            VehicleParams.from_dict({}, {"masss": 1.0})

    def test_round_trip(self):
        p = VehicleParams()
        self.assertEqual(VehicleParams.from_dict(p.to_dict()), p)

    def test_axle_loads_split_by_the_centre_of_mass(self):
        p = VehicleParams(mass=2000.0, wheel_base=3.0, com=(1.5, 0.0, 0.5))
        self.assertEqual(p.axle_loads(), (500.0, 500.0))
        front, rear = VehicleParams(mass=2000.0, wheel_base=3.0, com=(2.0, 0.0, 0.5)).axle_loads()
        self.assertAlmostEqual(2 * (front + rear), 2000.0)
        self.assertGreater(front, rear)

    def test_the_suspension_sits_level(self):
        p = VehicleParams()
        (front, rear), sag = p.suspension()
        # same frequency on each axle's sprung mass: the same sag, g / w^2
        for sprung, k, c in (front, rear):
            self.assertAlmostEqual(sprung * G / k, sag)
            self.assertAlmostEqual(c / (2 * math.sqrt(k * sprung)), 0.45)
        self.assertAlmostEqual(sag, G / (2 * math.pi * 1.6) ** 2)

    def test_yaw_inertia_is_a_sedans(self):
        # the box gives 3909 kg m^2 at 1760 kg, close to m a b = 3959
        self.assertAlmostEqual(VehicleParams().diagonal_inertia()[2], 3908.7, delta=1.0)

    def test_check_geometry_names_both_numbers(self):
        ours = {"wheel_base": 3.005, "wheel_tread": 1.667, "wheel_radius": 0.37}
        check_geometry("x", dict(ours, wheel_tread=1.667 + 0.049), ours)
        with self.assertRaises(VehicleMismatch) as e:
            check_geometry("vehicle.lincoln.mkz_2020", dict(ours, wheel_base=2.8604), ours)
        self.assertIn("3.005", str(e.exception))
        self.assertIn("2.860", str(e.exception))

    def test_bicycle_limit(self):
        b = bicycle_limit(math.radians(70), 3.0046, 1.6668)
        radius = 3.0046 / math.tan(b)
        self.assertAlmostEqual(math.degrees(math.atan(3.0046 / (radius - 1.6668 / 2))), 70.0)
        self.assertEqual(bicycle_limit(0.0, 3.0, 1.6), 0.0)


class RoadLoadTest(unittest.TestCase):
    def test_road_load(self):
        p = VehicleParams()
        self.assertEqual(road_load(p, 0.0), 0.0)
        roll = 0.0095 * 1760.0 * G
        self.assertAlmostEqual(road_load(p, 0.25), -(roll / 2 + 0.5 * 1.2 * 0.506 * 0.0625))
        self.assertAlmostEqual(road_load(p, -10.0), roll + 0.5 * 1.2 * 0.23 * 2.2 * 100)


class SteeringTest(unittest.TestCase):
    def test_ackermann_inner_wheel_turns_more(self):
        left, right = ackermann(0.3, 3.005, 1.667)
        self.assertGreater(left, 0.3)
        self.assertLess(right, 0.3)
        # both wheels about one turning centre
        r = 3.005 / math.tan(0.3)
        self.assertAlmostEqual(3.005 / math.tan(left), r - 1.667 / 2)
        self.assertEqual(ackermann(0.0, 3.0, 1.6), (0.0, 0.0))

    def test_gears(self):
        self.assertEqual(gear_from_command(GEAR_DRIVE), GEAR_DRIVE)
        self.assertEqual(gear_from_command(5), GEAR_DRIVE)  # DRIVE_4
        self.assertEqual(gear_from_command(GEAR_PARK), GEAR_PARK)
        self.assertIsNone(gear_from_command(0))


if __name__ == "__main__":
    unittest.main()
