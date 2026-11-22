"""Vehicle parameters, the road load and the steering linkage: the maths the PhysX vehicle
applies every substep. Pure. Ported from scensim's test_vehicle_params.py where it applies. The
onboard computers' (powertrain, brakes, steering actuator) are tested in sim_vehicle_ecu."""
import math
import unittest

_paths = __import__("_paths")  # sys.path, for direct runs
from aib.control import (GEAR_DRIVE, GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE, ackermann,
                         driven_wheels, gear_from_command, release_sticky)
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


class ReleaseStickyTest(unittest.TestCase):
    """Issue 070: a car in gear asked to move, its brakes released, gets the release torque on
    its driven wheels, so PhysX's sticky tyres let it roll; anything else passes unchanged."""
    REAR = driven_wheels("ev", "rear")
    ZERO, FLOOR = (0.0,) * 4, 50.0

    def test_the_driven_wheels(self):
        self.assertEqual(self.REAR, (False, False, True, True))
        self.assertEqual(driven_wheels("ev", "front"), (True, True, False, False))
        self.assertEqual(driven_wheels("ev", "all"), (True,) * 4)
        self.assertEqual(driven_wheels("ideal", "rear"), (True,) * 4)

    def test_released_brakes_and_a_target_get_the_release_torque(self):
        # Autoware's downhill start: -0.028 m/s^2 is ~4.7 N m a wheel, below the floor
        drive, brake = release_sticky(self.ZERO, (4.7,) * 4, 0.25, GEAR_DRIVE, self.FLOOR, 1.0,
                                      self.REAR, speed=0.0, standing=0.2)
        self.assertEqual(drive, (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(brake, self.ZERO)
        drive, _ = release_sticky(self.ZERO, self.ZERO, -0.5, GEAR_REVERSE, self.FLOOR, 1.0,
                                  self.REAR, speed=0.0, standing=0.2)
        self.assertEqual(drive, (0.0, 0.0, -1.0, -1.0))

    def test_anything_else_passes_unchanged(self):
        cases = [
            (self.ZERO, (1500.0,) * 4, 0.0, GEAR_DRIVE),     # target 0: the hold brake
            (self.ZERO, (4.7,) * 4, 0.005, GEAR_DRIVE),      # a target below MOVE_VELOCITY
            (self.ZERO, (60.0,) * 4, 0.25, GEAR_DRIVE),      # a brake past the floor
            ((0.0, 0.0, 30.0, 30.0), self.ZERO, 0.25, GEAR_DRIVE),  # already driving
            (self.ZERO, self.ZERO, 0.25, GEAR_NEUTRAL),
            (self.ZERO, self.ZERO, 0.25, GEAR_PARK),
        ]
        for drive, brake, v, gear in cases:
            self.assertEqual(release_sticky(drive, brake, v, gear, self.FLOOR, 1.0, self.REAR,
                                            speed=0.0, standing=0.2),
                             (drive, brake), (drive, brake, v, gear))
        # release_torque 0 turns it off
        self.assertEqual(release_sticky(self.ZERO, self.ZERO, 0.25, GEAR_DRIVE, self.FLOOR, 0.0,
                                        self.REAR, speed=0.0, standing=0.2),
                         (self.ZERO, self.ZERO))

    def test_a_moving_car_gets_no_release_torque(self):
        """Issue 103: traction control cut the drive to 0 at 0.9 m/s; the release torque then
        spun the driven wheels, TC read that as slip and held the drive at 0 for good."""
        for speed in (0.21, 0.87, -0.5, 10.0):
            self.assertEqual(release_sticky(self.ZERO, self.ZERO, 0.25, GEAR_DRIVE, self.FLOOR, 1.0,
                                            self.REAR, speed=speed, standing=0.2),
                             (self.ZERO, self.ZERO), speed)
        # standing, it still releases
        drive, _ = release_sticky(self.ZERO, self.ZERO, 0.25, GEAR_DRIVE, self.FLOOR, 1.0,
                                  self.REAR, speed=0.15, standing=0.2)
        self.assertEqual(drive, (0.0, 0.0, 1.0, 1.0))

