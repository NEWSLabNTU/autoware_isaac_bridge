"""Vehicle parameters, the powertrain seam and the steering: the maths the PhysX vehicle
applies every substep. Pure. Ported from swb's test_vehicle_params.py where it applies."""
import math
import unittest

_paths = __import__("_paths")  # sys.path, for direct runs
from aib import powertrain
from aib.control import (GEAR_DRIVE, GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE, Command,
                         ackermann, gear_from_command, steer)
from aib.params import (G, VehicleMismatch, VehicleParams, bicycle_limit, check_geometry)


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

    def test_braking_moves_load_and_brake_force_forward(self):
        p = VehicleParams()
        static = p.brake_split(0.0)
        self.assertAlmostEqual(static, p.com[0] / p.wheel_base)
        self.assertGreater(p.brake_split(8.0), static + 0.1)
        self.assertAlmostEqual(VehicleParams(brake_distribution="static").brake_split(8.0), static)

    def test_a_hard_stop_can_reach_the_tire_limit(self):
        p = VehicleParams()
        need = p.mass * G * p.tire_friction * p.wheel_radius
        self.assertLess(need * p.brake_split(G) / 2, p.max_brake_torque)

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


class IdealPowertrainTest(unittest.TestCase):
    def setUp(self):
        # No road load or wheel damping: the force is m a alone
        self.p = VehicleParams(powertrain="ideal", rolling_coefficient=0.0,
                               drag_coefficient=0.0, wheel_damping_rate=0.0)
        self.pt = powertrain.make(self.p)
        self.mass = self.p.mass + 2 * self.p.wheel_mass  # four wheels at 0.5 m r^2 / r^2

    def test_it_is_ideal(self):
        self.assertEqual(self.pt.name, "ideal")

    def test_standing_still_holds(self):
        drive, brakes = self.pt.torques(Command(), 0.0, 0.01)
        self.assertEqual(drive, (0.0,) * 4)
        self.assertEqual(brakes, (self.p.hold_brake_torque,) * 4)
        _, brakes = self.pt.torques(Command(acceleration=1.0, velocity=5.0, gear=GEAR_PARK), 3.0, 0.01)
        self.assertEqual(brakes, (self.p.hold_brake_torque,) * 4)

    def test_acceleration_is_m_a_over_four_wheels(self):
        drive, brakes = self.pt.torques(Command(acceleration=1.0, velocity=5.0), 2.0, 0.01)
        for d in drive:
            self.assertAlmostEqual(d, self.mass * 1.0 * self.p.wheel_radius / 4)
        self.assertEqual(brakes, (0.0,) * 4)

    def test_drive_is_capped_and_reverse_drives_backwards(self):
        drive, _ = self.pt.torques(Command(acceleration=50.0, velocity=5.0), 2.0, 0.01)
        self.assertEqual(drive, (self.p.max_drive_torque,) * 4)
        drive, _ = self.pt.torques(Command(acceleration=1.0, velocity=-2.0, gear=GEAR_REVERSE), -1.0, 0.01)
        self.assertLess(drive[0], 0.0)

    def test_deceleration_brakes_split_by_dynamic_load(self):
        drive, (fl, fr, rl, rr) = self.pt.torques(Command(acceleration=-3.0, velocity=5.0), 7.4, 0.01)
        self.assertEqual(drive, (0.0,) * 4)
        self.assertEqual((fl, rl), (fr, rr))
        total = self.mass * 3.0 * self.p.wheel_radius
        self.assertAlmostEqual(2 * (fl + rl), total)
        self.assertAlmostEqual(fl / (fl + rl), self.p.brake_split(3.0))
        self.assertGreater(fl, rl)

    def test_brakes_are_capped(self):
        _, brakes = self.pt.torques(Command(acceleration=-100.0, velocity=5.0), 10.0, 0.01)
        self.assertEqual(brakes[0], self.p.max_brake_torque)

    def test_neutral_coasts(self):
        self.assertEqual(self.pt.torques(Command(acceleration=1.0, velocity=3.0, gear=GEAR_NEUTRAL), 3.0, 0.01),
                         ((0.0,) * 4, (0.0,) * 4))

    def test_the_road_load_is_in_the_feedforward(self):
        # The physics applies the road load to the body; ideal pushes against it, so a
        # cruise at a = 0 still needs drive (and the request is still realized).
        p = VehicleParams(powertrain="ideal", wheel_damping_rate=0.0)
        drive, _ = powertrain.make(p).torques(Command(acceleration=0.0, velocity=20.0), 20.0, 0.01)
        load = -powertrain.resistance(p, 20.0)
        self.assertAlmostEqual(load, p.rolling_coefficient * p.mass * G
                               + 0.5 * 1.2 * 0.23 * 2.2 * 400.0)
        self.assertAlmostEqual(sum(drive), load * p.wheel_radius)

    def test_reserved_models_fail_loudly(self):
        for name in ("physx", "fmu"):
            with self.assertRaises(NotImplementedError):
                powertrain.make(VehicleParams(powertrain=name))
        with self.assertRaises(ValueError):
            powertrain.make(VehicleParams(powertrain="steam"))


class SteeringTest(unittest.TestCase):
    def test_clamped_to_the_limit(self):
        self.assertEqual(steer(0.0, 2.0, 0.01, 0.7), 0.7)
        self.assertEqual(steer(0.0, -2.0, 0.01, 0.7), -0.7)

    def test_lag_and_rate_limit(self):
        a = steer(0.0, 0.5, 0.1, 0.7, time_constant=0.1)
        self.assertAlmostEqual(a, 0.5 * (1 - math.exp(-1)))
        self.assertAlmostEqual(steer(0.0, 0.5, 0.1, 0.7, rate_limit=1.0), 0.1)

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
