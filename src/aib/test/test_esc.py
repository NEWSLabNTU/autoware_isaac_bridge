"""The stability control (`aib.esc`): its target, which wheel it brakes for which slide, the
sideslip limit, the motor cut, and the `ev` powertrain putting its brake through ABS. Pure: no
Isaac."""
import math
import unittest

_paths = __import__("_paths")  # sys.path, for direct runs
from aib import esc as esc_mod
from aib import powertrain as pt
from aib.control import Command
from aib.params import G, VehicleParams

DT = 1 / 60
FL, FR, RL, RR = range(4)


def params(**over):
    return VehicleParams(powertrain="ev", esc_enabled=True, **over)


def settle(esc, *args, seconds=0.5):
    out = None
    for _ in range(round(seconds / DT)):
        out = esc.step(*args, DT)
    return out


class TargetTest(unittest.TestCase):
    def test_the_bicycle_model_within_the_friction(self):
        e = esc_mod.Esc(params())
        p = e.p
        v, d = 10.0, math.radians(2.0)
        self.assertAlmostEqual(e.target(v, d, 1.0), v * math.tan(d) / p.wheel_base)
        # on mud the same steering asks more than 0.85 mu g / v allows
        r = e.target(v, math.radians(10.0), 0.23)
        self.assertAlmostEqual(r, 0.85 * 0.23 * G / v)
        self.assertAlmostEqual(e.target(v, -math.radians(10.0), 0.23), -r)


class InterventionTest(unittest.TestCase):
    def test_off_unless_enabled_or_below_its_speed(self):
        off = esc_mod.Esc(VehicleParams(powertrain="ev"))
        out = settle(off, 20.0, 0.8, -3.0, 0.0, 1.0)
        self.assertEqual(out.brake, (0.0,) * 4)
        slow = esc_mod.Esc(params())
        out = settle(slow, 2.0, 0.8, -1.0, 0.0, 1.0)
        self.assertEqual(out.brake, (0.0,) * 4)

    def test_nothing_inside_the_dead_band(self):
        e = esc_mod.Esc(params())
        d = math.radians(2.0)
        r = e.target(20.0, d, 1.0)
        out = settle(e, 20.0, r + 0.02, 0.0, d, 1.0)
        self.assertEqual(out.brake, (0.0,) * 4)
        self.assertEqual(out.motor_factor, 1.0)

    def test_oversteer_to_the_left_brakes_the_outer_front(self):
        e = esc_mod.Esc(params())
        d = math.radians(2.0)
        r = e.target(20.0, d, 1.0)
        out = settle(e, 20.0, r + 0.3, 0.0, d, 1.0)
        self.assertGreater(out.brake[FR], 500.0)
        self.assertEqual([b for i, b in enumerate(out.brake) if i != FR], [0.0] * 3)
        self.assertLess(out.moment, 0.0)
        self.assertLess(out.motor_factor, 0.1)  # the motor cut against oversteer

    def test_oversteer_to_the_right_brakes_the_left_front(self):
        e = esc_mod.Esc(params())
        d = -math.radians(2.0)
        r = e.target(20.0, d, 1.0)
        out = settle(e, 20.0, r - 0.3, 0.0, d, 1.0)
        self.assertGreater(out.brake[FL], 500.0)
        self.assertGreater(out.moment, 0.0)

    def test_understeer_brakes_the_inner_rear_and_keeps_the_motor(self):
        e = esc_mod.Esc(params())
        d = math.radians(2.0)
        r = e.target(20.0, d, 1.0)
        out = settle(e, 20.0, 0.3 * r, 0.0, d, 1.0)
        self.assertGreater(out.brake[RL], 0.0)
        self.assertEqual([b for i, b in enumerate(out.brake) if i != RL], [0.0] * 3)
        self.assertGreater(out.moment, 0.0)
        self.assertEqual(out.motor_factor, 1.0)

    def test_a_wide_slide_is_oversteer_whatever_the_yaw_rate(self):
        e = esc_mod.Esc(params())
        # turning left at the target's yaw rate, the tail out: velocity to the right of the nose
        d = math.radians(3.0)
        r = e.target(10.0, d, 0.23)
        out = settle(e, 10.0, r, -10.0 * math.tan(math.radians(10.0)), d, 0.23)
        self.assertGreater(out.brake[FR], 0.0)
        self.assertLess(out.moment, 0.0)

    def test_the_brake_is_capped_and_lags(self):
        e = esc_mod.Esc(params())
        first = e.step(20.0, 2.0, 0.0, 0.0, 1.0, DT)
        self.assertLess(first.brake[FR], e.p.esc_max_brake_torque)
        out = settle(e, 20.0, 2.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(out.brake[FR], e.p.esc_max_brake_torque, delta=1.0)
        # and lets go once the slide is over
        out = settle(e, 20.0, 0.0, 0.0, 0.0, 1.0)
        self.assertEqual(out.brake, (0.0,) * 4)


class PowertrainTest(unittest.TestCase):
    def wheels(self, p, speed, slips=(0.0,) * 4):
        return [pt.WheelState(speed / p.wheel_radius * (1 + s), s, True) for s in slips]

    def test_the_brake_goes_on_its_wheel_and_the_motor_is_cut(self):
        p = params()
        ev = pt.make(p)
        cmd = Command(acceleration=1.0, velocity=20.0, received=True)
        req = esc_mod.EscOut((0.0, 800.0, 0.0, 0.0), 0.0, -1000.0, 0.0)
        out = None
        for _ in range(30):
            out = ev.torques(cmd, 20.0, DT, self.wheels(p, 20.0), req)
        self.assertEqual(out.brake, (0.0, 800.0, 0.0, 0.0))
        self.assertEqual(out.drive, (0.0,) * 4)

    def test_abs_still_keeps_that_wheel_from_locking(self):
        p = params()
        ev = pt.make(p)
        cmd = Command(acceleration=0.0, velocity=20.0, received=True)
        req = esc_mod.EscOut((0.0, 1500.0, 0.0, 0.0), 1.0, -1000.0, 0.0)
        out = None
        for _ in range(30):
            out = ev.torques(cmd, 20.0, DT, self.wheels(p, 20.0, (0.0, -0.4, 0.0, 0.0)), req)
        self.assertLess(out.brake[FR], 1500.0)

    def test_without_a_request_nothing_changes(self):
        p = params()
        a, b = pt.make(p), pt.make(p)
        cmd = Command(acceleration=-2.0, velocity=0.0, received=True)
        for _ in range(30):
            x = a.torques(cmd, 15.0, DT, self.wheels(p, 15.0))
            y = b.torques(cmd, 15.0, DT, self.wheels(p, 15.0), esc_mod.IDLE)
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
