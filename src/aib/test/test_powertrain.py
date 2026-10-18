"""The `ev` powertrain (Model 3 RWD drive unit) and the road loads: the envelope, regen
blending, the brake hydraulics' step response, ABS and traction control on synthetic slip,
the feedforward. Pure: no Isaac."""
import math
import unittest

_paths = __import__("_paths")  # sys.path, for direct runs
from aib import powertrain as pt
from aib import spec as specmod
from aib.control import GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE, Command
from aib.params import G, VehicleParams

DT = 1 / 60


def ev(**over):
    return pt.make(VehicleParams(powertrain="ev", **over))


def wheels(speed, p, slips=(0.0,) * 4):
    return [pt.WheelState(speed / p.wheel_radius * (1 + s), s, True) for s in slips]


def run(model, cmd, speed, seconds, slips=(0.0,) * 4):
    out = []
    for _ in range(round(seconds / DT)):
        out.append(model.torques(cmd, speed, DT, wheels(speed, model.p, slips)))
    return out


class EnvelopeTest(unittest.TestCase):
    def setUp(self):
        self.p = VehicleParams()

    def test_peak_torque_to_base_speed_then_constant_power(self):
        base = self.p.motor_peak_power / self.p.motor_peak_torque  # rad/s
        self.assertAlmostEqual(base * 60 / (2 * math.pi), 5449, delta=1)  # Tesla: 5400 rpm
        self.assertEqual(pt.motor_limit(self.p, 0.0), 340.0)
        self.assertEqual(pt.motor_limit(self.p, 0.9 * base), 340.0)
        for w in (1.2 * base, 2 * base, 2.5 * base):
            self.assertAlmostEqual(pt.motor_limit(self.p, w) * w, 194000.0)
        self.assertEqual(pt.motor_limit(self.p, -0.5 * base), 340.0)  # either direction

    def test_nothing_past_the_maximum_speed(self):
        w_max = self.p.motor_max_rpm * pt.RPM
        self.assertEqual(pt.motor_limit(self.p, w_max), 0.0)
        self.assertEqual(pt.motor_limit(self.p, 1.1 * w_max), 0.0)
        mid = 0.975 * w_max  # half way down the taper
        self.assertAlmostEqual(pt.motor_limit(self.p, mid), 0.5 * 194000.0 / mid)
        # the taper starts at Tesla's top speed, 201 km/h
        v = 0.95 * w_max / self.p.gear_ratio * self.p.wheel_radius * 3.6
        self.assertAlmostEqual(v, 201.0, delta=3.0)

    def test_force_at_the_road_is_the_real_cars(self):
        # 340 N m x 9.03 on the Model 3's 0.334 m wheel = the effective 10.0 on 0.370 m
        real = 340.0 * 9.03 / 0.334
        ours = 340.0 * self.p.gear_ratio / self.p.wheel_radius
        self.assertAlmostEqual(ours / real, 1.0, delta=0.01)

    def test_regen_limits_and_fade(self):
        p = self.p
        omega = lambda v: v / p.wheel_radius * p.gear_ratio  # noqa: E731
        self.assertEqual(pt.regen_limit(p, omega(10.0), 10.0), 130.0)  # torque-limited
        self.assertAlmostEqual(pt.regen_limit(p, omega(30.0), 30.0) * omega(30.0), 60000.0)
        self.assertEqual(pt.regen_limit(p, omega(1.0), 1.0), 0.0)
        self.assertAlmostEqual(pt.regen_limit(p, omega(2.0), 2.0), 65.0)
        # ~0.2 g at the wheels below the power cap
        decel = 130.0 * p.gear_ratio / p.wheel_radius / p.mass
        self.assertAlmostEqual(decel / G, 0.2, delta=0.01)


class FeedforwardTest(unittest.TestCase):
    def test_road_load(self):
        p = VehicleParams()
        self.assertEqual(pt.resistance(p, 0.0), 0.0)
        roll = 0.0095 * 1760.0 * G
        self.assertAlmostEqual(pt.resistance(p, 0.25), -(roll / 2 + 0.5 * 1.2 * 0.506 * 0.0625))
        self.assertAlmostEqual(pt.resistance(p, -10.0), roll + 0.5 * 1.2 * 0.23 * 2.2 * 100)
        self.assertAlmostEqual(pt.resistance(p, 27.78), -(roll + 0.5 * 1.2 * 0.506 * 27.78 ** 2))

    def test_feedforward_is_m_a_plus_the_loads(self):
        p = VehicleParams()
        f = pt.feedforward(p, 1.0, 20.0)
        damping = 4 * p.wheel_damping_rate * 20.0 / p.wheel_radius ** 2
        self.assertAlmostEqual(f, (p.mass + 2 * p.wheel_mass) - pt.resistance(p, 20.0) + damping)

    def test_a_steady_request_is_the_feedforward_through_the_gear(self):
        m = ev()
        p = m.p
        out = run(m, Command(acceleration=1.0, velocity=20.0), 10.0, 1.0)[-1]
        self.assertAlmostEqual(sum(out.drive), pt.feedforward(p, 1.0, 10.0) * p.wheel_radius,
                               delta=0.5)
        self.assertEqual(out.drive[:2], (0.0, 0.0))  # rear-wheel drive
        self.assertEqual(out.drive[2], out.drive[3])  # an open differential
        self.assertEqual(out.brake, (0.0,) * 4)

    def test_full_throttle_is_the_envelope(self):
        m = ev()
        out = run(m, Command(acceleration=10.0, velocity=40.0), 25.0, 1.0)[-1]
        omega = 25.0 / m.p.wheel_radius * m.p.gear_ratio
        self.assertAlmostEqual(sum(out.drive), pt.motor_limit(m.p, omega) * m.p.gear_ratio,
                               delta=1.0)

    def test_reverse_drives_backwards(self):
        out = run(ev(), Command(acceleration=1.0, velocity=-2.0, gear=GEAR_REVERSE), -1.0, 1.0)[-1]
        self.assertLess(out.drive[2], 0.0)

    def test_a_point_mass_reaches_100_kmh_in_the_published_time(self):
        # Ideal traction, the envelope and the road loads: Model 3 RWD 0-100 km/h is
        # 5.8-6.1 s published; the tire model in Isaac takes some of it away.
        m = ev()
        p = m.p
        mass = p.mass + 2 * p.wheel_mass
        v, t = 0.0, 0.0
        cmd = Command(acceleration=10.0, velocity=60.0)
        while v < 100 / 3.6 and t < 20:
            out = m.torques(cmd, v, DT, wheels(v, p))
            force = sum(out.drive) / p.wheel_radius + pt.resistance(p, v) \
                - 4 * p.wheel_damping_rate * v / p.wheel_radius ** 2
            v += force / mass * DT
            t += DT
        self.assertGreater(t, 5.2)
        self.assertLess(t, 6.3)


class BlendingTest(unittest.TestCase):
    def test_mild_braking_is_all_regen(self):
        m = ev()
        out = run(m, Command(acceleration=-1.0, velocity=0.0), 15.0, 1.0)[-1]
        self.assertEqual(out.brake, (0.0,) * 4)
        self.assertLess(out.drive[2], 0.0)  # the motor brakes the rear wheels
        regen = -sum(out.drive) / m.p.wheel_radius
        self.assertAlmostEqual(regen, -pt.feedforward(m.p, -1.0, 15.0), delta=1.0)

    def test_hard_braking_tops_regen_up_with_friction(self):
        m = ev()
        p = m.p
        out = run(m, Command(acceleration=-4.0, velocity=0.0), 15.0, 1.0)[-1]
        regen = -sum(out.drive) / p.wheel_radius
        omega = 15.0 / p.wheel_radius * p.gear_ratio
        self.assertAlmostEqual(regen, pt.regen_limit(p, omega, 15.0) * p.gear_ratio / p.wheel_radius,
                               delta=1.0)
        friction = sum(out.brake) / p.wheel_radius
        self.assertAlmostEqual(regen + friction, -pt.feedforward(p, -4.0, 15.0), delta=2.0)
        self.assertGreater(out.brake[0], 0.0)

    def test_the_total_follows_the_dynamic_axle_load(self):
        m = ev()
        p = m.p
        out = run(m, Command(acceleration=-6.0, velocity=0.0), 15.0, 1.0)[-1]
        front = sum(out.brake[:2])
        rear = sum(out.brake[2:]) - sum(out.drive)  # rear friction plus regen
        self.assertAlmostEqual(front / (front + rear), p.brake_split(6.0), delta=0.01)

    def test_regen_fades_out_and_friction_takes_over(self):
        m = ev()
        out = run(m, Command(acceleration=-1.0, velocity=0.0), 0.8, 1.0)[-1]
        self.assertEqual(out.drive, (0.0,) * 4)
        self.assertGreater(out.brake[0], 0.0)

    def test_split_friction(self):
        m = ev()
        p = m.p
        share = p.brake_split(1.0)
        # regen beyond the rear share comes off the front
        brake = 2000.0
        f, r = m.split_friction(brake, 1500.0, 1.0)
        self.assertEqual(r, 0.0)
        self.assertAlmostEqual(f * 2 / p.wheel_radius, brake - 1500.0)
        f, r = m.split_friction(brake, 0.0, 1.0)
        self.assertAlmostEqual(f / (f + r), share)


class LagTest(unittest.TestCase):
    def test_motor_torque_lag(self):
        m = ev()
        tau = m.p.motor_time_constant
        out = run(m, Command(acceleration=1.0, velocity=20.0), 10.0, 0.1)
        target = pt.feedforward(m.p, 1.0, 10.0) * m.p.wheel_radius
        for n, o in enumerate(out, 1):
            self.assertAlmostEqual(sum(o.drive) / target, 1 - math.exp(-n * DT / tau))
        self.assertGreater(sum(out[0].drive), 0.0)  # no dead time on the motor

    def test_brake_dead_time_then_first_order(self):
        m = ev(regen_max_torque=0.0)  # friction only
        p = m.p
        out = run(m, Command(acceleration=-3.0, velocity=0.0), 10.0, 1.0)
        final = out[-1].brake[0]
        t = [i * DT + DT for i in range(len(out))]
        first = next(ti for ti, o in zip(t, out) if o.brake[0] > 0)
        self.assertAlmostEqual(first, p.brake_dead_time + DT, delta=DT + 1e-9)
        at = p.brake_dead_time + p.brake_time_constant
        i = min(range(len(t)), key=lambda k: abs(t[k] - at))
        self.assertAlmostEqual(out[i].brake[0] / final, 1 - math.exp(-1), delta=0.15)
        self.assertAlmostEqual(out[round(0.4 / DT)].brake[0] / final, 1.0, delta=0.02)

    def test_dead_time(self):
        d = pt.DeadTime(0.1)
        outs = [d.step(1.0, 0.02) for _ in range(10)]
        self.assertEqual(outs[:5], [0.0] * 5)  # what went in at 0.02 s comes out at 0.12 s
        self.assertEqual(outs[5:], [1.0] * 5)
        self.assertEqual(pt.DeadTime(0.0).step(2.0, 0.02), 2.0)

    def test_hold_and_park(self):
        m = ev()
        out = run(m, Command(), 0.0, 1.0)[-1]
        self.assertAlmostEqual(out.brake[0], m.p.hold_brake_torque, delta=1.0)
        self.assertEqual(out.drive, (0.0,) * 4)
        m = ev()
        out = run(m, Command(acceleration=1.0, velocity=5.0, gear=GEAR_PARK), 0.0, 1.0)[-1]
        self.assertAlmostEqual(out.brake[0], m.p.hold_brake_torque, delta=1.0)

    def test_neutral_coasts(self):
        out = run(ev(), Command(acceleration=1.0, velocity=3.0, gear=GEAR_NEUTRAL), 3.0, 1.0)[-1]
        self.assertEqual(out, ((0.0,) * 4, (0.0,) * 4))

    def test_released_brakes_are_zero_not_nearly(self):
        # PhysX holds a slow wheel on any brake torque, however small
        m = ev()
        run(m, Command(), 0.0, 1.0)
        out = run(m, Command(acceleration=1.0, velocity=5.0), 0.0, 1.0)[-1]
        self.assertEqual(out.brake, (0.0,) * 4)

    def test_reset_forgets(self):
        m = ev()
        run(m, Command(acceleration=-3.0), 10.0, 1.0)
        m.reset()
        self.assertEqual((m.motor_torque, m.pressure), (0.0, [0.0, 0.0]))


class AbsTest(unittest.TestCase):
    def test_a_locking_wheel_is_released_and_reapplied(self):
        m = ev()
        cmd = Command(acceleration=-9.0, velocity=0.0)
        regen = sum(run(m, cmd, 20.0, 0.5)[-1].drive)  # the pressure builds
        self.assertLess(regen, 0.0)
        full = m.pressure[0]
        # the front left locks (slip -1): its pressure drops, the others keep theirs
        out = run(m, cmd, 20.0, 0.15, slips=(-1.0, 0.0, 0.0, 0.0))[-1]
        self.assertLess(out.brake[0], 0.1 * full)
        self.assertAlmostEqual(out.brake[1], m.pressure[0])
        self.assertGreaterEqual(m.pressure[0], full)
        # while ABS works, regen is handed to the friction brakes (through the motor's lag)
        self.assertLess(abs(sum(out.drive)), 0.15 * abs(regen))
        # between the thresholds it creeps back; within the reapply threshold it comes back
        held = m.abs[0].factor
        run(m, cmd, 20.0, 0.1, slips=(-0.2, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(m.abs[0].factor, held + 0.25 * m.p.abs_apply_rate * 0.1,
                               delta=0.01)
        out = run(m, cmd, 20.0, 0.5, slips=(-0.02, 0.0, 0.0, 0.0))[-1]
        self.assertAlmostEqual(out.brake[0], m.pressure[0])
        # regen stays with the friction brakes for the rest of the stop, and comes back after
        self.assertFalse(any(x.active for x in m.abs))
        self.assertLess(abs(sum(out.drive)), 1.0)
        run(m, Command(acceleration=0.5, velocity=20.0), 20.0, 0.1)
        out = run(m, cmd, 20.0, 0.5)[-1]
        self.assertLess(sum(out.drive), 0.0)

    def test_a_cycle_on_a_simple_wheel_stays_bounded(self):
        # A wheel against a friction-limited tire: brake torque past the tire's spins it
        # down, ABS must keep its slip near the thresholds instead of locking.
        m = ev(regen_max_torque=0.0)
        p = m.p
        v, r = 20.0, p.wheel_radius
        J = 0.5 * p.wheel_mass * r * r
        fz = p.mass * G / 4
        omega = v / r
        slips = []
        cmd = Command(acceleration=-12.0, velocity=0.0)
        for _ in range(round(1.5 / DT)):
            s = (omega * r - v) / max(v, 4.0)
            ws = [pt.WheelState(omega, s, True)] * 4
            out = m.torques(cmd, v, DT, ws)
            for _ in range(8):  # the PhysX substeps
                s = (omega * r - v) / max(v, 4.0)
                tire = max(-1.0, min(1.0, s * p.long_stiff_value * G / fz)) * fz
                omega += (-tire * r - out.brake[0]) / J * (DT / 8)
                omega = max(omega, 0.0)
            v -= 0.9 * G * DT
            slips.append(s)
        late = slips[round(0.5 / DT):]
        self.assertGreater(min(late), -0.5)  # never locked
        self.assertLess(max(late), 0.0)  # still braking

    def test_off_when_slow_or_disabled(self):
        m = ev()
        out = run(m, Command(acceleration=-9.0), 1.0, 0.6, slips=(-1.0,) * 4)[-1]
        self.assertGreater(out.brake[0], 0.5 * m.pressure[0])
        m = ev(abs_enabled=False)
        out = run(m, Command(acceleration=-9.0), 20.0, 0.6, slips=(-1.0,) * 4)[-1]
        self.assertAlmostEqual(out.brake[0], m.pressure[0])


class TractionControlTest(unittest.TestCase):
    def test_a_spinning_driven_wheel_cuts_motor_torque(self):
        m = ev()
        cmd = Command(acceleration=10.0, velocity=30.0)
        base = sum(run(m, cmd, 5.0, 0.5)[-1].drive)
        out = run(m, cmd, 5.0, 0.2, slips=(0.0, 0.0, 0.5, 0.0))[-1]
        self.assertLess(sum(out.drive), 0.5 * base)
        out = run(m, cmd, 5.0, 0.5)[-1]
        self.assertAlmostEqual(sum(out.drive), base, delta=1.0)

    def test_undriven_wheels_do_not_count(self):
        m = ev()
        cmd = Command(acceleration=10.0, velocity=30.0)
        base = sum(run(m, cmd, 5.0, 0.5)[-1].drive)
        out = run(m, cmd, 5.0, 0.2, slips=(0.5, 0.5, 0.0, 0.0))[-1]
        self.assertAlmostEqual(sum(out.drive), base, delta=1.0)

    def test_all_wheel_drive_splits(self):
        m = ev(drive_axle="all", front_drive_share=0.4)
        out = run(m, Command(acceleration=1.0, velocity=20.0), 10.0, 1.0)[-1]
        self.assertAlmostEqual(sum(out.drive[:2]) / sum(out.drive), 0.4)


class ShippedSpecEvTest(unittest.TestCase):
    def test_the_shipped_spec_builds_an_ev(self):
        p = specmod.vehicle_params(specmod.load(_paths.SPEC))
        p.powertrain = "ev"
        m = pt.make(p)
        self.assertEqual(m.name, "ev")
        self.assertEqual(m.driven, (False, False, True, True))

    def test_bad_drive_axle(self):
        with self.assertRaises(ValueError):
            ev(drive_axle="middle")


if __name__ == "__main__":
    unittest.main()
