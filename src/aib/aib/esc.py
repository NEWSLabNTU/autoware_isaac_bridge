"""Electronic stability control: one wheel's brake against a yaw rate the tires cannot hold.
Pure: no Isaac.

The usual design (Rajamani, *Vehicle Dynamics and Control*, ch. 8; van Zanten's Bosch ESP):
a target yaw rate from the bicycle model at the present speed and tire angle, capped by what
the ground's friction can carry (|r| <= mu g / v), against the measured one. Past a dead band:

- oversteer (turning more than the target, same sense): a yaw moment against the turn, from
  the outer front wheel's brake;
- understeer (turning less): a yaw moment into the turn, from the inner rear wheel's brake.

A sideslip past `esc_sideslip_limit` adds to the oversteer moment whatever the yaw rate says
(a car sliding wide at a modest yaw rate). While it works against oversteer the motor's torque
is cut, as an ESC asks the engine to. The moment becomes a brake torque on the one wheel
(F = M / (tread / 2), at the wheel radius), through a first-order lag for the hydraulics; the
powertrain adds it to the wheel's brake ahead of ABS, so ABS still keeps that wheel from
locking.

The friction is the host's: the friction it writes under each wheel (`Vehicle.friction_values`),
which a real ESC estimates from the lateral acceleration it measures. So this ESC knows the
ground; it is an idealised one.
"""
import math
from typing import NamedTuple

from .params import G

# Wheel order: aib's WHEELS, front left, front right, rear left, rear right.
FRONT_LEFT, FRONT_RIGHT, REAR_LEFT, REAR_RIGHT = range(4)


class EscOut(NamedTuple):
    brake: tuple  # extra brake torque per wheel, N m
    motor_factor: float  # on the motor's delivered torque, 0..1
    moment: float  # the yaw moment asked, N m (+: left)
    target: float  # the target yaw rate, rad/s


IDLE = EscOut((0.0,) * 4, 1.0, 0.0, 0.0)


class Esc:
    def __init__(self, params):
        self.p = params
        self.reset()

    def reset(self):
        self.brake = [0.0] * 4
        self.motor_factor = 1.0
        self.out = IDLE

    def target(self, speed, tire_angle, mu):
        """The bicycle model's yaw rate at `speed` and `tire_angle`, rad/s, capped at
        esc_target_margin * mu g / speed."""
        p = self.p
        v = max(abs(speed), 0.1)
        r = speed * math.tan(tire_angle) / (p.wheel_base * (1 + p.esc_understeer * v * v))
        cap = p.esc_target_margin * mu * G / v
        return max(-cap, min(cap, r))

    def step(self, speed, yaw_rate, lateral_speed, tire_angle, mu, dt):
        p = self.p
        if not p.esc_enabled or abs(speed) < p.esc_min_speed:
            return self._settle(0.0, None, 0.0, dt, False)
        r_ref = self.target(speed, tire_angle, mu)
        err = yaw_rate - r_ref
        band = p.esc_dead_band + p.esc_dead_band_share * abs(r_ref)
        moment = 0.0
        oversteer = False
        if abs(err) > band:
            excess = err - math.copysign(band, err)
            moment = -p.esc_gain * excess  # against the error
            # the same sense as the yaw rate and more of it: oversteer
            oversteer = yaw_rate * err > 0 and abs(yaw_rate) > abs(r_ref)
        beta = math.atan2(lateral_speed, max(abs(speed), 0.1))
        if abs(beta) > p.esc_sideslip_limit:
            # sliding wide: beta's sign is against the turn's in a spin (r > 0, beta < 0), so
            # a moment of beta's sign stops it
            moment += p.esc_sideslip_gain * (beta - math.copysign(p.esc_sideslip_limit, beta))
            oversteer = True
        if moment == 0.0:
            return self._settle(0.0, None, r_ref, dt, False)
        # A moment to the left (+) brakes a left wheel, to the right a right wheel; the front
        # one against oversteer, the rear one against understeer.
        if moment > 0:
            wheel = FRONT_LEFT if oversteer else REAR_LEFT
        else:
            wheel = FRONT_RIGHT if oversteer else REAR_RIGHT
        return self._settle(moment, wheel, r_ref, dt, oversteer)

    def _settle(self, moment, wheel, r_ref, dt, oversteer):
        p = self.p
        torque = 0.0
        if wheel is not None:
            force = abs(moment) / (p.wheel_tread / 2)
            torque = min(force * p.wheel_radius, p.esc_max_brake_torque)
        a = 1 - math.exp(-dt / p.brake_time_constant) if p.brake_time_constant > 0 else 1.0
        for i in range(4):
            target = torque if i == wheel else 0.0
            self.brake[i] += (target - self.brake[i]) * a
            if self.brake[i] < 1.0 and target == 0.0:
                self.brake[i] = 0.0
        if oversteer:
            self.motor_factor = max(0.0, self.motor_factor - p.esc_motor_cut_rate * dt)
        else:
            self.motor_factor = min(1.0, self.motor_factor + p.esc_motor_restore_rate * dt)
        self.out = EscOut(tuple(self.brake), self.motor_factor, moment, r_ref)
        return self.out
