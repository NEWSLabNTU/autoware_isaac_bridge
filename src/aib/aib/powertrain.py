"""The powertrain seam: how Autoware's longitudinal command becomes wheel torques. Pure.

A powertrain turns the command (acceleration, velocity, gear), the vehicle's speed and, when
the host has them, the wheels' states (spin, longitudinal slip) into a drive torque and a
brake torque for every wheel. The PhysX vehicle runs with no engine model of its own ("drive
none") and applies what the powertrain returns every physics step (`Vehicle.apply`, 60 Hz;
PhysX substeps the vehicle further inside each step).

Models (`powertrain.model` in the spec):

- `ideal`: acb's contract on CARLA. The requested acceleration is realized as the force
  `m a + F_resist(v)`, split evenly over the four wheels as drive torque, capped per wheel; a
  negative force is braking (never the motor), split between the axles by the dynamic axle
  load. A standing car with no request, or in park, holds on its hold brake. No lag, no motor
  map: the car does what is asked within its limits.
- `ev`: an electric drive unit modelled on a real one (the Tesla Model 3 RWD's, in the
  shipped spec). The same feedforward `m a + F_resist(v)` becomes a motor torque request
  through the gear reduction; the motor delivers it through its torque/power envelope (peak
  torque to base speed, then constant power, nothing past its maximum speed) and a
  first-order torque lag. Deceleration is regenerative first (the motor as a generator,
  limited by its regen torque and power cap, fading out at low speed); the friction brakes
  cover the rest, through a hydraulic dead time and a first-order pressure lag, split by the
  dynamic axle load. ABS releases and reapplies each wheel's brake on its longitudinal slip;
  traction control cuts motor torque when a driven wheel spins. No hidden speed loop: what
  the model cannot deliver, Autoware's longitudinal controller sees and corrects, as on a
  real car.
- `physx` (reserved): PhysX's own DriveStandard (engine, gears, clutch, differential), with
  the command mapped to throttle and brake through accel/brake maps as a real vehicle
  interface would.
- `fmu` (reserved): an FMI 2.0 co-simulation unit (a supplier's powertrain model), stepped
  each substep.

The reserved models raise NotImplementedError from `make`, so a spec naming them fails loudly
at load instead of driving a different car.

Road loads (rolling resistance, aerodynamic drag: `resistance(params, v)`) are not part of
any powertrain: the host applies them to the body as a force (`Vehicle.apply`), so they act
through the physics whichever model drives. Both models' feedforward includes the same
estimate, plus PhysX's wheel damping, so `ideal` still realizes the request and `ev` starts
from the right torque.
"""
import math
from collections import deque
from typing import NamedTuple

from .control import GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE
from .params import G

MODELS = ("ideal", "physx", "ev", "fmu")
DRIVE_AXLES = ("rear", "front", "all")
# The order of every per-wheel tuple: aib.isaac.vehicle.WHEELS
IS_FRONT = (True, True, False, False)
RPM = 2 * math.pi / 60  # rad/s per rpm
# Below this speed rolling resistance ramps to zero, so a standing car is not pushed about.
ROLLING_RAMP = 0.5  # m/s
# A releasing brake below this is released: PhysX keeps a slow wheel in its sticky-tire hold
# while any brake torque at all is applied, so a pressure decaying exponentially towards 0
# held the car for seconds (seen in Isaac: 6 s at full drive torque with 1e-10 N m left).
# 50 N m is ~0.03 m/s^2 of braking per wheel; with the lags, the hold releases in ~0.3 s.
BRAKE_FLOOR = 50.0  # N m


class Torques(NamedTuple):
    """Per wheel, in aib.isaac.vehicle.WHEELS order (FL, FR, RL, RR), N m. `drive` is signed
    along base_link x; `brake` is a magnitude (PhysX opposes the wheel's spin with it)."""

    drive: tuple
    brake: tuple


class WheelState(NamedTuple):
    """What PhysX reports for one wheel after a step."""

    rotation_speed: float = 0.0  # rad/s, positive rolling forward
    slip: float = 0.0  # longitudinal slip: + the wheel spins faster than the road, - slower
    on_ground: bool = True


def resistance(p, v):
    """The road load at a speed `v` (m/s along x), N, signed against `v`: rolling resistance
    (Crr m g, ramped in below ROLLING_RAMP) and aerodynamic drag (rho Cd A v^2 / 2). What the
    host applies to the body."""
    roll = p.rolling_coefficient * p.mass * G * min(1.0, abs(v) / ROLLING_RAMP)
    drag = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * v * v
    return -math.copysign(roll + drag, v) if v else 0.0


def feedforward(p, accel, speed_along):
    """The tractive force (N, along the gear's direction) that gives `accel` at
    `speed_along`: the car and its spinning wheels times the acceleration, plus the road load
    and PhysX's wheel damping it must overcome."""
    mass = p.mass + 4 * 0.5 * p.wheel_mass  # four wheels at 0.5 m r^2: m_w / 2 each, reflected
    damping = 4 * p.wheel_damping_rate * speed_along / (p.wheel_radius ** 2)
    return mass * accel - resistance(p, speed_along) + damping


class Powertrain:
    """The interface: `torques(cmd, speed, dt, wheels=None) -> Torques`. `speed` is the
    body's forward speed (m/s), `wheels` the four WheelStates of the last step when the host
    has them (None: none yet, or a host that cannot read them)."""

    name = ""

    def __init__(self, params):
        self.p = params

    def torques(self, cmd, speed, dt, wheels=None):  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self):
        """A new episode: forget any internal state (none for `ideal`)."""


def _holding(cmd, speed):
    return (cmd.gear == GEAR_PARK
            or (abs(cmd.velocity) < 0.01 and cmd.acceleration <= 0.01 and abs(speed) < 0.1))


def _direction(cmd):
    return -1.0 if cmd.gear == GEAR_REVERSE else 1.0


class IdealPowertrain(Powertrain):
    name = "ideal"

    def torques(self, cmd, speed, dt, wheels=None):
        p = self.p
        r = p.wheel_radius
        if _holding(cmd, speed):
            return Torques((0.0,) * 4, (p.hold_brake_torque,) * 4)
        if cmd.gear == GEAR_NEUTRAL:
            return Torques((0.0,) * 4, (0.0,) * 4)
        direction = _direction(cmd)
        # Along the gear's direction: positive pushes that way, negative slows it down.
        push = feedforward(p, cmd.acceleration, direction * speed)
        if push >= 0:
            return Torques((direction * min(p.max_drive_torque, push * r / 4),) * 4, (0.0,) * 4)
        # Brakes, never the motor: decelerating to a stop does not drive it backwards. Split
        # by the dynamic axle loads, so a hard stop reaches the tires' limit on all four
        # together instead of locking the unloaded rear first.
        front = p.brake_split(-cmd.acceleration)
        total = -push * r
        f = min(total * front / 2, p.max_brake_torque)
        b = min(total * (1 - front) / 2, p.max_brake_torque)
        return Torques((0.0,) * 4, (f, f, b, b))


# --- ev -------------------------------------------------------------------------------------

def motor_limit(p, omega):
    """The motor's torque envelope at `omega` (rad/s at the motor shaft, either sign), N m:
    peak torque up to base speed (peak power / peak torque), then peak power / omega, tapering
    linearly to 0 over the last 5 % below the maximum speed, 0 past it."""
    w = abs(omega)
    w_max = p.motor_max_rpm * RPM
    if w >= w_max:
        return 0.0
    t = p.motor_peak_torque
    if w > 0:
        t = min(t, p.motor_peak_power / w)
    taper_from = 0.95 * w_max
    if w > taper_from:
        t *= (w_max - w) / (w_max - taper_from)
    return t


def regen_limit(p, omega, speed):
    """The most braking torque the motor may regenerate at `omega` (rad/s, motor) and road
    `speed` (m/s), N m: its regen torque limit and power cap, within the motor's envelope,
    faded linearly to 0 from `regen_fade_high` down to `regen_fade_low` m/s."""
    w = abs(omega)
    t = min(p.regen_max_torque, motor_limit(p, omega))
    if w > 0:
        t = min(t, p.regen_max_power / w)
    v = abs(speed)
    lo, hi = p.regen_fade_low, p.regen_fade_high
    if v <= lo:
        return 0.0
    if v < hi:
        t *= (v - lo) / (hi - lo)
    return t


def lag(current, target, dt, time_constant):
    """One step of a first-order lag (exact for a target held over `dt`)."""
    if time_constant <= 0:
        return target
    return current + (target - current) * (1 - math.exp(-dt / time_constant))


class DeadTime:
    """A pure delay: `step(value, dt)` returns what went in `delay` s earlier (`initial`
    before that), at the steps' own times."""

    def __init__(self, delay, initial=0.0):
        self.delay = delay
        self.initial = initial
        self.reset()

    def reset(self):
        self.t = 0.0
        self.queue = deque()
        self.out = self.initial

    def step(self, value, dt):
        self.t += dt
        if self.delay <= 0:
            self.out = value
            return value
        self.queue.append((self.t, value))
        while self.queue and self.queue[0][0] <= self.t - self.delay + 1e-9:
            self.out = self.queue.popleft()[1]
        return self.out


class SlipModulator:
    """Slip-based release and reapply of one actuator, as ABS does a wheel's brake pressure
    and traction control the motor's torque: a factor in [0, 1] that falls at `release_rate`
    (1/s) while the slip is past `release`, climbs back at `apply_rate` once it is within
    `reapply`, and creeps back at a quarter of that in between (a pure hold there can park
    the factor below 1 for good while the tire works at its peak). The slip is a magnitude in
    the direction limited."""

    def __init__(self, release, reapply, release_rate, apply_rate):
        self.release, self.reapply = release, reapply
        self.release_rate, self.apply_rate = release_rate, apply_rate
        self.factor = 1.0

    @property
    def active(self):
        return self.factor < 1.0

    def step(self, slip, dt, enabled=True):
        if not enabled:
            self.factor = min(1.0, self.factor + self.apply_rate * dt)
        elif slip > self.release:
            self.factor = max(0.0, self.factor - self.release_rate * dt)
        elif slip < self.reapply:
            self.factor = min(1.0, self.factor + self.apply_rate * dt)
        else:
            self.factor = min(1.0, self.factor + 0.25 * self.apply_rate * dt)
        return self.factor

    def reset(self):
        self.factor = 1.0


class EvPowertrain(Powertrain):
    name = "ev"

    def __init__(self, params):
        super().__init__(params)
        p = params
        if p.drive_axle not in DRIVE_AXLES:
            raise ValueError(f"drive_axle {p.drive_axle!r}: one of {', '.join(DRIVE_AXLES)}")
        for k in ("gear_ratio", "motor_peak_torque", "motor_peak_power", "motor_max_rpm"):
            if not getattr(p, k) > 0:
                raise ValueError(f"ev powertrain: {k} must be positive")
        self.front_share = {"rear": 0.0, "front": 1.0, "all": p.front_drive_share}[p.drive_axle]
        s = self.front_share
        # Each wheel's share of the axle torque: an open differential halves it.
        self.drive_share = (s / 2, s / 2, (1 - s) / 2, (1 - s) / 2)
        self.driven = tuple(x > 0 for x in self.drive_share)
        self.brake_delay = (DeadTime(p.brake_dead_time), DeadTime(p.brake_dead_time))
        self.abs = [SlipModulator(p.abs_slip_release, p.abs_slip_reapply,
                                  p.abs_release_rate, p.abs_apply_rate) for _ in range(4)]
        self.tc = SlipModulator(p.tc_slip_limit, p.tc_slip_reapply,
                                p.tc_release_rate, p.tc_apply_rate)
        self.reset()

    def reset(self):
        self.motor_torque = 0.0  # N m at the motor shaft, along the gear's direction
        self.pressure = [0.0, 0.0]  # friction brake torque per wheel, front and rear axle
        for d in self.brake_delay:
            d.reset()
        for m in self.abs:
            m.reset()
        self.tc.reset()
        self.abs_event = False  # ABS has worked a wheel in this stop: no regen until it ends
        self.debug = {}

    def motor_speed(self, speed, wheels):
        """Motor shaft speed, rad/s: the driven wheels' mean spin (the body's speed when the
        wheels are not known) through the reduction."""
        if wheels:
            spins = [w.rotation_speed for w, d in zip(wheels, self.driven) if d]
            wheel = sum(spins) / len(spins)
        else:
            wheel = speed / self.p.wheel_radius
        return wheel * self.p.gear_ratio

    def torques(self, cmd, speed, dt, wheels=None):
        p = self.p
        r, ratio = p.wheel_radius, p.gear_ratio
        direction = _direction(cmd)
        omega = self.motor_speed(speed, wheels)
        hold = _holding(cmd, speed)
        regen_force = 0.0
        friction = (0.0, 0.0)  # per wheel friction torque requests, front and rear axle
        if hold or cmd.gear == GEAR_NEUTRAL or abs(speed) < p.abs_min_speed:
            self.abs_event = False
        if hold or cmd.gear == GEAR_NEUTRAL:
            request = 0.0
            if hold:
                friction = (p.hold_brake_torque, p.hold_brake_torque)
        else:
            push = feedforward(p, cmd.acceleration, direction * speed)
            if push >= 0:
                self.abs_event = False
                request = min(push * r / ratio, motor_limit(p, omega))
            else:
                brake = -push
                # Regeneration first, unless ABS has worked a wheel in this stop: the brake
                # controller then hands all braking to the friction brakes until the stop
                # ends (handing it back mid-stop would leave a gap the size of the
                # hydraulics' lag every time ABS lets go).
                if any(m.active for m in self.abs):
                    self.abs_event = True
                available = 0.0 if self.abs_event else regen_limit(p, omega, speed)
                regen_force = min(brake, available * ratio / r)
                request = -regen_force * r / ratio
                # The axle split follows the deceleration the car can reach: an EBD works
                # from the measured deceleration, which the tires cap near mu g.
                decel = min(-cmd.acceleration, p.tire_friction * G)
                friction = self.split_friction(brake, regen_force, decel)
        # The motor: the inverter's torque lag, within the envelope at the present speed.
        limit = motor_limit(p, omega)
        self.motor_torque = lag(self.motor_torque, request, dt, p.motor_time_constant)
        self.motor_torque = max(-limit, min(limit, self.motor_torque))
        delivered = self.motor_torque
        # Traction control: a driven wheel spinning past its slip limit cuts motor torque.
        if wheels and p.tc_enabled and delivered > 0:
            spin = max(direction * w.slip for w, d in zip(wheels, self.driven) if d)
            delivered *= self.tc.step(spin, dt)
        else:
            self.tc.step(0.0, dt, enabled=False)
        axle = direction * delivered * ratio
        drive = tuple(axle * s for s in self.drive_share)
        # The friction brakes: dead time, then the pressure's first-order build-up, per axle.
        for i in range(2):
            delayed = self.brake_delay[i].step(friction[i], dt)
            self.pressure[i] = lag(self.pressure[i], delayed, dt, p.brake_time_constant)
            if self.pressure[i] < BRAKE_FLOOR and delayed < BRAKE_FLOOR:
                self.pressure[i] = 0.0
        # ABS per wheel on its slip against the direction of travel (regen counts as braking
        # on a driven wheel); off below abs_min_speed, where the slip means little.
        abs_on = (p.abs_enabled and wheels is not None and not hold
                  and abs(speed) > p.abs_min_speed)
        brake = []
        for i in range(4):
            pressure = self.pressure[0 if IS_FRONT[i] else 1]
            lock = -direction * wheels[i].slip if wheels else 0.0
            braking = pressure > 1.0 or (self.driven[i] and self.motor_torque < 0)
            factor = self.abs[i].step(lock, dt, enabled=abs_on and braking)
            brake.append(min(pressure * factor, p.max_brake_torque))
        self.debug = {"motor_torque": self.motor_torque, "delivered": delivered,
                      "motor_limit": limit, "omega": omega, "regen_force": regen_force,
                      "friction_request": friction, "pressure": tuple(self.pressure),
                      "abs": tuple(m.factor for m in self.abs), "tc": self.tc.factor}
        return Torques(drive, tuple(brake))

    def split_friction(self, brake, regen, decel):
        """Per wheel friction torque (front, rear) for a total braking force `brake` (N) of
        which the motor regenerates `regen` on the driven axle(s): the dynamic axle load
        distribution of the total, with regen taking its axle's share first; regen beyond
        that share (mild braking) comes off the other axle's friction."""
        p = self.p
        share = p.brake_split(decel)
        front, rear = brake * share, brake * (1 - share)
        regen_front, regen_rear = regen * self.front_share, regen * (1 - self.front_share)
        excess = max(0.0, regen_front - front) + max(0.0, regen_rear - rear)
        front = max(0.0, front - regen_front)
        rear = max(0.0, rear - regen_rear)
        cut = min(front, excess)
        front, excess = front - cut, excess - cut
        rear -= min(rear, excess)
        return front * p.wheel_radius / 2, rear * p.wheel_radius / 2


def make(params):
    """The powertrain `params.powertrain` names."""
    name = params.powertrain or "ideal"
    if name == "ideal":
        return IdealPowertrain(params)
    if name == "ev":
        return EvPowertrain(params)
    if name in MODELS:
        raise NotImplementedError(
            f"powertrain {name!r} is reserved and not implemented yet; 'ideal' and 'ev' are "
            "(docs/design.md, Powertrain)")
    raise ValueError(f"unknown powertrain {name!r}; known: {', '.join(MODELS)}")
