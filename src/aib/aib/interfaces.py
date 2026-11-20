"""The vehicle interfaces besides Autoware's, without ROS (scensim roadmap 091 W18): pure.

A vehicle interface is the ego's ROS face for its stack. `autoware` is aib's own (`aib.ros`);
the two here are for an `ackermann_msgs` stack, after f1tenth_system's `humble-devel` (its
`jazzy-devel` has the same names; scensim's docs/design/roboracer-indoor-twin.md, 4 and 6):

| profile | takes | publishes |
|---|---|---|
| `ackermann` | `/drive` (AckermannDriveStamped) | `/odom` and tf `odom` -> `base_link` (`Odometer`), `/sensors/imu/raw` |
| `vesc` | `commands/motor/{speed,duty_cycle,current,brake,position}`, `commands/servo/position` (Float64) | `sensors/core` (the component's telemetry), `sensors/servo_position_command`, `sensors/imu`, `sensors/imu/raw` |

The `vesc` profile is the VESC driver (f1tenth/vesc `vesc_driver_node`), so the user's
`ackermann_to_vesc` and `vesc_to_odom` run unchanged: each command is clipped to the driver's
limits (`DriverLimits`, the user's `vesc.yaml`), the last message on any motor topic wins, and
the clipped servo command is echoed at once, as the driver does. Its topic names are relative,
as the driver's: a namespace moves them all.

What the component gets is `sim_vehicle_ecu.signals`' `AckermannCommand` or `VescCommand`, with
the command's age in simulation time; what it reports in `Actuation.telemetry` goes out as
`sensors/core` (`vesc_state`).
"""
import math

from .deps import use

use("sim_vehicle_ecu")
from sim_vehicle_ecu.signals import MOTOR_MODES, AckermannCommand, VescCommand  # noqa: E402

PROFILES = ("autoware", "ackermann", "vesc")
# the command type each profile hands the component (sim_vehicle_ecu.component.COMMANDS)
COMMANDS = {"autoware": "autoware", "ackermann": "ackermann", "vesc": "vesc"}
# the driver's command topics, relative (commands/motor/<topic> -> the mode it sets)
MOTOR_TOPICS = {"speed": "speed", "duty_cycle": "duty", "current": "current", "brake": "brake",
                "position": "position"}
# f1tenth_system humble-devel's config/vesc.yaml: the defaults of `DriverLimits`
F1TENTH_LIMITS = {"duty_cycle": (0.0, 0.0), "current": (0.0, 100.0), "brake": (-20000.0, 200000.0),
                  "speed": (-23250.0, 23250.0), "position": (0.0, 0.0), "servo": (0.15, 0.85)}
# the driver's feasible ranges, which bound a user's limits (vesc_driver.cpp's CommandLimit)
FEASIBLE = {"duty_cycle": (-1.0, 1.0), "servo": (0.0, 1.0)}
# the fields of vesc_msgs/VescState a component's telemetry may set, by the names the design
# gives them (docs/design/roboracer-indoor-twin.md 2.2) or the message's own
TELEMETRY_FIELDS = {"erpm": "speed", "duty": "duty_cycle", "tachometer": "displacement",
                    "tachometer_abs": "distance_traveled"}
STATE_FLOATS = ("temp_fet", "temp_motor", "current_motor", "current_input", "avg_id", "avg_iq",
                "duty_cycle", "speed", "voltage_input", "charge_drawn", "charge_regen",
                "energy_drawn", "energy_regen", "pid_pos_now", "ntc_temp_mos1", "ntc_temp_mos2",
                "ntc_temp_mos3", "avg_vd", "avg_vq")
STATE_INTS = ("displacement", "distance_traveled", "fault_code", "controller_id")
CORE_RATE_HZ = 50.0  # the driver polls the VESC at 50 Hz (vesc_driver's 20 ms timer)
G = 9.80665  # m/s^2 per g: the VESC reports acceleration in g


def check_profile(profile):
    """The profile's name, or ValueError naming the ones there are."""
    if profile not in PROFILES:
        raise ValueError(f"vehicle_interface {profile!r} is not one of {', '.join(PROFILES)}")
    return profile


INTERFACE_DEFAULTS = {"profile": "autoware", "namespace": "", "remap": {}, "limits": {},
                      "wheelbase": None, "publish_tf": True,
                      "frames": {"odom": "odom", "base": "base_link"},
                      "core_rate_hz": CORE_RATE_HZ}


def interface_settings(vehicle_interface):
    """The ego config's `vehicle_interface` (aib.interfaces) with its defaults; ValueError
    for an unknown key or profile."""
    vi = dict(vehicle_interface or {})
    unknown = set(vi) - set(INTERFACE_DEFAULTS)
    if unknown:
        raise ValueError(f"vehicle_interface: unknown key(s) {sorted(unknown)} (the keys: "
                         f"{', '.join(INTERFACE_DEFAULTS)})")
    out = {**INTERFACE_DEFAULTS, **vi}
    check_profile(out["profile"])
    out["frames"] = {**INTERFACE_DEFAULTS["frames"], **(vi.get("frames") or {})}
    if not isinstance(out["remap"], dict):
        raise ValueError("vehicle_interface.remap: a mapping of topic to topic")
    if out["profile"] != "vesc" and out["limits"]:
        raise ValueError("vehicle_interface.limits are the VESC driver's: the vesc profile only")
    rate = out["core_rate_hz"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not rate > 0:
        raise ValueError(f"vehicle_interface.core_rate_hz {rate!r}: a rate in Hz, > 0")
    return out


class DriverLimits:
    """The VESC driver's command limits: `<name>: [min, max]` for duty_cycle, current, brake,
    speed, position and servo, as the user's vesc.yaml gives `<name>_min` and `<name>_max`.
    Missing ones are f1tenth_system's; each is bounded by the driver's feasible range, and a
    min above its max is swapped, as the driver does."""

    def __init__(self, limits=None):
        limits = dict(limits or {})
        unknown = sorted(set(limits) - set(F1TENTH_LIMITS))
        if unknown:
            raise ValueError(f"vesc limits: unknown {', '.join(unknown)} (the driver's: "
                             f"{', '.join(F1TENTH_LIMITS)})")
        self.range = {}
        for name, default in F1TENTH_LIMITS.items():
            lo, hi = (float(x) for x in limits.get(name, default))
            flo, fhi = FEASIBLE.get(name, (-math.inf, math.inf))
            lo, hi = min(max(lo, flo), fhi), min(max(hi, flo), fhi)
            self.range[name] = (min(lo, hi), max(lo, hi))

    def clip(self, name, value):
        lo, hi = self.range[name]
        return min(max(float(value), lo), hi)


class AckermannInput:
    """The last `/drive`, as an `AckermannCommand` with its age."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.last = None
        self.at = None

    def on_drive(self, now, speed, acceleration=0.0, jerk=0.0, steering_angle=0.0,
                 steering_angle_velocity=0.0):
        self.last = (float(speed), float(acceleration), float(jerk), float(steering_angle),
                     float(steering_angle_velocity))
        self.at = now

    def command(self, now):
        if self.last is None:
            return AckermannCommand()
        return AckermannCommand(*self.last, age=max(0.0, now - self.at))


class VescInput:
    """The VESC driver's command side: the last motor command (any topic) and the servo, each
    clipped to the driver's limits; `on_servo` returns the echo the driver publishes."""

    def __init__(self, limits=None):
        self.limits = limits if isinstance(limits, DriverLimits) else DriverLimits(limits)
        self.reset()

    def reset(self):
        self.motor = None
        self.value = 0.0
        self.at = None
        self.servo = None

    def on_motor(self, now, topic, value):
        """A message on commands/motor/<topic>."""
        mode = MOTOR_TOPICS[topic]
        self.motor, self.value, self.at = mode, self.limits.clip(topic, value), now

    def on_servo(self, value):
        """A message on commands/servo/position; the clipped value, which the driver echoes on
        sensors/servo_position_command."""
        self.servo = self.limits.clip("servo", value)
        return self.servo

    def command(self, now):
        if self.motor is None:
            return VescCommand(servo=self.servo)
        assert self.motor in MOTOR_MODES
        return VescCommand(self.motor, self.value, self.servo, max(0.0, now - self.at))


def vesc_state(telemetry):
    """vesc_msgs/VescState's fields from a component's telemetry (a mapping): `erpm`, `duty`,
    `tachometer` and `tachometer_abs` by the design's names, the others by the message's;
    absent ones 0. Unknown names are ValueError: a misspelt field would publish a silent 0."""
    out = {f: 0.0 for f in STATE_FLOATS}
    out.update({f: 0 for f in STATE_INTS})
    for k, v in dict(telemetry or {}).items():
        f = TELEMETRY_FIELDS.get(k, k)
        if f in STATE_INTS:
            out[f] = int(round(v))
        elif f in STATE_FLOATS:
            out[f] = float(v)
        else:
            raise ValueError(f"telemetry {k!r} is not a VescState field (nor one of "
                             f"{', '.join(TELEMETRY_FIELDS)})")
    return out


class TelemetryClock:
    """`sensors/core` at the driver's own rate, not the frames': the component's telemetry is
    noted after every physics step, and each frame gives the ticks of `rate_hz` that fell in it,
    each with the last telemetry at or before it (as the driver polls the VESC). Times are
    simulation seconds."""

    def __init__(self, rate_hz=CORE_RATE_HZ):
        self.period = 1.0 / float(rate_hz)
        self.reset()

    def reset(self):
        self.samples = []  # (time, telemetry), oldest first
        self.next = None

    def note(self, t, telemetry):
        self.samples.append((float(t), telemetry))

    def due(self, now):
        """[(tick, telemetry)] for the ticks up to `now`; the first call ticks at once."""
        if self.next is None:
            self.next = now
        out = []
        while self.next <= now + 1e-9:
            out.append((self.next, self._at(self.next)))
            self.next += self.period
        if self.samples:  # keep the last at or before the next tick
            keep = max((i for i, (t, _) in enumerate(self.samples) if t <= self.next + 1e-9),
                       default=0)
            del self.samples[:keep]
        return out

    def _at(self, t):
        best = None
        for ts, tel in self.samples:
            if ts <= t + 1e-9:
                best = tel
            else:
                break
        if best is None and self.samples:
            best = self.samples[0][1]
        return best


def vesc_imu(reading):
    """The VESC's IMU as its driver reports it, from aib's reading (orientation wxyz, gyro rad/s,
    specific force m/s^2): `(ypr deg, acceleration g, gyro deg/s, quaternion wxyz)`. The driver
    puts these units on both `sensors/imu` and `sensors/imu/raw` (vesc_packet.cpp: roll, pitch
    and yaw in degrees, acc_* in g, gyr_* in deg/s), with no frame_id."""
    (w, x, y, z), gyro, acc = reading
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    ypr = tuple(math.degrees(a) for a in (roll, pitch, yaw))
    return (ypr, tuple(float(a) / G for a in acc), tuple(math.degrees(float(g)) for g in gyro),
            (float(w), float(x), float(y), float(z)))


class Odometer:
    """`vesc_to_odom`'s kinematics, for the `ackermann` profile's `/odom`: a planar pose from
    the forward speed and the steering angle, integrated by forward Euler from the last update
    (the first sets none), yaw rate speed * tan(angle) / wheelbase; a speed under 0.05 m/s
    counts as 0, as there."""

    def __init__(self, wheelbase):
        if not wheelbase > 0:
            raise ValueError(f"wheelbase {wheelbase!r}: must be > 0")
        self.wheelbase = float(wheelbase)
        self.reset()

    def reset(self):
        self.x = self.y = self.yaw = 0.0
        self.t = None
        self.speed = self.yaw_rate = 0.0

    def update(self, now, speed, steering_angle):
        """Advance to `now` (s) with the speed and angle measured at it; returns
        (x, y, yaw, speed, yaw_rate)."""
        speed = 0.0 if abs(speed) < 0.05 else float(speed)
        yaw_rate = speed * math.tan(steering_angle) / self.wheelbase
        dt = 0.0 if self.t is None else now - self.t
        self.x += speed * math.cos(self.yaw) * dt
        self.y += speed * math.sin(self.yaw) * dt
        self.yaw += yaw_rate * dt
        self.t, self.speed, self.yaw_rate = now, speed, yaw_rate
        return self.x, self.y, self.yaw, speed, yaw_rate


def rolling_speed(wheel_speeds, radius, driven):
    """The forward speed the driven wheels' encoders would give: their mean spin times the
    radius (a VESC counts the motor, which turns with them)."""
    w = [s for s, d in zip(wheel_speeds, driven) if d] or list(wheel_speeds)
    return sum(w) / len(w) * radius
