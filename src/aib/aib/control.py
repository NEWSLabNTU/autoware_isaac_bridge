"""Autoware's commands, the vehicle's state and the steering linkage: pure, no Isaac.

The actuation follows acb's contract on CARLA: Autoware's `Control` asks for a longitudinal
acceleration and a tire angle, and the vehicle realizes them. The vehicle interface keeps the
latest `Command` and hands it to the car's onboard computers as their by-wire request
(`sim_vehicle_ecu`: the drive unit, the brakes, the power steering and its actuator); the
linkage here splits the steering actuator's bicycle angle into Ackermann angles for the two
front wheels.
"""
import math
from dataclasses import dataclass, field

# Autoware's GearCommand / GearReport values
GEAR_NEUTRAL, GEAR_DRIVE, GEAR_REVERSE, GEAR_PARK = 1, 2, 20, 22
MOVE_VELOCITY = 0.01  # m/s: a target above this asks the car to move


def driven_wheels(powertrain: str, drive_axle: str) -> tuple:
    """Which wheels (front left, front right, rear left, rear right) the powertrain drives:
    `ideal` all four, otherwise its `drive_axle` (rear, front or all)."""
    if powertrain == "ideal" or drive_axle == "all":
        return (True,) * 4
    front = drive_axle == "front"
    return (front, front, not front, not front)


def release_sticky(drive, brake, velocity, gear, brake_floor, torque, driven, *, speed,
                   standing):
    """The torques PhysX gets: `drive` and `brake` as the onboard computers put them, unless
    the car stands (its own `speed` within `standing`, PhysX's sticky-tyre threshold speed), in
    drive or reverse, asked to move (target `velocity` past MOVE_VELOCITY), with no drive torque
    and every brake below `brake_floor` (released). Then `torque` N m on each `driven` wheel,
    along the target's direction, and no brake: PhysX's sticky tyres hold a standing car until
    a wheel gets drive torque, where a real car with its brakes released rolls (issue 070). A
    target of 0 (stopping, parked) is left to the hold brake.

    Only a standing car: the sticky state exists only below the threshold speed. A moving car
    whose drive torque is 0 because traction control cut it got the release torque on its
    driven wheels too, which kept them spinning past the road. TC read that spin as slip and
    held the torque at 0, so the car never drove again (scensim issue 103)."""
    if (torque <= 0 or abs(speed) > standing or gear not in (GEAR_DRIVE, GEAR_REVERSE)
            or abs(velocity) <= MOVE_VELOCITY
            or any(d != 0.0 for d in drive) or any(b >= brake_floor for b in brake)):
        return tuple(drive), tuple(brake)
    sign = -1.0 if gear == GEAR_REVERSE else 1.0
    return tuple(sign * torque if d else 0.0 for d in driven), (0.0,) * len(brake)


@dataclass
class Command:
    """The latest of Autoware's commands, as the vehicle interface keeps them."""

    acceleration: float = 0.0
    velocity: float = 0.0
    tire_angle: float = 0.0
    gear: int = GEAR_DRIVE
    received: bool = False


@dataclass
class State:
    """The vehicle as Autoware's status topics and the IMU see it, in base_link."""

    position: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    velocity: tuple = (0.0, 0.0, 0.0)  # base_link frame, m/s
    angular_velocity: tuple = (0.0, 0.0, 0.0)  # base_link frame, rad/s
    acceleration: tuple = (0.0, 0.0, 0.0)  # base_link frame, m/s^2, gravity not included
    tire_angle: float = 0.0
    accel_history: list = field(default_factory=list)


def gear_from_command(command):
    """GearCommand.command as the gear the vehicle keeps, or None to ignore it."""
    if command in (GEAR_DRIVE, GEAR_REVERSE, GEAR_PARK, GEAR_NEUTRAL):
        return int(command)
    if 2 <= command <= 19:  # DRIVE_2 ... DRIVE_18
        return GEAR_DRIVE
    return None


def ackermann(tire_angle, wheel_base, wheel_tread):
    """Front (left, right) wheel angles for a bicycle tire angle: tan(inner) = L / (R - t/2),
    tan(outer) = L / (R + t/2), R = L / tan(bicycle)."""
    if abs(tire_angle) < 1e-5:
        return 0.0, 0.0
    radius = wheel_base / math.tan(tire_angle)
    half_t = wheel_tread / 2
    return math.atan(wheel_base / (radius - half_t)), math.atan(wheel_base / (radius + half_t))


def truth_values(state, wheel_states, names, map_origin=(0.0, 0.0, 0.0)):
    """The ground truth a control component declared (sim_vehicle_ecu.component.TRUTH_INPUTS), as
    the world knows it after the last step: only `names`. Pure."""
    w, x, y, z = state.quat
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    vx, vy = state.velocity[0], state.velocity[1]
    out = {}
    for n in names:
        if n == "pose":
            out[n] = (state.position[0] - map_origin[0], state.position[1] - map_origin[1], yaw)
        elif n == "yaw":
            out[n] = yaw
        elif n == "velocity":
            out[n] = (vx, vy)
        elif n == "speed":
            out[n] = vx
        elif n == "yaw_rate":
            out[n] = state.angular_velocity[2]
        elif n == "wheel_slip":
            out[n] = tuple(ws.slip for ws in wheel_states) if wheel_states else (0.0,) * 4
        elif n == "sideslip":
            out[n] = math.atan2(vy, vx) if abs(vx) > 0.1 else 0.0
        else:
            raise ValueError(f"no ground truth {n!r}")
    return out
