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
