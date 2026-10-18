"""The powertrain seam: how Autoware's longitudinal command becomes wheel torques. Pure.

A powertrain turns the command (acceleration, velocity, gear) and the vehicle's speed into
one drive torque for every wheel and a brake torque per axle. The PhysX vehicle runs with no
engine model of its own ("drive none") and applies what the powertrain returns each substep.

Models (`powertrain.model` in the spec):

- `ideal` (implemented, the default): acb's contract on CARLA. The requested acceleration is
  realized as the force `m a + F_resist(v)`, split evenly over the four wheels as drive
  torque, capped per wheel; a negative force is braking (never the motor), split between the
  axles by the dynamic axle load. A standing car with no request, or in park, holds on its
  hold brake. No lag, no motor map: the car does what is asked within its limits.
- `physx` (reserved): PhysX's own DriveStandard (engine, gears, clutch, differential), with
  the command mapped to throttle and brake through accel/brake maps as a real vehicle
  interface would.
- `ev` (reserved): an electric drive unit per axle, torque-speed envelope, regeneration
  blended with the friction brakes, actuator lag.
- `fmu` (reserved): an FMI 2.0 co-simulation unit (a supplier's powertrain model), stepped
  each substep.

Only `ideal` exists; the others raise NotImplementedError from `make` so a spec naming them
fails loudly at load instead of driving a different car.
"""
from .control import GEAR_NEUTRAL, GEAR_PARK, GEAR_REVERSE

MODELS = ("ideal", "physx", "ev", "fmu")


class Powertrain:
    """The interface: `torques(cmd, speed, dt) -> (drive per wheel, (brake front, brake rear))`,
    torques in N m, drive signed along base_link x."""

    name = ""

    def __init__(self, params):
        self.p = params

    def torques(self, cmd, speed, dt):  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self):
        """A new episode: forget any internal state (none for `ideal`)."""


class IdealPowertrain(Powertrain):
    name = "ideal"

    def torques(self, cmd, speed, dt):
        p = self.p
        r = p.wheel_radius
        hold = (cmd.gear == GEAR_PARK
                or (abs(cmd.velocity) < 0.01 and cmd.acceleration <= 0.01 and abs(speed) < 0.1))
        if hold:
            return 0.0, (p.hold_brake_torque, p.hold_brake_torque)
        if cmd.gear == GEAR_NEUTRAL:
            return 0.0, (0.0, 0.0)
        direction = -1.0 if cmd.gear == GEAR_REVERSE else 1.0
        # Along the gear's direction: positive pushes that way, negative slows it down. The
        # wheels spin up with the car: their inertia is mass the torque also moves.
        mass = p.mass + 4 * 0.5 * p.wheel_mass
        push = mass * cmd.acceleration + p.rolling_resistance * abs(speed)
        if push >= 0:
            return direction * min(p.max_drive_torque, push * r / 4), (0.0, 0.0)
        # Brakes, never the motor: decelerating to a stop does not drive it backwards. Split
        # by the dynamic axle loads, so a hard stop reaches the tires' limit on all four
        # together instead of locking the unloaded rear first.
        front = p.brake_split(-cmd.acceleration)
        total = -push * r
        return 0.0, (min(total * front / 2, p.max_brake_torque),
                     min(total * (1 - front) / 2, p.max_brake_torque))


def make(params):
    """The powertrain `params.powertrain` names."""
    name = params.powertrain or "ideal"
    if name == "ideal":
        return IdealPowertrain(params)
    if name in MODELS:
        raise NotImplementedError(
            f"powertrain {name!r} is reserved and not implemented yet; only 'ideal' is "
            "(docs/design.md, Powertrain)")
    raise ValueError(f"unknown powertrain {name!r}; known: {', '.join(MODELS)}")
