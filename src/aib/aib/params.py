"""What the ego's PhysX vehicle is: pure data, no Isaac.

`VehicleParams` is the flat form the Isaac side consumes. It is written into a rig file by
the generator from a vehicle spec (`aib.gen.vehicle_params`), and a host may override any
field by name afterwards (an ego config's `vehicle:` mapping).

Defaults are the shipped spec's (sim_vehicle_spec's vehicles/tesla_model3.yaml) so a bare
`VehicleParams()` is the same car; the spec, not these defaults, is the source of truth.
"""
import math
from dataclasses import asdict, dataclass, fields

G = 9.81
# How far two descriptions of one car may disagree on wheel base, tread or radius, m.
GEOMETRY_TOLERANCE = 0.05
GEOMETRY_KEYS = ("wheel_base", "wheel_tread", "wheel_radius")


class VehicleMismatch(ValueError):
    """Two descriptions (e.g. vehicle_info and a body's catalogue entry) are different cars."""


@dataclass
class VehicleParams:
    model: str = ""  # the body (catalogue id) the geometry was checked against, if any
    # Geometry: base_link is the rear axle centre on the ground
    wheel_base: float = 3.005
    wheel_tread: float = 1.667
    wheel_radius: float = 0.370
    wheel_width: float = 0.235
    max_steer_angle: float = 0.70  # bicycle tire angle, rad
    # Inertia
    mass: float = 1760.0
    com: tuple = (0.47 * 3.005, 0.0, 0.46)  # base_link
    inertia_box: tuple = (4.8, 1.9, 1.4)  # m: the body's inertia as a uniform box this size
    # Wheels and tires
    wheel_mass: float = 20.0
    wheel_damping_rate: float = 0.25
    tire_friction: float = 1.0
    lat_stiff_max_load: float = 2.0
    lat_stiff_value: float = 17.0
    long_stiff_value: float = 4612.5
    friction_vs_slip: tuple = ((0.0, 1.0), (0.1, 1.0), (1.0, 0.8))
    # Actuation
    powertrain: str = "ev"
    max_drive_torque: float = 1500.0  # per wheel, N m (ideal)
    max_brake_torque: float = 3000.0  # per wheel, N m
    hold_brake_torque: float = 1500.0  # per wheel, N m
    brake_distribution: str = "dynamic_axle_load"
    # Friction brake hydraulics (ev): dead time, then a first-order pressure lag
    brake_dead_time: float = 0.08  # s
    brake_time_constant: float = 0.06  # s
    # ABS (ev): per wheel, on longitudinal slip against the direction of travel
    abs_enabled: bool = True
    abs_slip_release: float = 0.25
    abs_slip_reapply: float = 0.15
    abs_release_rate: float = 8.0  # 1/s, of the brake pressure factor
    abs_apply_rate: float = 4.0  # 1/s
    abs_min_speed: float = 1.5  # m/s
    # Electric drive unit (ev)
    drive_axle: str = "rear"  # rear | front | all
    front_drive_share: float = 0.0  # the front axle's share of motor torque when "all"
    gear_ratio: float = 10.0  # motor to wheel, single speed
    motor_peak_torque: float = 340.0  # N m at the motor shaft
    motor_peak_power: float = 194000.0  # W
    motor_max_rpm: float = 15200.0
    motor_time_constant: float = 0.04  # s, the inverter/motor torque lag
    regen_max_torque: float = 130.0  # N m at the motor shaft
    regen_max_power: float = 60000.0  # W
    regen_fade_low: float = 1.0  # m/s: no regen at or below
    regen_fade_high: float = 3.0  # m/s: full regen at or above
    # Traction control (ev): driven wheels' slip cuts motor torque
    tc_enabled: bool = True
    tc_slip_limit: float = 0.25
    tc_slip_reapply: float = 0.15
    tc_release_rate: float = 20.0  # 1/s, of the motor torque factor
    tc_apply_rate: float = 5.0  # 1/s
    # Road loads, applied to the body as a force (all models)
    rolling_coefficient: float = 0.0095  # Crr
    drag_coefficient: float = 0.23  # Cd
    frontal_area: float = 2.2  # m^2
    air_density: float = 1.2  # kg/m^3
    ackermann: bool = True
    steer_time_constant: float = 0.0  # s; 0: none
    steer_rate_limit: float = 0.0  # rad/s; 0: none
    # Suspension
    suspension_frequency: float = 1.6  # Hz
    suspension_damping_ratio: float = 0.45
    suspension_travel: float = 0.2  # m
    # PhysX vehicle integration
    physx_substeps: int = 8
    substep_threshold_speed: float = 5.0
    min_active_slip_denominator: float = 4.0  # m/s
    min_passive_slip_denominator: float = 4.0
    min_lateral_slip_denominator: float = 1.0
    sticky_tire_threshold_speed: float = 0.2
    sticky_tire_threshold_time: float = 1.0
    longitudinal_sticky_tire_damping: float = 200.0
    lateral_sticky_tire_damping: float = 20.0

    @classmethod
    def from_dict(cls, values, overrides=None):
        """`values` (a rig's `vehicle:`), then `overrides`; unknown names are refused."""
        known = {f.name for f in fields(cls)}
        merged = {**(values or {}), **(overrides or {})}
        unknown = set(merged) - known
        if unknown:
            raise ValueError(f"unknown vehicle field(s) {sorted(unknown)}; known: {sorted(known)}")
        return cls(**{k: _tuple(v) for k, v in merged.items()})

    def to_dict(self):
        return {k: _list(v) for k, v in asdict(self).items()}

    # --- derived ---------------------------------------------------------------------------

    def axle_loads(self):
        """Sprung mass per wheel (front, rear), kg: the mass split by the centre of mass."""
        front = min(max(self.com[0] / self.wheel_base, 0.05), 0.95)
        return self.mass * front / 2, self.mass * (1 - front) / 2

    def brake_split(self, decel):
        """The front axle's share of a braking force at `decel` (m/s^2). With
        `dynamic_axle_load`, the dynamic front load's, so all four tires reach their limit
        together (ideal distribution / EBD); with `static`, the static load's."""
        a, h = self.com[0], self.com[2]
        shift = h * max(decel, 0.0) / G if self.brake_distribution == "dynamic_axle_load" else 0.0
        front = (a + shift) / self.wheel_base
        return min(max(front, 0.05), 0.95)

    def suspension(self):
        """Per axle (front, rear): (sprung mass, spring k N/m, damper c N s/m), and the static
        sag (m), the same on both: each axle at one frequency sags g / w^2."""
        omega = 2 * math.pi * self.suspension_frequency
        out = []
        for sprung in self.axle_loads():
            k = sprung * omega ** 2
            out.append((sprung, k, 2 * self.suspension_damping_ratio * math.sqrt(k * sprung)))
        return tuple(out), G / omega ** 2

    def diagonal_inertia(self):
        lx, ly, lz = self.inertia_box
        s = self.mass / 12.0
        return ((ly * ly + lz * lz) * s, (lx * lx + lz * lz) * s, (lx * lx + ly * ly) * s)


def _tuple(v):
    if isinstance(v, list):
        return tuple(_tuple(x) for x in v)
    return v


def _list(v):
    if isinstance(v, tuple):
        return [_list(x) for x in v]
    return v


def read_vehicle_info(path):
    """The geometry Autoware plans with, from a `vehicle_info.param.yaml`."""
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f)
    p = next(iter(doc.values()))["ros__parameters"]
    return {k: float(p[k]) for k in (*GEOMETRY_KEYS, "wheel_width", "max_steer_angle") if k in p}


def check_geometry(name, theirs, ours):
    """Refuse two descriptions of one car that differ on wheel base, tread or radius by more
    than GEOMETRY_TOLERANCE, naming both numbers. `theirs`/`ours`: dicts of those keys."""
    bad = [f"{k} {ours[k]:.3f} m here, {theirs[k]:.3f} m in {name}" for k in GEOMETRY_KEYS
           if k in theirs and k in ours and abs(ours[k] - theirs[k]) > GEOMETRY_TOLERANCE]
    if bad:
        raise VehicleMismatch(f"the vehicle is not {name}: " + "; ".join(bad)
                              + f" (tolerance {GEOMETRY_TOLERANCE * 100:.0f} cm)")


def bicycle_limit(inner_limit, wheel_base, wheel_tread):
    """The bicycle tire angle at which the inner wheel reaches `inner_limit` (rad), under the
    Ackermann split of `aib.control.ackermann`."""
    if inner_limit <= 0:
        return 0.0
    if inner_limit >= math.pi / 2 - 1e-6:
        return math.atan(wheel_base / (wheel_tread / 2))
    radius = wheel_base / math.tan(inner_limit) + wheel_tread / 2
    return math.atan(wheel_base / radius)
