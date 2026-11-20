"""What the ego's PhysX vehicle is: pure data, no Isaac.

`VehicleParams` is the flat form the Isaac side consumes: the body, the wheels and tires, the
suspension, the mechanical parts (the steering's stop, the Ackermann linkage), the road loads
the body meets, PhysX's settings. It is written into a rig file by the generator from a
vehicle spec (`aib.gen.vehicle_params`), and a host may override any field by name afterwards
(an ego config's `vehicle:` mapping). What drives the wheels -- the drive unit's controller,
the brake module with ABS, traction control and ESC, the power steering -- is the car's
onboard computers, `sim_vehicle_ecu` (roadmap 055), with a calibration of its own.

Defaults are the shipped spec's (sim_vehicle_spec's vehicles/tesla_model3.yaml) so a bare
`VehicleParams()` is the same car; the spec, not these defaults, is the source of truth.
"""
import math
from dataclasses import asdict, dataclass, field, fields

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
    # One axle's tyres where they differ from the fields above (scensim roadmap 091 W24):
    # {"front"|"rear": {AXLE_TIRE_FIELDS: value}}; empty: both axles alike
    axle_tires: dict = field(default_factory=dict)
    # The body's physics material against walls and objects (W24): restitution,
    # static_friction, dynamic_friction, combine; empty: the simulator's default material
    contact: dict = field(default_factory=dict)
    # Road loads, applied to the body as a force (`road_load`)
    rolling_coefficient: float = 0.0095  # Crr
    drag_coefficient: float = 0.23  # Cd
    frontal_area: float = 2.2  # m^2
    air_density: float = 1.2  # kg/m^3
    ackermann: bool = True  # the linkage splits the bicycle angle into Ackermann angles
    # Suspension
    suspension_frequency: float = 1.6  # Hz
    suspension_damping_ratio: float = 0.45
    suspension_travel: float = 0.2  # m, the real one
    # PhysX's wheel query (scensim issue 077): "raycast" (one ray down the wheel's centre) or
    # "sweep" (the wheel's cylinder: it climbs a sharp curb, the raycast does not)
    wheel_query: str = "raycast"
    # PhysX's travel, m (0: suspension_travel, or SWEEP_TRAVEL times it with the sweep). Longer:
    # its frame rises by the difference, so the droop stays the real one and its rigid limit
    # lies beyond the real travel, where the bump stop (aib.bump_stop) stops the wheel; the
    # sweep needs it (issues 029, 077: 0.50 m on the Model 3's 0.20)
    physx_travel: float = 0.0
    # m: the stop's engagement before full compression, at most half the real travel (a
    # small car's 3 cm travel takes 1.5 cm; scensim roadmap 091 W6)
    bump_stop: float = 0.06
    bump_rate: float = 0.0  # N/m at full compression (0: 5 times the axle's spring)
    bump_damping: float = 0.5  # the stop's damping ratio at full compression, on the corner
    # PhysX vehicle integration
    physx_substeps: int = 8
    substep_threshold_speed: float = 5.0
    min_active_slip_denominator: float = 4.0  # m/s
    min_passive_slip_denominator: float = 4.0
    min_lateral_slip_denominator: float = 1.0
    sticky_tire_threshold_speed: float = 0.2
    sticky_tire_threshold_time: float = 1.0
    longitudinal_sticky_tire_damping: float = 200.0
    lateral_sticky_tire_damping: float = 200.0
    # PhysX leaves its sticky-tyre state only on drive torque: a car in gear asked to move,
    # with no drive and its brakes released (below the onboard computers' brake_floor), gets
    # this much on each driven wheel, so it can roll as a real one does (issue 070: a downhill
    # start held for good; 1 N m is ~0.003 m/s^2 on a Model 3)
    release_torque: float = 1.0  # N m per driven wheel

    def __post_init__(self):
        for axle, over in (self.axle_tires or {}).items():
            if axle not in ("front", "rear"):
                raise ValueError(f"axle_tires: {axle!r}: front or rear")
            unknown = set(over) - set(AXLE_TIRE_FIELDS)
            if unknown:
                raise ValueError(f"axle_tires.{axle}: unknown field(s) {sorted(unknown)}; "
                                 f"known: {list(AXLE_TIRE_FIELDS)}")
        unknown = set(self.contact or {}) - set(CONTACT_FIELDS)
        if unknown:
            raise ValueError(f"contact: unknown field(s) {sorted(unknown)}; "
                             f"known: {list(CONTACT_FIELDS)}")

    def wheel_tire(self, front):
        """One axle's tyre: {AXLE_TIRE_FIELDS: value}, its override on the car's."""
        out = {k: getattr(self, k) for k in AXLE_TIRE_FIELDS}
        out.update((self.axle_tires or {}).get("front" if front else "rear") or {})
        return out

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
        """The rig's `vehicle:`. An empty axle_tires or contact is left out, so a car without
        them writes the rig it wrote before them."""
        return {k: _list(v) for k, v in asdict(self).items()
                if not (k in ("axle_tires", "contact") and not v)}

    # --- derived ---------------------------------------------------------------------------

    def axle_loads(self):
        """Sprung mass per wheel (front, rear), kg: the mass split by the centre of mass."""
        front = min(max(self.com[0] / self.wheel_base, 0.05), 0.95)
        return self.mass * front / 2, self.mass * (1 - front) / 2

    def suspension(self):
        """Per axle (front, rear): (sprung mass, spring k N/m, damper c N s/m), and the static
        sag (m), the same on both: each axle at one frequency sags g / w^2."""
        omega = 2 * math.pi * self.suspension_frequency
        out = []
        for sprung in self.axle_loads():
            k = sprung * omega ** 2
            out.append((sprung, k, 2 * self.suspension_damping_ratio * math.sqrt(k * sprung)))
        return tuple(out), G / omega ** 2

    def travels(self):
        """(PhysX's travel, the real travel), m: `physx_travel` when set, else with the sweep
        SWEEP_TRAVEL times the real one, else the real one; never shorter than the real one."""
        px = self.physx_travel or (SWEEP_TRAVEL * self.suspension_travel
                                   if self.wheel_query == "sweep" else 0.0)
        return max(px, self.suspension_travel), self.suspension_travel

    def bump_engagement(self):
        """The bump stop's engagement, m: `bump_stop`, at most half the real travel."""
        return min(self.bump_stop, 0.5 * self.suspension_travel)

    def diagonal_inertia(self):
        lx, ly, lz = self.inertia_box
        s = self.mass / 12.0
        return ((ly * ly + lz * lz) * s, (lx * lx + lz * lz) * s, (lx * lx + ly * ly) * s)


# The tyre fields an axle may set apart (sim_vehicle_spec's tires.front, tires.rear)
AXLE_TIRE_FIELDS = ("tire_friction", "lat_stiff_max_load", "lat_stiff_value", "long_stiff_value")
# The body's contact material (sim_vehicle_spec's body.contact)
CONTACT_FIELDS = ("restitution", "static_friction", "dynamic_friction", "combine")

# With the sweep and no physx_travel, PhysX's travel is this many times the real one: at 2.0 the
# Model 3's sweep still threw the car off a 10 cm cleat at 20 m/s, at 2.5 not (scensim issue 077)
SWEEP_TRAVEL = 2.5

# Below this speed rolling resistance ramps to zero, so a standing car is not pushed about.
ROLLING_RAMP = 0.5  # m/s


def road_load(p, v):
    """The road load at a speed `v` (m/s along x), N, signed against `v`: rolling resistance
    (Crr m g, ramped in below ROLLING_RAMP) and aerodynamic drag (rho Cd A v^2 / 2). The host
    applies it to the body; the onboard computers' feedforward has its own estimate of it."""
    roll = p.rolling_coefficient * p.mass * G * min(1.0, abs(v) / ROLLING_RAMP)
    drag = 0.5 * p.air_density * p.drag_coefficient * p.frontal_area * v * v
    return -math.copysign(roll + drag, v) if v else 0.0


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
