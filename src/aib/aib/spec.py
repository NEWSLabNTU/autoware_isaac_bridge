"""A vehicle spec: the one YAML a vehicle is described in (docs/spec-format.md). Pure.

`load(path)` reads and validates it; `vehicle_params(spec)` gives the PhysX vehicle's flat
parameters, `sensors(spec)` each sensor with its base_link transform, and
`camera_intrinsics` a pinhole camera's matrix. The generator (`aib.gen`) writes Autoware's
files and the Isaac rig from these.
"""
import math
from dataclasses import dataclass
from pathlib import Path

from .frames import compose, mount_transform
from .params import VehicleParams
from .powertrain import MODELS

SPEC_VERSION = 1
SENSOR_TYPES = ("lidar", "camera", "imu", "gnss")
MOUNT_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")

# section -> (required keys, optional keys)
SECTIONS = {
    "autoware": (("vehicle_model", "sensor_model"), ()),
    "body": ((), ("catalogue_model",)),
    "geometry": (("wheel_base", "wheel_tread", "wheel_radius", "wheel_width", "front_overhang",
                  "rear_overhang", "left_overhang", "right_overhang", "vehicle_height",
                  "max_steer_angle"), ()),
    "mirror": (("min_longitudinal_offset", "max_longitudinal_offset", "min_lateral_offset",
                "max_lateral_offset", "min_height_offset", "max_height_offset"), ()),
    "inertia": (("mass", "com_front_fraction", "com_height"), ("com_lateral", "inertia_box")),
    "wheels": (("mass",), ("damping_rate",)),
    "tires": (("friction",), ("lat_stiff_max_load", "lat_stiff_value", "long_stiff_value",
                              "friction_vs_slip")),
    "brakes": (("max_brake_torque",), ("hold_brake_torque", "distribution")),
    "powertrain": (("model", "max_drive_torque"), ("rolling_resistance",)),
    "steering": ((), ("ackermann", "time_constant", "rate_limit")),
    "suspension": (("frequency", "damping_ratio", "travel"), ()),
    "physx": ((), ("substeps", "substep_threshold_speed", "min_active_slip_denominator",
                   "min_passive_slip_denominator", "min_lateral_slip_denominator",
                   "sticky_tire_threshold_speed", "sticky_tire_threshold_time",
                   "longitudinal_sticky_tire_damping", "lateral_sticky_tire_damping")),
    "sensor_kit": (("mount",), ()),
}
TOP = {"spec_version", "name", "description", "sensors", "provenance", *SECTIONS}
SENSOR_COMMON = ("type", "calibration_key", "link", "frame", "topic", "mount", "isaac")
SENSOR_EXTRA = {
    "lidar": ("channels", "rate_hz", "range", "vertical_fov"),
    "camera": ("width", "height", "horizontal_fov", "rate_hz", "info_topic"),
    "imu": ("rate_hz",),
    "gnss": ("rate_hz",),
}
BRAKE_DISTRIBUTIONS = ("dynamic_axle_load", "static")


class SpecError(ValueError):
    """A spec that is not valid, with the file and the key in the message."""


@dataclass
class Spec:
    path: Path
    doc: dict

    @property
    def name(self):
        return self.doc["name"]

    def section(self, name):
        return self.doc.get(name) or {}


def load(path):
    import yaml

    path = Path(path)
    doc = yaml.safe_load(path.read_text()) or {}
    validate(doc, str(path))
    return Spec(path=path, doc=doc)


def validate(doc, where="spec"):
    def fail(msg):
        raise SpecError(f"{where}: {msg}")

    if doc.get("spec_version") != SPEC_VERSION:
        fail(f"spec_version must be {SPEC_VERSION}, is {doc.get('spec_version')!r}")
    if not doc.get("name"):
        fail("no name")
    unknown = set(doc) - TOP
    if unknown:
        fail(f"unknown key(s) {sorted(unknown)}")
    for section, (required, optional) in SECTIONS.items():
        body = doc.get(section)
        if body is None:
            if required:
                fail(f"no {section}")
            continue
        if not isinstance(body, dict):
            fail(f"{section} is not a mapping")
        missing = [k for k in required if k not in body]
        if missing:
            fail(f"{section}: missing {missing}")
        extra = set(body) - set(required) - set(optional)
        if extra:
            fail(f"{section}: unknown key(s) {sorted(extra)}")
    _check_mount(doc["sensor_kit"]["mount"], "sensor_kit.mount", fail)
    model = doc["powertrain"]["model"]
    if model not in MODELS:
        fail(f"powertrain.model {model!r}: one of {', '.join(MODELS)}")
    dist = doc["brakes"].get("distribution", "dynamic_axle_load")
    if dist not in BRAKE_DISTRIBUTIONS:
        fail(f"brakes.distribution {dist!r}: one of {', '.join(BRAKE_DISTRIBUTIONS)}")
    g = doc["geometry"]
    for k in ("wheel_base", "wheel_tread", "wheel_radius", "wheel_width", "max_steer_angle"):
        if not float(g[k]) > 0:
            fail(f"geometry.{k} must be positive")
    if not 0.0 < float(doc["inertia"]["com_front_fraction"]) < 1.0:
        fail("inertia.com_front_fraction must be in (0, 1)")
    sensors = doc.get("sensors") or {}
    if not isinstance(sensors, dict):
        fail("sensors is not a mapping")
    keys, links = set(), set()
    for name, s in sensors.items():
        where_s = f"sensors.{name}"
        t = s.get("type")
        if t not in SENSOR_TYPES:
            fail(f"{where_s}.type {t!r}: one of {', '.join(SENSOR_TYPES)}")
        extra = set(s) - set(SENSOR_COMMON) - set(SENSOR_EXTRA[t])
        if extra:
            fail(f"{where_s}: unknown key(s) {sorted(extra)} for a {t}")
        for k in ("link", "frame", "topic", "mount"):
            if k not in s:
                fail(f"{where_s}: missing {k}")
        _check_mount(s["mount"], f"{where_s}.mount", fail)
        key = s.get("calibration_key", s["link"])
        if key in keys:
            fail(f"{where_s}: calibration_key {key!r} used twice")
        if s["link"] in links:
            fail(f"{where_s}: link {s['link']!r} used twice")
        keys.add(key)
        links.add(s["link"])
        if t == "camera":
            for k in ("width", "height", "horizontal_fov"):
                if k not in s:
                    fail(f"{where_s}: a camera needs {k}")
    for key, entry in (doc.get("provenance") or {}).items():
        head = key.split(".")[0]
        if head not in TOP:
            fail(f"provenance.{key}: no such section")
        if not isinstance(entry, dict) or "basis" not in entry:
            fail(f"provenance.{key}: needs a basis")


def _check_mount(m, where, fail):
    if not isinstance(m, dict) or set(m) != set(MOUNT_KEYS):
        fail(f"{where} must have exactly {', '.join(MOUNT_KEYS)}")


def assumptions(spec):
    """The provenance keys marked `assumed: true`."""
    return sorted(k for k, v in (spec.doc.get("provenance") or {}).items() if v.get("assumed"))


def vehicle_params(spec):
    """The flat VehicleParams of a spec."""
    d = spec.doc
    g, i, w, t = d["geometry"], d["inertia"], d["wheels"], d["tires"]
    b, p = d["brakes"], d["powertrain"]
    st, su, px = spec.section("steering"), d["suspension"], spec.section("physx")
    base = VehicleParams()
    values = dict(
        model=spec.section("body").get("catalogue_model", ""),
        wheel_base=g["wheel_base"], wheel_tread=g["wheel_tread"],
        wheel_radius=g["wheel_radius"], wheel_width=g["wheel_width"],
        max_steer_angle=g["max_steer_angle"],
        mass=i["mass"],
        com=(i["com_front_fraction"] * g["wheel_base"], i.get("com_lateral", 0.0), i["com_height"]),
        inertia_box=tuple(i.get("inertia_box", base.inertia_box)),
        wheel_mass=w["mass"], wheel_damping_rate=w.get("damping_rate", base.wheel_damping_rate),
        tire_friction=t["friction"],
        lat_stiff_max_load=t.get("lat_stiff_max_load", base.lat_stiff_max_load),
        lat_stiff_value=t.get("lat_stiff_value", base.lat_stiff_value),
        long_stiff_value=t.get("long_stiff_value", base.long_stiff_value),
        friction_vs_slip=tuple(tuple(x) for x in t.get("friction_vs_slip", base.friction_vs_slip)),
        powertrain=p["model"], max_drive_torque=p["max_drive_torque"],
        rolling_resistance=p.get("rolling_resistance", 0.0),
        max_brake_torque=b["max_brake_torque"],
        hold_brake_torque=b.get("hold_brake_torque", base.hold_brake_torque),
        brake_distribution=b.get("distribution", base.brake_distribution),
        ackermann=bool(st.get("ackermann", True)),
        steer_time_constant=st.get("time_constant", 0.0),
        steer_rate_limit=st.get("rate_limit", 0.0),
        suspension_frequency=su["frequency"], suspension_damping_ratio=su["damping_ratio"],
        suspension_travel=su["travel"],
    )
    names = {"substeps": "physx_substeps"}
    for k, v in px.items():
        values[names.get(k, k)] = v
    out = {}
    for k, v in values.items():
        default = getattr(base, k)
        if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
            v = float(v)
        out[k] = v
    return VehicleParams.from_dict(out)


def box(spec):
    """The chassis box from the geometry: (centre, dims) in base_link."""
    g = spec.doc["geometry"]
    length = g["wheel_base"] + g["front_overhang"] + g["rear_overhang"]
    width = g["wheel_tread"] + g["left_overhang"] + g["right_overhang"]
    cx = (g["wheel_base"] + g["front_overhang"] - g["rear_overhang"]) / 2
    cy = (g["left_overhang"] - g["right_overhang"]) / 2
    h = g["vehicle_height"]
    return (cx, cy, h / 2), (length, width, h)


def sensors(spec):
    """name -> dict of the sensor with `key` (calibration key), `base_link` (translation,
    quaternion wxyz) and `enabled` (on Isaac)."""
    kit = mount_transform(spec.doc["sensor_kit"]["mount"])
    out = {}
    for name, s in (spec.doc.get("sensors") or {}).items():
        t, q = compose(kit, mount_transform(s["mount"]))
        isaac = dict(s.get("isaac") or {})
        out[name] = dict(s, key=s.get("calibration_key", s["link"]), base_link=(t, q),
                         enabled=bool(isaac.pop("enabled", True)), isaac=isaac)
    return out


def camera_intrinsics(width, height, horizontal_fov_deg):
    """Pinhole K (row-major 3x3) for a camera with square pixels, principal point centred."""
    fx = (width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)
    return [fx, 0.0, width / 2, 0.0, fx, height / 2, 0.0, 0.0, 1.0]
