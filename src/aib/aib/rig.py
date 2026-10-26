"""The Isaac rig: the generated file the library builds an ego from, and the ego config that
names it. Pure.

A rig (`config/<vehicle>.rig.yaml`, written by `aib.gen` from a vehicle spec) is the vehicle:
its flat physics (`VehicleParams`), its chassis box, its sensors in base_link. Loading it also
loads the spec it names (under `deps/`) for the car's onboard computers' calibration
(`sim_vehicle_ecu.Calibration`), and refuses a spec whose `spec_hash` is not the rig's. An ego
config (`config/ego.yaml`) is a run of it: which rig, the ROS domain, and hand overrides of
the PhysX vehicle, the calibration and the sensors, applied last.
"""
from dataclasses import dataclass, field, fields
from pathlib import Path

from .deps import DEPS, use
from .params import VehicleParams

use("sim_vehicle_spec", "sim_vehicle_ecu")

from sim_vehicle_ecu import Calibration  # noqa: E402

SENSOR_OVERRIDES = ("enabled", "topic", "frame", "rate_hz", "isaac")


@dataclass
class SensorRig:
    name: str
    type: str
    enabled: bool
    link: str
    frame: str
    topic: str
    translation: tuple
    quat: tuple  # w, x, y, z
    extra: dict = field(default_factory=dict)  # channels, width, k, info_topic, ...
    isaac: dict = field(default_factory=dict)

    @property
    def transform(self):
        return (tuple(self.translation), tuple(self.quat))

    @property
    def period(self):
        rate = float(self.extra.get("rate_hz") or 0.0)
        return 1.0 / rate if rate > 0 else 0.0


@dataclass
class Rig:
    name: str
    vehicle: VehicleParams
    ecu: Calibration  # the onboard computers', from the spec
    box_center: tuple
    box_dims: tuple
    catalogue_model: str = ""
    sensors: dict = field(default_factory=dict)  # name -> SensorRig
    assumptions: list = field(default_factory=list)
    spec_hash: str = ""  # the sim_vehicle_spec spec this rig was generated from
    path: str = ""

    @classmethod
    def load(cls, path, vehicle_overrides=None, sensor_overrides=None, ecu_overrides=None):
        import yaml

        path = Path(path)
        doc = yaml.safe_load(path.read_text()) or {}
        sensor_overrides = sensor_overrides or {}
        unknown = set(sensor_overrides) - set(doc.get("sensors") or {})
        if unknown:
            raise ValueError(f"{path}: override(s) for unknown sensor(s) {sorted(unknown)}")
        sensors = {}
        for name, s in (doc.get("sensors") or {}).items():
            s = dict(s)
            over = sensor_overrides.get(name) or {}
            bad = set(over) - set(SENSOR_OVERRIDES)
            if bad:
                raise ValueError(f"sensor {name}: cannot override {sorted(bad)}; "
                                 f"can: {', '.join(SENSOR_OVERRIDES)}")
            if "isaac" in over:
                s["isaac"] = {**(s.get("isaac") or {}), **over.pop("isaac")}
            s.update(over)
            core = {k: s.pop(k) for k in ("type", "enabled", "link", "frame", "topic")}
            t, q = tuple(s.pop("translation")), tuple(s.pop("quat"))
            isaac = s.pop("isaac", {}) or {}
            sensors[name] = SensorRig(name=name, translation=t, quat=q, extra=s, isaac=isaac,
                                      **core)
        body = doc.get("body") or {}
        spec_hash = str(doc.get("spec_hash") or "")
        return cls(name=doc.get("name", path.stem),
                   vehicle=VehicleParams.from_dict(doc.get("vehicle"), vehicle_overrides),
                   ecu=Calibration.from_spec(spec_doc(path, doc.get("spec"), spec_hash),
                                             ecu_overrides),
                   box_center=tuple(body.get("box_center", (1.5, 0.0, 0.75))),
                   box_dims=tuple(body.get("box_dims", (4.8, 2.1, 1.5))),
                   catalogue_model=body.get("catalogue_model", ""),
                   sensors=sensors, assumptions=list(doc.get("assumptions") or []),
                   spec_hash=spec_hash, path=str(path))

    def of_type(self, kind, enabled_only=True):
        return [s for s in self.sensors.values()
                if s.type == kind and (s.enabled or not enabled_only)]


def spec_doc(rig_path, spec, spec_hash):
    """The document of the spec a rig names (`spec:`, relative to `deps/`), refused when its
    spec_hash is not the rig's: the rig would be stale (`just generate`)."""
    from sim_vehicle_spec import spec as specmod

    if not spec:
        raise ValueError(f"{rig_path}: no spec (regenerate the rig: just generate)")
    loaded = specmod.load(DEPS / spec)
    have = specmod.spec_hash(loaded)
    if spec_hash and have != spec_hash:
        raise ValueError(f"{rig_path}: generated from spec_hash {spec_hash[:12]}, but "
                         f"{DEPS / spec} is {have[:12]}: run just generate")
    return loaded.doc


@dataclass
class EgoConfig:
    """A run of a rig: what swb's `acb_ego.yaml` held, minus the vehicle itself."""

    rig: str = ""  # path, relative to this file
    domain: int = 1
    node_name: str = "isaac_vehicle"
    vehicle: dict = field(default_factory=dict)  # VehicleParams overrides, by field name
    ecu: dict = field(default_factory=dict)  # sim_vehicle_ecu.Calibration overrides, by name
    sensors: dict = field(default_factory=dict)  # sensor name -> overrides (SENSOR_OVERRIDES)
    # Where the vehicle takes Autoware's Control from. Another topic puts a node between
    # Autoware's command gate and the car (autoware-drift's drift_controller, swb roadmap 057).
    control_topic: str = "/control/command/control_cmd"

    @classmethod
    def load(cls, path):
        import yaml

        path = Path(path)
        doc = yaml.safe_load(path.read_text()) or {}
        if "clock_offset" in doc:
            raise ValueError(f"{path}: clock_offset is gone (scenario-world-bridge roadmap 051): "
                             "the ego stamps sim time + BASE_NS, and a stack that sees time go "
                             "back ends; remove the key")
        unknown = set(doc) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"{path}: unknown key(s) {sorted(unknown)}")
        cfg = cls(**doc)
        if not cfg.rig:
            raise ValueError(f"{path}: no rig")
        if not Path(cfg.rig).is_absolute():
            cfg.rig = str((path.parent / cfg.rig).resolve())
        return cfg

    def load_rig(self):
        return Rig.load(self.rig, self.vehicle, self.sensors, self.ecu)
