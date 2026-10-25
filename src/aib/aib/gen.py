"""The Isaac rig from a vehicle spec (roadmap 051): what aib builds the PhysX vehicle and its
sensors from (aib.rig).

    python -m aib.gen [--specs deps/sim_vehicle_spec/vehicles] [--root .]           # write
    python -m aib.gen ... --check   # exit 1 when a rig is stale

The spec is sim_vehicle_spec's (deps/sim_vehicle_spec, a submodule): one per car, the same one
Autoware's packages (autoware_sim_vehicle), CARLA's files and the SSv2 catalog come from. This
writes only what is aib's: `src/aib/config/<name>.rig.yaml`, whose `vehicle` is aib's own
parameter schema (aib.params.VehicleParams). Every rig names its spec and spec_hash. The car's
onboard computers (sim_vehicle_ecu) read their calibration from the same spec when the rig is
loaded (aib.rig), so the rig holds the PhysX vehicle only.
"""
import argparse
import sys
from pathlib import Path

from .deps import DEPS, REPO, use

use("sim_vehicle_spec")
SPECS = DEPS / "sim_vehicle_spec/vehicles"

from sim_vehicle_spec import emit  # noqa: E402
from sim_vehicle_spec import spec as specmod  # noqa: E402

from .params import VehicleParams  # noqa: E402

TOOL = "aib.gen (autoware_isaac_bridge `just generate`)"


def vehicle_params(spec):
    """The flat VehicleParams of a spec: the PhysX vehicle."""
    d = spec.doc
    g, i, w, t = d["geometry"], d["inertia"], d["wheels"], d["tires"]
    rs = d.get("resistance")
    st, su = spec.section("steering"), d["suspension"]
    px = spec.section("isaac").get("physx") or {}
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
        ackermann=bool(st.get("ackermann", True)),
        suspension_frequency=su["frequency"], suspension_damping_ratio=su["damping_ratio"],
        suspension_travel=su["travel"],
    )
    # No `resistance` section: no road load (a spec says what its car has).
    for key in ("rolling_coefficient", "drag_coefficient", "frontal_area"):
        values[key] = (rs or {}).get(key, getattr(base, key) if rs is not None else 0.0)
    values["air_density"] = (rs or {}).get("air_density", base.air_density)
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


def rig(spec, header, spec_rel):
    center, dims = specmod.box(spec)
    sensors = {}
    for name, s in specmod.sensors(spec).items():
        (t, q) = s["base_link"]
        entry = {"type": s["type"], "enabled": s["enabled"], "link": s["link"],
                 "frame": s["frame"], "topic": s["topic"],
                 "translation": list(t), "quat": list(q)}
        for k in specmod.SENSOR_EXTRA[s["type"]]:
            if k in s:
                entry[k] = s[k]
        if s["type"] == "camera":
            entry["k"] = specmod.camera_intrinsics(s["width"], s["height"], s["horizontal_fov"])
        if s["isaac"]:
            entry["isaac"] = s["isaac"]
        sensors[name] = entry
    doc = {
        "spec": spec_rel,
        "name": spec.name,
        "description": spec.doc.get("description", ""),
        "assumptions": specmod.assumptions(spec),
        "spec_hash": specmod.spec_hash(spec),
        "body": {"catalogue_model": spec.section("body").get("catalogue_model", ""),
                 "box_center": list(center), "box_dims": list(dims)},
        "vehicle": vehicle_params(spec).to_dict(),
        "sensors": sensors,
    }
    return emit.comment(header) + emit.comment(
        "The Isaac rig: what aib builds the PhysX vehicle and its sensors from (aib.rig).\n"
        "Sensor transforms are base_link -> link, translation in m, quaternion w x y z.") \
        + emit.yaml_text(doc)


# --- driver ---------------------------------------------------------------------------------

def outputs(specs=SPECS, root=REPO):
    """{path relative to root: content} for every spec under `specs`."""
    out = {}
    for path in sorted(Path(specs).glob("*.yaml")):
        spec = specmod.load(path)
        rel = f"sim_vehicle_spec/vehicles/{path.name}"
        out[f"src/aib/config/{spec.name}.rig.yaml"] = rig(spec, emit.header(spec, TOOL, rel), rel)
    return out


def stale(specs=SPECS, root=REPO):
    return emit.stale(outputs(specs, root), root)


def write(specs=SPECS, root=REPO):
    return emit.write(outputs(specs, root), root)


def main(argv=None):
    ap = argparse.ArgumentParser(description="the Isaac rig from a vehicle spec")
    ap.add_argument("--specs", type=Path, default=SPECS)
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 listing the rigs that are stale")
    args = ap.parse_args(argv)
    if args.check:
        bad = stale(args.specs, args.root)
        for p in bad:
            print(f"stale: {p}; run `just generate`", file=sys.stderr)
        print("generated files: " + ("STALE" if bad else "up to date"))
        return 1 if bad else 0
    for p in write(args.specs, args.root):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
