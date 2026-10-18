"""The generator: a vehicle spec to Autoware's vehicle and sensor kit files and the Isaac rig.

    python -m aib.gen vehicles/tesla_model3.yaml [--root .]           # write
    python -m aib.gen vehicles/tesla_model3.yaml [--root .] --check   # exit 1 when stale

Every file it writes starts with a header naming the spec and saying not to edit it. Paths
follow Autoware's naming: for `autoware.vehicle_model: V` and `sensor_model: S`,

    src/V_launch/V_description/config/vehicle_info.param.yaml, mirror.param.yaml
    src/S_launch/S_description/config/sensors_calibration.yaml, sensor_kit_calibration.yaml
    src/S_launch/S_description/urdf/sensor_kit.xacro
    src/aib_individual_params/config/default/S/  (the same two calibration files)
    src/aib/config/<name>.rig.yaml               (the Isaac rig, aib.rig)
"""
import argparse
import sys
from pathlib import Path

from . import spec as specmod

HEADER = "Generated from {spec} by aib.gen (`just generate`): do not edit."


def _num(v):
    if isinstance(v, bool) or isinstance(v, int):
        return v
    if isinstance(v, float):
        r = round(v, 9)
        return 0.0 if r == 0 else r
    return v


def _round(v):
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_round(x) for x in v]
    return _num(v)


def _fmt(v):
    """A value from the spec as written (no rounding: a calibration keeps its digits)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return repr(v)


def _comment(text, mark="#"):
    return "".join(f"{mark} {line}".rstrip() + "\n" for line in text.splitlines())


def _yaml(doc):
    import yaml

    return yaml.safe_dump(_round(doc), sort_keys=False, default_flow_style=None, width=100)


# --- Autoware vehicle description -----------------------------------------------------------

def vehicle_info(spec, header):
    g = spec.doc["geometry"]
    length = g["wheel_base"] + g["front_overhang"] + g["rear_overhang"]
    width = g["wheel_tread"] + g["left_overhang"] + g["right_overhang"]
    rows = [
        ("wheel_radius", "ground to wheel centre"),
        ("wheel_width", "lateral width of the tire"),
        ("wheel_base", "front to rear axle centre"),
        ("wheel_tread", "left to right wheel centre"),
        ("front_overhang", "front axle to the front edge"),
        ("rear_overhang", "rear axle to the rear edge"),
        ("left_overhang", "left wheels to the left edge"),
        ("right_overhang", "right wheels to the right edge"),
        ("vehicle_height", "ground to the roof"),
        ("max_steer_angle", "bicycle-model tire angle limit, rad"),
    ]
    out = _comment(header) + _comment(
        f"Autoware's vehicle_info for {spec.name} (autoware_vehicle_info_utils::VehicleInfo).\n"
        f"Length {_num(length)} m, width {_num(width)} m, height {_num(g['vehicle_height'])} m.")
    out += "/**:\n  ros__parameters:\n"
    for key, what in rows:
        out += f"    {key}: {_fmt(g[key])}  # {what}\n"
    return out


def mirror(spec, header):
    m = spec.doc["mirror"]
    out = _comment(header) + "/**:\n  ros__parameters:\n"
    for key in specmod.SECTIONS["mirror"][0]:
        out += f"    {key}: {_fmt(m[key])}\n"
    return out


# --- sensor kit -----------------------------------------------------------------------------

def _mount(m, indent):
    pad = " " * indent
    return "".join(f"{pad}{k}: {_fmt(m[k])}\n" for k in specmod.MOUNT_KEYS)


def sensors_calibration(spec, header):
    out = _comment(header) + _comment("base_link -> sensor_kit_base_link (the kit's mount).")
    out += "base_link:\n  sensor_kit_base_link:\n" + _mount(spec.doc["sensor_kit"]["mount"], 4)
    return out


def sensor_kit_calibration(spec, header):
    out = _comment(header) + _comment(
        "Each sensor's mount in sensor_kit_base_link: metres (x forward, y left, z up), radians.")
    out += "sensor_kit_base_link:\n"
    for name, s in specmod.sensors(spec).items():
        out += f"  # {name}: {s['type']}\n  {s['key']}:\n" + _mount(s["mount"], 4)
    return out


OPTICAL_RPY = "-1.5707963267948966 0 -1.5707963267948966"


def sensor_kit_xacro(spec, header):
    lines = [
        '<?xml version="1.0"?>',
        f"<!-- {header} -->",
        f'<robot name="{spec.doc["autoware"]["sensor_model"]}" xmlns:xacro="http://ros.org/wiki/xacro">',
        "  <!-- The sensor kit's links, each placed from sensor_kit_calibration.yaml under",
        "       config_dir (Autoware's vehicle launch passes the description's config). -->",
        f'  <xacro:arg name="config_dir" default="$(find {spec.doc["autoware"]["sensor_model"]}_description)/config"/>',
        "",
        '  <xacro:macro name="sensor_kit_macro" params="parent x y z roll pitch yaw">',
        "    <xacro:property name=\"calibration\" value=\"${xacro.load_yaml('$(arg config_dir)/sensor_kit_calibration.yaml')}\"/>",
        '    <joint name="sensor_kit_base_link_joint" type="fixed">',
        '      <origin rpy="${roll} ${pitch} ${yaw}" xyz="${x} ${y} ${z}"/>',
        '      <parent link="${parent}"/>',
        '      <child link="sensor_kit_base_link"/>',
        "    </joint>",
        '    <link name="sensor_kit_base_link"/>',
    ]
    for name, s in specmod.sensors(spec).items():
        key, link, frame = s["key"], s["link"], s["frame"]
        c = f"calibration['sensor_kit_base_link']['{key}']"
        lines += [
            "",
            f"    <!-- {name}: {s['type']}, published in {frame} on {s['topic']} -->",
            f'    <joint name="{_joint(link)}" type="fixed">',
            f'      <origin xyz="${{{c}[\'x\']}} ${{{c}[\'y\']}} ${{{c}[\'z\']}}"',
            f'              rpy="${{{c}[\'roll\']}} ${{{c}[\'pitch\']}} ${{{c}[\'yaw\']}}"/>',
            '      <parent link="sensor_kit_base_link"/>',
            f'      <child link="{link}"/>',
            "    </joint>",
            f'    <link name="{link}"/>',
        ]
        if frame != link:
            rpy = OPTICAL_RPY if s["type"] == "camera" else "0 0 0"
            lines += [
                f'    <joint name="{_joint(frame)}" type="fixed">',
                f'      <origin xyz="0 0 0" rpy="{rpy}"/>',
                f'      <parent link="{link}"/>',
                f'      <child link="{frame}"/>',
                "    </joint>",
                f'    <link name="{frame}"/>',
            ]
    lines += ["  </xacro:macro>", "</robot>", ""]
    return "\n".join(lines)


def _joint(link):
    return link.replace("/", "_") + "_joint"


# --- the Isaac rig --------------------------------------------------------------------------

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
        "body": {"catalogue_model": spec.section("body").get("catalogue_model", ""),
                 "box_center": list(center), "box_dims": list(dims)},
        "vehicle": specmod.vehicle_params(spec).to_dict(),
        "sensors": sensors,
    }
    return _comment(header) + _comment(
        "The Isaac rig: what aib builds the PhysX vehicle and its sensors from (aib.rig).\n"
        "Sensor transforms are base_link -> link, translation in m, quaternion w x y z.") \
        + _yaml(doc)


# --- driver ---------------------------------------------------------------------------------

def outputs(spec_path, root):
    """{path relative to root: content} for one spec."""
    root = Path(root).resolve()
    spec_path = Path(spec_path).resolve()
    spec = specmod.load(spec_path)
    try:
        rel = spec_path.relative_to(root).as_posix()
    except ValueError:
        rel = spec_path.name
    header = HEADER.format(spec=rel)
    vm = spec.doc["autoware"]["vehicle_model"]
    sm = spec.doc["autoware"]["sensor_model"]
    vdesc = f"src/{vm}_launch/{vm}_description"
    sdesc = f"src/{sm}_launch/{sm}_description"
    sc = sensors_calibration(spec, header)
    skc = sensor_kit_calibration(spec, header)
    return {
        f"{vdesc}/config/vehicle_info.param.yaml": vehicle_info(spec, header),
        f"{vdesc}/config/mirror.param.yaml": mirror(spec, header),
        f"{sdesc}/config/sensors_calibration.yaml": sc,
        f"{sdesc}/config/sensor_kit_calibration.yaml": skc,
        f"{sdesc}/urdf/sensor_kit.xacro": sensor_kit_xacro(spec, header),
        f"src/aib_individual_params/config/default/{sm}/sensors_calibration.yaml": sc,
        f"src/aib_individual_params/config/default/{sm}/sensor_kit_calibration.yaml": skc,
        f"src/aib/config/{spec.name}.rig.yaml": rig(spec, header, rel),
    }


def stale(spec_path, root):
    """The outputs whose file differs from what the spec gives (or is missing)."""
    root = Path(root)
    return [p for p, text in outputs(spec_path, root).items()
            if not (root / p).is_file() or (root / p).read_text() != text]


def write(spec_path, root):
    root = Path(root)
    written = []
    for p, text in outputs(spec_path, root).items():
        f = root / p
        if f.is_file() and f.read_text() == text:
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        written.append(p)
    return written


def default_root(spec_path):
    """The repository the spec is in: the first parent holding a `src` directory."""
    for d in Path(spec_path).resolve().parents:
        if (d / "src").is_dir():
            return d
    return Path.cwd()


def main(argv=None):
    ap = argparse.ArgumentParser(description="aib vehicle spec generator")
    ap.add_argument("specs", nargs="+", help="vehicle spec YAML(s)")
    ap.add_argument("--root", default=None, help="repository root (default: the spec's)")
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 listing the outputs that are stale")
    args = ap.parse_args(argv)
    bad = 0
    for s in args.specs:
        root = Path(args.root) if args.root else default_root(s)
        if args.check:
            for p in stale(s, root):
                print(f"stale: {p} (from {s}); run `just generate`", file=sys.stderr)
                bad += 1
        else:
            for p in write(s, root):
                print(f"wrote {p}")
    if args.check:
        print("generated files: " + ("STALE" if bad else "up to date"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
