"""A vehicle body on the stand-alone runner's ego: render only, physics stays the box.

`aib.runner` puts the PhysX vehicle on the stage with a box chassis and cylinder wheels. An ego
config's `body:` names a vehicle catalogue entry (the `vehicles.json` entry format of swb's
Isaac vehicle catalogue, `swb_isaac_world.vehicle_assets`; for one car its `entry.json`), and
this puts that body on it the way swb's world server dresses its egos:
- the body's rear axle sits on base_link;
- the body's own wheels are switched off, and each wheel model goes on the PhysX wheel prim of
  its corner, so that it steers and spins;
- the box and the stand-in cylinders are hidden.

The body carries no colliders. Cameras and the RTX lidar see it, as a real car's sensors see
the car; the stack's crop box removes it from the lidar, as on the vehicle.
"""
import json
import logging
from pathlib import Path

log = logging.getLogger("aib")

# aib.isaac.vehicle's PhysX wheel prims -> a catalogue entry's wheel names
PHYSX_WHEELS = {"frontLeft": "Wheel_Front_Left", "frontRight": "Wheel_Front_Right",
                "rearLeft": "Wheel_Rear_Left", "rearRight": "Wheel_Rear_Right"}


def load_entry(path):
    """A catalogue entry (one JSON object, or a vehicles.json list of one), with the files it
    names resolved against the entry's directory."""
    path = Path(path)
    doc = json.loads(path.read_text())
    if isinstance(doc, list):
        if len(doc) != 1:
            raise ValueError(f"{path}: {len(doc)} entries; name a car's own entry.json")
        doc = doc[0]
    base = path.parent

    def resolve(p):
        return str(p) if Path(p).is_absolute() else str((base / p).resolve())

    if not (base / doc["usd"]).exists():
        base = base.parent  # entry.json beside the USDs, paths relative to the catalogue root
    doc["usd"] = resolve(doc["usd"])
    for w in doc.get("wheels") or []:
        if w.get("usd"):
            w["usd"] = resolve(w["usd"])
    return doc


def _invisible(prim):
    from pxr import UsdGeom

    UsdGeom.Imageable(prim).MakeInvisible()


def dress(stage, vehicle_path, entry, chassis_path=None):
    """Put `entry`'s body on the vehicle at `vehicle_path` (frame base_link). Returns how many
    wheels wear the body's own wheel model."""
    from pxr import Gf, UsdGeom

    path = f"{vehicle_path}/model"
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(-(entry.get("rear_axle_x") or 0.0), 0.0, 0.0))
    x.GetPrim().GetReferences().AddReference(entry["usd"])
    wheels = {w["name"]: w for w in entry.get("wheels", [])}
    root = entry.get("prim", "/Vehicle")
    for w in wheels.values():
        stage.OverridePrim(path + w["path"][len(root):]).SetActive(False)
    dressed = 0
    for physx_name, name in PHYSX_WHEELS.items():
        wp = f"{vehicle_path}/{physx_name}"
        w = wheels.get(name)
        if w is None or not w.get("usd") or not stage.GetPrimAtPath(wp):
            continue
        mesh = UsdGeom.Xform.Define(stage, f"{wp}/model")
        mesh.GetPrim().GetPayloads().AddPayload(w["usd"], w["usd_prim"])
        stand_in = stage.GetPrimAtPath(f"{wp}/render")
        if stand_in:
            _invisible(stand_in)
        dressed += 1
    if chassis_path is not None:
        _invisible(stage.GetPrimAtPath(chassis_path))
    log.info("ego wears %s (rear axle at x=%.3f in the asset, %d wheel(s))",
             entry.get("id", entry["usd"]), entry.get("rear_axle_x") or 0.0, dressed)
    return dressed
