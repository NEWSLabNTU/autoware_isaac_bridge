"""The flat ground's check under Isaac, with no route to NVIDIA's asset bucket (scensim issue 117).

    ISAAC_ROOT=... source scripts/isaac_env.sh
    python -m aib.ground_check [--isaac-default] [--online]

The asset root is pointed at a port nothing listens on (`--online` leaves Isaac's own, the S3
bucket), so a ground that needs the network fails here as it would offline. `--isaac-default`
builds Isaac's `Scene.add_default_ground_plane` instead of `aib.isaac.ground`: the state before
issue 117. Then, on the flat world as aib.runner and steptest build it (`ensure_light` after):

1. the ground is built;
2. a 1 m box dropped from 3 m comes to rest on it (its centre at 0.5 m, within 2 cm);
3. the plane's physics material is Isaac's (static and dynamic friction 0.5, restitution 0.8);
4. a camera 72 m from the origin, 1.8 m up, looking along +x 10 degrees down, is not dark:
   the mean of its RGB over the frame, 0 to 255, is at least 25. It has no auto exposure
   (aib's ego camera has), so the light on the stage is what it reads.

Prints one GROUND-CHECK json line and exits 0 only when every check passes.
"""
import argparse
import json
import os

UNREACHABLE = "http://127.0.0.1:9/Assets/Isaac/6.2"  # the discard port: nothing answers
CAMERA_AT = (60.0, 40.0, 1.8)
CAMERA_ROTATE_XYZ = (80.0, 0.0, -90.0)  # -z forward turned to +x, 10 degrees down
MIN_MEAN_RGB = 25.0
BOX_REST_Z, BOX_TOLERANCE = 0.5, 0.02


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--isaac-default", action="store_true",
                    help="Isaac's default ground plane (before issue 117)")
    ap.add_argument("--online", action="store_true", help="leave Isaac's asset root as it is")
    args = ap.parse_args(argv)

    from isaacsim import SimulationApp

    extra = ["--/crashreporter/url="] + os.environ.get("ISAAC_KIT_ARGS", "").split()
    if not args.online:
        extra += [f"--/persistent/isaac/asset_root/default={UNREACHABLE}",
                  "--/persistent/isaac/asset_root/timeout=2.0"]
    app = SimulationApp({"headless": True, "extra_args": extra})
    out = {"ground": "isaac-default" if args.isaac_default else "aib", "online": args.online}
    ok = False
    try:
        out.update(_checks(args.isaac_default))
        ok = all(v for k, v in out.items() if k.endswith("_ok"))
    except Exception as e:  # printed before app.close(), which ends the process
        out["error"] = f"{type(e).__name__}: {e}"
    out["pass"] = ok
    print("GROUND-CHECK " + json.dumps(out), flush=True)
    if not ok:  # app.close() ends the process with 0, whatever came before
        os._exit(1)
    app.close()


def _checks(isaac_default):
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.api import World
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    from .isaac import ground
    from .runner import ensure_light

    world = World(stage_units_in_meters=1.0, physics_dt=1 / 60, rendering_dt=1 / 60)
    stage = omni.usd.get_context().get_stage()
    if isaac_default:
        world.scene.add_default_ground_plane()
    else:
        ground.add_ground_plane(stage)
    out = {"built_ok": True, "light_added": ensure_light(stage)}

    # the plane's physics material, as bound for physics
    colliders = [p for p in stage.Traverse()
                 if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetPath().HasPrefix(ground.PATH)]
    mat, _ = UsdShade.MaterialBindingAPI(colliders[0]).ComputeBoundMaterial("physics")
    api = UsdPhysics.MaterialAPI(mat.GetPrim())
    values = [round(float(a.Get()), 3) for a in (api.GetStaticFrictionAttr(),
                                                 api.GetDynamicFrictionAttr(),
                                                 api.GetRestitutionAttr())]
    out["material"] = values
    out["material_ok"] = values == [ground.STATIC_FRICTION, ground.DYNAMIC_FRICTION,
                                    ground.RESTITUTION]

    # a box dropped on it
    box = UsdGeom.Cube.Define(stage, "/World/box")
    box.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(box.GetPrim()).SetTranslate(Gf.Vec3d(0.0, 0.0, 3.0))
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
    UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateMassAttr(100.0)

    # the camera, away from the origin
    cam = UsdGeom.Camera.Define(stage, "/World/check_camera")
    xf = UsdGeom.XformCommonAPI(cam.GetPrim())
    xf.SetTranslate(Gf.Vec3d(*CAMERA_AT))
    xf.SetRotate(Gf.Vec3f(*CAMERA_ROTATE_XYZ), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    rp = rep.create.render_product(str(cam.GetPath()), resolution=(320, 240))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp.path])

    world.reset()
    world.play()
    for _ in range(240):  # 4 s: the box falls, bounces and settles; the renderer converges
        world.step(render=True)
    import omni.physx  # PhysX's own pose: USD may not be written back (Fabric)
    z = float(omni.physx.get_physx_interface().get_rigidbody_transformation(
        "/World/box")["position"][2])
    out["box_z"] = round(z, 3)
    out["box_ok"] = abs(z - BOX_REST_Z) <= BOX_TOLERANCE
    frame = np.asarray(rgb.get_data())
    mean = float(frame[..., :3].mean()) if frame.size else 0.0
    out["camera_mean_rgb"] = round(mean, 1)
    out["camera_ok"] = mean >= MIN_MEAN_RGB
    world.stop()
    return out


if __name__ == "__main__":
    main()
