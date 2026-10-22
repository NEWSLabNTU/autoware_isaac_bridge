"""aib's standalone runner: one ego in an Isaac world, no csb, no SSv2.

    python -m aib.runner --ego config/ego.yaml [--world <usd>] [--pose x,y,z,yaw]
        [--frame-rate 20] [--physics-dt 0.016667] [--display :8] [--realtime] [--duration s]

Run it through `just isaac-ego` (Isaac's environment: scripts/isaac_env.sh). It starts Isaac
Sim (headless unless `--display`), loads the world (a USD stage with its own colliders; no
world: a flat ground plane), puts the ego at `--pose` (map frame, base_link on the ground,
yaw in rad) and steps the world in fixed frames forever (or `--duration` s of sim time),
publishing `/clock`, the vehicle status and the sensors on the ego's domain and applying
Autoware's commands -- what csb's world server does for its ego, without a scenario.
Autoware on the same domain: `just autoware` (aib_launch isaac_simulator.launch.xml).

`--realtime` paces frames to the wall clock; without it the world runs as fast as it can,
which Autoware follows through `/clock`.
"""
import argparse
import logging
import math
import os
import signal
import sys
import time


def parse_pose(text):
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("pose is x,y,z,yaw")
    x, y, z, yaw = vals
    return (x, y, z), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ego", required=True, help="ego config YAML (config/ego.yaml)")
    ap.add_argument("--world", default="", help="USD stage to drive in; empty: a ground plane")
    ap.add_argument("--pose", type=parse_pose, default=parse_pose("0,0,0.05,0"),
                    help="x,y,z,yaw of base_link in the world (m, rad)")
    ap.add_argument("--frame-rate", type=float, default=20.0, help="frames per second (Hz)")
    ap.add_argument("--physics-dt", type=float, default=1 / 60, help="physics step (s)")
    ap.add_argument("--display", default="", help="X display for a window; empty: headless")
    ap.add_argument("--realtime", action="store_true", help="pace frames to the wall clock")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this much sim time")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("aib")

    from .rig import EgoConfig

    cfg = EgoConfig.load(args.ego)
    rig = cfg.load_rig()
    substeps = max(1, round(1.0 / (args.frame_rate * args.physics_dt)))
    dt = 1.0 / (args.frame_rate * substeps)
    if args.display:
        os.environ["DISPLAY"] = args.display
    else:
        os.environ.pop("DISPLAY", None)

    from isaacsim import SimulationApp

    # Kit uploads a crash's minidump to NVIDIA by default: kept on disk only (its dumpDir).
    app = SimulationApp({"headless": not args.display, "extra_args": ["--/crashreporter/url="]})

    import omni.kit.app
    import omni.physx
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.extensions import enable_extension
    from isaacsim.core.utils.stage import add_reference_to_stage

    from .isaac.ego import Ego
    from .isaac.vehicle import ensure_vehicle_scene
    from .ros import EgoRos

    for ext in ("isaacsim.sensors.experimental.rtx", "isaacsim.sensors.rtx.nodes"):
        enable_extension(ext)
    omni.kit.app.get_app().update()
    ros = EgoRos.from_config(cfg, rig)
    ros.publish_clock(0.0)
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    stage = omni.usd.get_context().get_stage()
    if args.world:
        add_reference_to_stage(usd_path=os.path.realpath(args.world), prim_path="/World/env")
    else:
        world.scene.add_default_ground_plane()
    ensure_vehicle_scene(stage)
    world.reset()
    world.play()
    physx = omni.physx.get_physx_interface()
    ego = Ego(rig, ros, stage, "/World/ego", args.pose, name=rig.name)
    # The chassis box is the collider, not a body: the roof lidar sees it (11400 points of
    # the ego itself in Autoware's obstacle cloud, which then stops for "route-obstacle"), so
    # hide it from rendering as csb does when it puts a catalogue body on the ego.
    from pxr import UsdGeom

    UsdGeom.Imageable(stage.GetPrimAtPath(ego.vehicle.body_path)).MakeInvisible()

    def warm_step():
        ego.before_substep(dt)
        world.step(render=True)
        ego.after_substep(physx, dt)

    frames = ego.warm_up(warm_step)
    log.info("ego %s (%s) at %s; sensors warm after %d frame(s); %d substep(s) of %.4f s "
             "per frame", rig.name, rig.path, args.pose[0], frames, substeps, dt)

    stop = {"now": False}

    def on_signal(signum, _frame):
        stop["now"] = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    sim_time, frame = 0.0, 0
    wall0 = time.monotonic()
    render = ego.renders
    try:
        while not stop["now"] and app.is_running():
            ego.take_commands()
            for s in range(substeps):
                ego.before_substep(dt)
                world.step(render=render and s == substeps - 1)
                ego.after_substep(physx, dt)
            sim_time += substeps * dt
            frame += 1
            stamp = ros.publish_clock(sim_time)
            ego.publish(sim_time, stamp)
            if args.duration and sim_time >= args.duration:
                break
            if args.realtime:
                lag = wall0 + sim_time - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
    finally:
        log.info("stopping after %d frame(s), sim time %.2f s", frame, sim_time)
        ego.close()
        ros.close()
        app.close()


if __name__ == "__main__":
    main()
