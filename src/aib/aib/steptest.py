"""Open-loop step tests of the ego's PhysX vehicle and powertrain in Isaac: no ROS, no Autoware.

    python -m aib.steptest --out DIR [--ego config/ego.yaml] [--powertrain ev|ideal]
        [--set field=value ...] [--world <usd>] [--only test,...]

Run it through `just isaac-steptest` (Isaac's environment). The vehicle is built from the
ego config's rig exactly as `aib.runner` builds it (aib.isaac.vehicle.Vehicle), on a flat
ground plane unless `--world`, and stepped as a host steps it: 0.05 s frames of 3 physics
steps, the command set once per frame (what the ego's command subscriber does). Per test, a
CSV of every frame goes to DIR, and a summary of the measurements to DIR/summary.json:

- `launch`:    full request from standstill: 0-100 km/h and 0-60 mph times
- `brake_100`: 100 km/h, then -12 m/s^2 (past the tires): the stopping distance from 100
- `regen_60`:  60 km/h, then -1.5 m/s^2 (all regen) and -4 m/s^2 (regen + friction)
- `coast`:     100 km/h, then neutral: the road load's deceleration against the model's
- `steps`:     at 10 m/s, +1.5 m/s^2 then -3 m/s^2 steps: the response's dead time and lag
- `accel_brake`, `hard_brake`, `limit_brake`: csb's step-response sequences (roadmap 043)
"""
import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

FRAME = 0.05
SUB = 3
KMH = 1 / 3.6


def speed_hold(target, v, gain=1.0, limit=1.5):
    return max(-limit, min(limit, gain * (target - v)))


# --- the tests: (name, length s, command(t, v, memo) -> (accel, velocity)) -----------------

def launch(t, v, memo):
    if t < 1.0:
        return 0.0, 0.0
    if v >= 100 * KMH or memo.get("done"):
        memo["done"] = True
        return -3.0, 0.0
    return 10.0, 60.0


def brake_100(t, v, memo):
    if t < 1.0:
        return 0.0, 0.0
    if not memo.get("braking"):
        if v >= 100 * KMH:
            memo["braking"] = True
        else:
            return 10.0, 60.0
    return (-12.0, 0.0) if v > 0.05 else (0.0, 0.0)


def regen_60(t, v, memo):
    if t < 1.0:
        return 0.0, 0.0
    if "at" not in memo:
        if v >= 60 * KMH:
            memo["at"] = t
        else:
            return 3.0, 60 * KMH
    dt = t - memo["at"]
    if dt < 2.0:
        return speed_hold(60 * KMH, v), 60 * KMH
    if dt < 4.0:
        return -1.5, 0.0
    if dt < 6.0:
        return speed_hold(60 * KMH, v, limit=2.5), 60 * KMH
    if dt < 8.0:
        return -4.0, 0.0
    return speed_hold(0.0, v, limit=2.0), 0.0


def steps(t, v, memo):
    if t < 1.0:
        return 0.0, 0.0
    if t < 8.0:
        return speed_hold(10.0, v), 10.0
    if t < 10.0:
        return 1.5, 20.0
    if t < 14.0:
        return speed_hold(13.0, v), 13.0
    if t < 15.5:
        return -3.0, 0.0
    return speed_hold(0.0, v, limit=2.0), 0.0


def coast(t, v, memo):
    """To 100 km/h, then neutral: the road load alone (checks the body force)."""
    if t < 1.0:
        return 0.0, 0.0
    if not memo.get("coasting"):
        if v >= 100 * KMH:
            memo["coasting"] = True
        else:
            return 10.0, 60.0
    return 0.0, 30.0, "neutral"


def accel_brake(t, v, memo):
    if t < 1.0:
        return 0.0, 0.0
    if t < 6.0:
        return 1.5, 10.0
    if v > 0.05 and t < 10.0:
        return -3.0, 0.0
    return 0.0, 0.0


def _brake_from_15(decel):
    def cmd(t, v, memo):
        if t < 1.0:
            return 0.0, 0.0
        if t < 11.0:
            return speed_hold(15.0, v, limit=2.5), 15.0
        if v > 0.05:
            return -decel, 0.0
        return 0.0, 0.0
    return cmd


TESTS = {
    "launch": (14.0, launch),
    "brake_100": (24.0, brake_100),
    "regen_60": (24.0, regen_60),
    "steps": (20.0, steps),
    "coast": (30.0, coast),
    "accel_brake": (11.0, accel_brake),
    "hard_brake": (16.0, _brake_from_15(8.0)),
    "limit_brake": (16.0, _brake_from_15(12.0)),
}


# --- measurements from the frames -----------------------------------------------------------

def _cross(rows, key, level, start=0):
    """First time `key` reaches `level`, linearly interpolated, from row `start`."""
    for i in range(max(start, 1), len(rows)):
        a, b = rows[i - 1][key], rows[i][key]
        if (a - level) * (b - level) <= 0 and a != b:
            return rows[i - 1]["t"] + (level - a) / (b - a) * (rows[i]["t"] - rows[i - 1]["t"])
    return None


def _first(rows, pred, start=0):
    return next((i for i in range(start, len(rows)) if pred(rows[i])), None)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def measure(test, rows):
    out = {}
    if test == "launch":
        i0 = _first(rows, lambda r: r["cmd_a"] > 5)
        t0 = rows[i0]["t"] - FRAME  # the command applies from the frame's start
        for name, v in (("0_100_kmh", 100 * KMH), ("0_60_mph", 26.8224)):
            t = _cross(rows, "v", v, i0)
            out[name + "_s"] = None if t is None else round(t - t0, 3)
        sl = [r["slip_rl"] for r in rows[i0:] if r["cmd_a"] > 5]
        out["max_rear_slip"] = round(max(sl), 3)
        out["peak_accel"] = round(max(r["a"] for r in rows[i0:] if r["cmd_a"] > 5), 2)
    elif test in ("brake_100", "limit_brake", "hard_brake"):
        i0 = _first(rows, lambda r: r["cmd_a"] < -5)
        if i0 is None:
            return out
        r0 = rows[i0 - 1]
        stop = _first(rows, lambda r: r["v"] < 0.05, i0)
        if stop is not None:
            out["from_kmh"] = round(r0["v"] / KMH, 1)
            out["distance_m"] = round(rows[stop]["x"] - r0["x"], 2)
            out["time_s"] = round(rows[stop]["t"] - r0["t"], 2)
            # Scaled to exactly 100 km/h (v^2): the request starts within a frame of it
            out["distance_from_100_m"] = round(out["distance_m"] * (100 * KMH / r0["v"]) ** 2, 2)
            mid = [r for r in rows[i0:stop] if r["v"] > 3.0 and r["t"] > r0["t"] + 0.5]
            out["mean_decel_mid"] = round(-_mean(r["a"] for r in mid), 2)
            out["min_slip"] = round(min(min(r["slip_fl"], r["slip_fr"], r["slip_rl"], r["slip_rr"])
                                        for r in rows[i0:stop]), 3)
            out["frames_any_wheel_locked"] = sum(
                1 for r in rows[i0:stop] if r["v"] > 2.0 and min(
                    abs(r["omega_fl"]), abs(r["omega_fr"]), abs(r["omega_rl"]), abs(r["omega_rr"])) < 0.5)
            out["abs_min_factor"] = round(min(r["abs_min"] for r in rows[i0:stop]), 3)
    elif test == "regen_60":
        for req, label in ((-1.5, "regen_1_5"), (-4.0, "blend_4")):
            seg = [r for r in rows if abs(r["cmd_a"] - req) < 1e-6]
            if not seg:
                continue
            seg = seg[10:]  # past the lags
            out[label] = {
                "from_kmh": round(seg[0]["v"] / KMH, 1),
                "decel": round(-_mean(r["a"] for r in seg), 3),
                "regen_decel": round(_mean(r["regen_force"] for r in seg) / r_mass(), 3),
                "friction_torque_sum": round(_mean(r["brake_sum"] for r in seg), 1),
            }
    if test == "coast":
        seg = [r for r in rows if r["neutral"] and 70 * KMH <= r["v"] <= 90 * KMH]
        if seg:
            out["decel_70_90_kmh"] = round(-_mean(r["a"] for r in seg), 4)
            out["expected"] = round(_EXPECT[0](_mean(r["v"] for r in seg)), 4)
            out["mean_pitch_z"] = round(_mean(r["pitch_z"] for r in seg), 5)
    if test in ("steps", "accel_brake"):
        # `steps`: from a 10 m/s cruise (t = 8 s) and 13 m/s (t = 14 s); `accel_brake`: from
        # standstill (the hold brake's release included) and from +1.5
        at = {"steps": {1.5: 8.0, -3.0: 14.0}}.get(test, {1.5: 0.0, -3.0: 0.0})
        for req in (1.5, -3.0):
            i0 = _first(rows, lambda r: abs(r["cmd_a"] - req) < 1e-6 and r["t"] > at[req] + 1e-6)
            if i0 is None:
                continue
            base = _mean(r["a"] for r in rows[max(0, i0 - 10):i0])
            t0 = rows[i0]["t"] - FRAME
            seg = rows[i0:i0 + 30]
            final = _mean(r["a"] for r in seg[16:30])
            span = final - base
            res = {"base": round(base, 3), "settled": round(final, 3)}
            for frac in (0.1, 0.63, 0.9):
                t = _cross(rows, "a", base + frac * span, i0)
                res[f"t{int(frac * 100)}_s"] = None if t is None else round(t - t0, 3)
            out[f"step_{req:+g}"] = res
    if test in ("steps", "accel_brake", "hard_brake"):
        err = [abs(r["a"] - r["cmd_a"]) for r in rows if abs(r["cmd_a"]) > 0.2 and r["v"] > 1.0]
        out["mean_abs_accel_error"] = round(_mean(err), 3)
    return out


_MASS = [1.0]
_EXPECT = [lambda v: float("nan")]


def r_mass():
    return _MASS[0]


# --- driver ---------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ego", default=str(Path(__file__).resolve().parents[1] / "config/ego.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--powertrain", default="")
    ap.add_argument("--set", action="append", default=[], help="VehicleParams field=value")
    ap.add_argument("--world", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("aib.steptest")
    from .control import GEAR_DRIVE, GEAR_NEUTRAL, Command, State
    from .rig import EgoConfig

    cfg = EgoConfig.load(args.ego)
    overrides = dict(cfg.vehicle)
    if args.powertrain:
        overrides["powertrain"] = args.powertrain
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = json.loads(v) if v[:1] in "-0123456789tf[" else v
    cfg.vehicle = overrides
    rig = cfg.load_rig()
    params = rig.vehicle
    _MASS[0] = params.mass
    from . import powertrain as pt

    def expected(v):
        m = params.mass + 2 * params.wheel_mass
        return (-pt.resistance(params, v)
                + 4 * params.wheel_damping_rate * v / params.wheel_radius ** 2) / m
    _EXPECT[0] = expected
    label = args.label or params.powertrain
    out = Path(args.out) / label
    out.mkdir(parents=True, exist_ok=True)
    os.environ.pop("DISPLAY", None)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]})  # no crash uploads
    import omni.physx
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import add_reference_to_stage

    from .isaac.vehicle import Vehicle, ensure_vehicle_scene

    dt = FRAME / SUB
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
    pose = ((0.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0))
    veh = Vehicle(stage, "/World/ego", params, rig.box_center, rig.box_dims, pose)
    log.info("vehicle %s, powertrain %s, %d x %.4f s per frame", rig.name, params.powertrain,
             SUB, dt)
    summary = {"powertrain": params.powertrain, "overrides": {k: str(v) for k, v in overrides.items()},
               "tests": {}}
    only = [t for t in args.only.split(",") if t]
    try:
        for test, (length, command) in TESTS.items():
            if only and test not in only:
                continue
            veh.teleport(pose, physx)
            state = State()
            cmd = Command(received=True, gear=GEAR_DRIVE)
            idle = Command(received=True, gear=GEAR_DRIVE)
            for _ in range(40):  # settle on the hold brake
                for _ in range(SUB):
                    veh.apply(idle, dt, state)
                    world.step(render=False)
                    veh.sample(physx, dt, state)
            x0 = state.position[0]
            rows, memo, t = [], {}, 0.0
            v_prev = state.velocity[0]
            while t < length - 1e-9:
                v = state.velocity[0]
                c = command(t, v, memo)
                cmd.acceleration, cmd.velocity = c[0], c[1]
                cmd.gear = GEAR_NEUTRAL if len(c) > 2 else GEAR_DRIVE
                for _ in range(SUB):
                    veh.apply(cmd, dt, state)
                    world.step(render=False)
                    veh.sample(physx, dt, state)
                t += FRAME
                vx = state.velocity[0]
                ws = veh.wheel_states or [None] * 4
                dbg = getattr(veh.powertrain, "debug", {}) or {}
                tq = veh.torques
                row = {"t": round(t, 3), "v": vx, "a": (vx - v_prev) / FRAME,
                       "x": state.position[0] - x0, "y": state.position[1],
                       "cmd_a": cmd.acceleration, "cmd_v": cmd.velocity,
                       "drive_sum": sum(tq.drive), "brake_sum": sum(tq.brake),
                       "brake_f": tq.brake[0], "brake_r": tq.brake[2],
                       "motor_torque": dbg.get("motor_torque", 0.0),
                       "regen_force": dbg.get("regen_force", 0.0),
                       "tc": dbg.get("tc", 1.0),
                       "abs_min": min(dbg.get("abs", (1.0,))),
                       "pitch_z": state.position[2], "neutral": int(cmd.gear == GEAR_NEUTRAL)}
                for name, w in zip(("fl", "fr", "rl", "rr"), ws):
                    row[f"slip_{name}"] = w.slip if w else 0.0
                    row[f"omega_{name}"] = w.rotation_speed if w else 0.0
                rows.append(row)
                v_prev = vx
            with open(out / f"{test}.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
            m = measure(test, rows)
            summary["tests"][test] = m
            log.info("%s: %s", test, json.dumps(m))
    finally:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        app.close()


if __name__ == "__main__":
    main()
