"""Open-loop step tests of the ego's PhysX vehicle and its onboard computers in Isaac: no ROS,
no Autoware.

    python -m aib.steptest --out DIR [--ego config/ego.yaml] [--powertrain ev|ideal]
        [--set field=value ...] [--world <usd>] [--only test,...]

`--set` takes a field of the PhysX vehicle (aib.params.VehicleParams) or of the onboard
computers' calibration (sim_vehicle_ecu.Calibration), by name; `--powertrain` is the
calibration's `powertrain`.

Run it through `just isaac-steptest` (Isaac's environment). The vehicle is built from the
ego config's rig exactly as `aib.runner` builds it (aib.isaac.vehicle.Vehicle), on a flat
ground plane unless `--world`, and stepped as a host steps it: 0.05 s frames of 3 physics
steps, the command set once per frame (what the ego's command subscriber does). Per test, a
CSV of every frame goes to DIR, and a summary of the measurements to DIR/summary.json:

- `launch`:    full request from standstill: 0-100 km/h and 0-60 mph times
- `brake_100`: 100 km/h, then -12 m/s^2 (past the tires): the stopping distance from 100
- `regen_60`:  60 km/h, then -1.5 m/s^2 (all regen) and -4 m/s^2 (regen + friction)
- `regen_30`:  the same from 30 km/h, which a car on low grip reaches within the test
- `coast`:     100 km/h, then neutral: the road load's deceleration against the model's
- `steps`:     at 10 m/s, +1.5 m/s^2 then -3 m/s^2 steps: the response's dead time and lag
- `accel_brake`, `hard_brake`, `limit_brake`: scensim's step-response sequences (roadmap 043)
- `start_down`, `start_flat`, `start_up`, `hold_down`, `hold_flat`: from rest on a slope (the
  gravity tilted, downhill ahead), under one constant request: Autoware's downhill start
  (-0.028 m/s^2 towards 0.25 m/s, which only the release torque lets roll; issue 070), the
  same on the flat (a creep), an uphill start, and a target of 0 (held)
- `hold_camber`: 60 s held (a target of 0) across a 3 % road camber (the gravity tilted
  sideways, the road falling to the car's right): the sideways creep a parked car must not
  have (csb roadmap 059 N8: 0.9 m in 60 s at a lateral sticky damping of 20, 0.1 cm at 200;
  scensim issue 100's constraint)
- `start_turn`, `start_turn_camber`: from rest with the wheel at full lock (the spec's
  max_steer_angle) under a parking creep (+0.6 m/s^2 towards 1 m/s), on the flat and across the
  3 % camber: whether a stiff sideways hold fights a turn out of a parking spot (issue 102)
- `chatter_camber`: 60 s held across the camber, the body's sideways velocity and the wheels'
  spin read at every physics step (`<test>_substeps.csv`): their RMS and spectrum, since a stiff
  sticky damper can ring (issue 102)
"""
import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

from . import app_exit

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


def _regen(kmh):
    """Up to `kmh`, held 2 s, then -1.5 m/s^2 (regen alone) 2 s, back to `kmh`, then -4 m/s^2
    (regen and friction) 2 s."""
    def script(t, v, memo):
        if t < 1.0:
            return 0.0, 0.0
        if "at" not in memo:
            if v >= kmh * KMH:
                memo["at"] = t
            else:
                return 3.0, kmh * KMH
        dt = t - memo["at"]
        if dt < 2.0:
            return speed_hold(kmh * KMH, v), kmh * KMH
        if dt < 4.0:
            return -1.5, 0.0
        if dt < 6.0:
            return speed_hold(kmh * KMH, v, limit=2.5), kmh * KMH
        if dt < 8.0:
            return -4.0, 0.0
        return speed_hold(0.0, v, limit=2.0), 0.0
    return script


regen_60 = _regen(60)
# On low grip the car takes too long to reach 60 km/h for regen_60's 24 s (scensim issue 106)
regen_30 = _regen(30)


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


def constant(accel, velocity):
    def cmd(t, v, memo):
        return accel, velocity
    return cmd


# the slope tests: degrees, downhill ahead (+x); the others run on the flat
SLOPE = {"start_down": 3.1, "start_up": -3.1, "hold_down": 3.1}
# sideways: degrees, the road falling to the car's right (-y); 3 % is atan(0.03)
CAMBER = {t: math.degrees(math.atan(0.03))
          for t in ("hold_camber", "start_turn_camber", "chatter_camber")}
# the turning starts' tire angle: full lock (set from the rig's max_steer_angle at run time)
STEER = {"start_turn": "max", "start_turn_camber": "max"}
# the tests read at every physics step, not only every frame
SUBSTEP_LOG = {"chatter_camber"}
# The onboard computer's regen and ABS state at every step it takes (one per physics step), to
# `<test>_steps.csv`: which acts first is a question of steps, not of 0.05 s rows (scensim 106)
STEP_LOG = {"regen_60", "regen_30"}

TESTS = {
    "launch": (14.0, launch),
    "brake_100": (24.0, brake_100),
    "regen_60": (24.0, regen_60),
    "regen_30": (24.0, regen_30),
    "steps": (20.0, steps),
    "coast": (30.0, coast),
    "accel_brake": (11.0, accel_brake),
    "hard_brake": (16.0, _brake_from_15(8.0)),
    "limit_brake": (16.0, _brake_from_15(12.0)),
    "start_down": (10.0, constant(-0.028, 0.25)),
    "start_flat": (20.0, constant(-0.028, 0.25)),
    "start_up": (10.0, constant(0.6, 0.25)),
    "hold_down": (10.0, constant(-1.0, 0.0)),
    "hold_flat": (10.0, constant(0.0, 0.0)),
    "hold_camber": (60.0, constant(-1.0, 0.0)),
    "start_turn": (12.0, constant(0.6, 1.0)),
    "start_turn_camber": (12.0, constant(0.6, 1.0)),
    "chatter_camber": (60.0, constant(-1.0, 0.0)),
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
    elif test in ("regen_60", "regen_30"):
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
    if test in ("start_down", "start_flat", "start_up", "hold_down", "hold_flat"):
        at5 = _first(rows, lambda r: r["t"] >= 5.0 - 1e-9)
        out["v_5s"] = round(rows[at5]["v"], 4)
        out["v_end"] = round(rows[-1]["v"], 4)
        out["x_end"] = round(rows[-1]["x"], 4)
    if test in ("start_turn", "start_turn_camber"):
        # the start: when it first moved 0.1 m, where it is after the test, and the turn's radius
        # over the last 4 s (speed over yaw rate) against the bicycle's L / tan(lock)
        d = [math.hypot(r["x"], r["y"] - rows[0]["y"]) for r in rows]
        moved = next((r["t"] for r, dd in zip(rows, d) if dd >= 0.1), None)
        out["t_0p1m_s"] = moved
        out["dist_end_m"] = round(d[-1], 4)
        late = [r for r in rows if r["t"] >= rows[-1]["t"] - 4.0]
        w = _mean([r["yaw_rate"] for r in late])
        sp = _mean([math.hypot(r["v"], r["vy"]) for r in late])
        out["v_end"] = round(sp, 4)
        out["radius_m"] = round(sp / w, 3) if abs(w) > 1e-4 else None
    if test in ("hold_camber", "chatter_camber"):
        # the sideways creep after the settle: its distance over the test, and its rate over
        # the last 30 s, where a steady creep shows
        y0 = rows[0]["y"]
        out["y_end_m"] = round(rows[-1]["y"] - y0, 5)
        out["x_end_m"] = round(rows[-1]["x"], 5)
        late = [r for r in rows if r["t"] >= rows[-1]["t"] - 30.0]
        out["y_rate_cm_s"] = round(100 * (late[-1]["y"] - late[0]["y"])
                                   / max(late[-1]["t"] - late[0]["t"], 1e-9), 5)
    if test in ("steps", "accel_brake", "hard_brake"):
        err = [abs(r["a"] - r["cmd_a"]) for r in rows if abs(r["cmd_a"]) > 0.2 and r["v"] > 1.0]
        out["mean_abs_accel_error"] = round(_mean(err), 3)
    return out


def chatter(sub_rows, dt):
    """The held car's ringing, from every physics step after the first 5 s: the RMS of the body's
    sideways velocity and of each wheel's spin about its mean, and the frequency and share of
    the largest peak in the sideways velocity's spectrum."""
    import numpy as np
    keep = [r for r in sub_rows if r["t"] > 5.0]
    vy = np.array([r["vy"] for r in keep])
    out = {"vy_rms_mm_s": round(float(np.sqrt(np.mean((vy - vy.mean()) ** 2))) * 1000, 5)}
    om = np.array([[r[f"omega_{n}"] for n in ("fl", "fr", "rl", "rr")] for r in keep])
    out["omega_rms_rad_s"] = round(float(np.sqrt(np.mean((om - om.mean(0)) ** 2))), 6)
    spec = np.abs(np.fft.rfft(vy - vy.mean())) ** 2
    freq = np.fft.rfftfreq(len(vy), dt)
    if len(spec) > 1 and spec[1:].sum() > 0:
        k = 1 + int(np.argmax(spec[1:]))
        out["vy_peak_hz"] = round(float(freq[k]), 3)
        out["vy_peak_share"] = round(float(spec[k] / spec[1:].sum()), 4)
    out["sample_hz"] = round(1 / dt, 2)
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
    ap.add_argument("--set", action="append", default=[],
                    help="VehicleParams or Calibration field=value")
    ap.add_argument("--world", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--component", default="",
                    help="a control component's entry point in place of the spec's (scensim 091)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("aib.steptest")
    from dataclasses import fields

    from .control import GEAR_DRIVE, GEAR_NEUTRAL, Command, State
    from .params import VehicleParams, road_load
    from .rig import EgoConfig

    cfg = EgoConfig.load(args.ego)
    vehicle, ecu = dict(cfg.vehicle), dict(cfg.ecu)
    if args.powertrain:
        ecu["powertrain"] = args.powertrain
    physx_fields = {f.name for f in fields(VehicleParams)}
    for kv in args.set:
        k, v = kv.split("=", 1)
        (vehicle if k in physx_fields else ecu)[k] = (
            json.loads(v) if v[:1] in "-0123456789tf[" else v)
    cfg.vehicle, cfg.ecu = vehicle, ecu
    if args.component:
        cfg.component = {"entry_point": args.component}
    overrides = {**vehicle, **ecu}
    rig = cfg.load_rig()
    params = rig.vehicle
    _MASS[0] = params.mass

    def expected(v):
        m = params.mass + 2 * params.wheel_mass
        return (-road_load(params, v)
                + 4 * params.wheel_damping_rate * v / params.wheel_radius ** 2) / m
    _EXPECT[0] = expected
    label = args.label or args.component or rig.ecu.powertrain
    out = Path(args.out) / label
    out.mkdir(parents=True, exist_ok=True)
    os.environ.pop("DISPLAY", None)

    from isaacsim import SimulationApp

    # no crash uploads; the host's Kit settings (ISAAC_KIT_ARGS), as the runner takes them
    app = SimulationApp({"headless": True, "extra_args": ["--/crashreporter/url="]
                         + os.environ.get("ISAAC_KIT_ARGS", "").split()})
    # the rig's fingerprints: what the measurements are of. An override (the ego config's or
    # --set) changes the car the spec describes, so a run with one says the hash is not the car's
    summary = {"powertrain": rig.ecu.powertrain, "overrides": {k: str(v) for k, v in overrides.items()},
               "rig": rig.name, "spec_hash": rig.spec_hash,
               "isaac_physics_hash": rig.isaac_physics_hash,
               "isaac_physics_hash_is_the_cars": not overrides,
               "tests": {}}
    only = [t for t in args.only.split(",") if t]
    code = 0
    try:
        import omni.physx
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage

        from pxr import Gf, UsdPhysics

        from .isaac import ground
        from .isaac.vehicle import Vehicle, ensure_vehicle_scene

        dt = FRAME / SUB
        world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
        stage = omni.usd.get_context().get_stage()
        if args.world:
            add_reference_to_stage(usd_path=os.path.realpath(args.world), prim_path="/World/env")
        else:
            ground.add_ground_plane(stage)  # local: no asset root (scensim issue 117)
        ensure_vehicle_scene(stage)
        scene = next(UsdPhysics.Scene(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
        gravity = scene.CreateGravityDirectionAttr()
        world.reset()
        world.play()
        physx = omni.physx.get_physx_interface()
        pose = ((0.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0))
        veh = Vehicle(stage, "/World/ego", params, rig.make_component(), rig.box_center,
                      rig.box_dims, pose, truth_settings=rig.component.get("truth"))
        log.info("vehicle %s, powertrain %s, %d x %.4f s per frame", rig.name, rig.ecu.powertrain,
                 SUB, dt)
        for test, (length, command) in TESTS.items():
            if only and test not in only:
                continue
            th = math.radians(SLOPE.get(test, 0.0))
            ph = math.radians(CAMBER.get(test, 0.0))
            gravity.Set(Gf.Vec3f(math.sin(th) * math.cos(ph), -math.sin(ph),
                                 -math.cos(th) * math.cos(ph)))
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
            steer = STEER.get(test, 0.0)
            cmd.tire_angle = params.max_steer_angle if steer == "max" else float(steer)
            sub_rows, step_rows, ts = [], [], 0.0
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
                    ts += dt
                    if test in STEP_LOG:
                        d = getattr(veh.ecu, "debug", {})
                        step_rows.append({"t": round(ts, 5), "v": state.velocity[0],
                                          "cmd_a": cmd.acceleration,
                                          "regen_force": d.get("regen_force", 0.0),
                                          "regen_factor": d.get("regen_factor", 1.0),
                                          "regen_cap": d.get("regen_cap"),
                                          "axle_mu": d.get("axle_mu"),
                                          "abs_min": min(d.get("abs", (1.0,))),
                                          "brake_r": veh.torques.brake[2]})
                    if test in SUBSTEP_LOG:
                        wsub = veh.wheel_states or []
                        sub_rows.append({"t": round(ts, 5), "vy": state.velocity[1],
                                         "vx": state.velocity[0],
                                         **{f"omega_{n}": (w.rotation_speed if w else 0.0)
                                            for n, w in zip(("fl", "fr", "rl", "rr"), wsub)}})
                t += FRAME
                vx = state.velocity[0]
                ws = veh.wheel_states or [None] * 4
                dbg = getattr(veh.ecu, "debug", {})
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
                       "pitch_z": state.position[2], "neutral": int(cmd.gear == GEAR_NEUTRAL),
                       "vy": state.velocity[1], "yaw_rate": state.angular_velocity[2]}
                for name, w in zip(("fl", "fr", "rl", "rr"), ws):
                    row[f"slip_{name}"] = w.slip if w else 0.0
                    row[f"omega_{name}"] = w.rotation_speed if w else 0.0
                rows.append(row)
                v_prev = vx
            if step_rows:
                with open(out / f"{test}_steps.csv", "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(step_rows[0]))
                    wr.writeheader()
                    wr.writerows(step_rows)
            with open(out / f"{test}.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
            if sub_rows:
                with open(out / f"{test}_substeps.csv", "w", newline="") as f:
                    wr = csv.DictWriter(f, fieldnames=list(sub_rows[0]))
                    wr.writeheader()
                    wr.writerows(sub_rows)
            m = measure(test, rows)
            if sub_rows:
                m.update(chatter(sub_rows, dt))
            summary["tests"][test] = m
            log.info("%s: %s", test, json.dumps(m))
    except BaseException as e:  # app.close() would end the process with 0 (scensim issue 118)
        code = app_exit.code_of(e)
        if code:
            summary["error"] = f"{type(e).__name__}: {e}"
    finally:
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        app_exit.close(app, code)


if __name__ == "__main__":
    main()
