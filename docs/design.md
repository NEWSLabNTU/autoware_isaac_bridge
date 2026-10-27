# Design

## Pieces

```
deps/sim_vehicle_spec/vehicles/<name>.yaml ──aib.gen──> src/aib/config/<name>.rig.yaml (the Isaac rig)
   (the vehicle spec, a submodule)        ──autoware_gen (autoware_sim_vehicle)──> <name>_description,
                                             <name>_launch, <name>_sensor_kit_{description,launch}

Autoware (ROS domain N) <── /vehicle/status/*, /clock, sensors ── Ego (aib.isaac) in the
   aib_launch isaac_simulator.launch.xml   ── /control/command/* ──>   process stepping the world:
   = autoware_sim_profile's                                              swb_isaac_world, or
     autoware.launch.xml, nothing added                                  aib.runner
     (tesla_model3, its sensor kit)
```

- **The spec** is the vehicle: one YAML per vehicle, sim_vehicle_spec's (roadmap 051 of
  scenario-world-bridge; its `docs/spec-format.md`). The same spec gives Autoware's packages
  (autoware_sim_vehicle), CARLA's files and the SSv2 catalog.
- **The generator** (`aib.gen`) writes the rig, the one output that is aib's, and has a
  check mode that fails when it is stale. Autoware reads autoware_sim_vehicle's description
  and calibration as for any other vehicle; the Isaac side reads the rig.
- **The rig** (`<name>.rig.yaml`) is the spec flattened for the Isaac side: `VehicleParams`
  (one flat field per physics number), the chassis box, each sensor with its base_link
  transform resolved. A host never reads Autoware's files to build the ego.
- **The ego config** (`config/ego.yaml`) is a run of a rig: the ROS domain, the node name,
  and hand overrides of vehicle fields and sensor options (`enabled`, `topic`, `frame`,
  `rate_hz`, `isaac`), applied last. It has no clock setting: `clock_offset` went with
  roadmap 051, and a config naming it, or any other unknown key, is refused at load.

## Library layout

| module | needs | what |
|---|---|---|
| `aib.gen` | yaml, sim_vehicle_spec | the rig from a spec (VehicleParams from the spec's `isaac.physx` and physics, the box, sensors in base_link, camera K, the `spec_hash`) and its `--check` |
| `aib.sim_time` | - | `BASE_NS`, `nanos`, `sim_ns`: the ROS time of a simulation time |
| `aib.params` | - | `VehicleParams`; axle loads, brake split, suspension rates, inertia; geometry check |
| `aib.control` | - | `Command`, `State`, gears, the steering actuator, Ackermann |
| `aib.deps` | - | `deps/sim_vehicle_spec` and `deps/sim_vehicle_ecu` on `sys.path` |
| `aib.frames` | - | quaternions, URDF rpy, transforms, lat/lon about the map origin |
| `aib.sensing` | numpy | IMU and GNSS readings from the state, the PointCloud2 layout, periodic timers |
| `aib.rig` | yaml | `Rig`, `SensorRig`, `EgoConfig` |
| `aib.ros` | Isaac's rclpy + Autoware msgs | `EgoRos`: the vehicle interface, the sensor publishers and the host's traffic signals on one domain |
| `aib.signals` | - | a signal colour as Autoware's `TrafficLightElement` (V2X input, as acb's publisher) |
| `aib.isaac.vehicle` | Isaac | the PhysX vehicle ("drive none"): build, apply, sample, teleport |
| `aib.isaac.sensors` | Isaac | RTX lidar, RGB camera, IMU, GNSS |
| `aib.isaac.ego` | Isaac | `Ego`: one vehicle with its sensors and ROS side |
| `aib.runner` | Isaac | one ego in a world, no swb, no SSv2 |
| `aib.steptest` | Isaac | open-loop step tests of the PhysX vehicle and its onboard computers, no ROS |

The pure modules are what the unit tests cover, under the system Python (3.10) and Isaac's
(3.12) alike.

## The frame

A host owns the stage and the clock. Per frame (20 Hz by default, 3 physics substeps of
1/60 s):

```
ego.take_commands()                 # Autoware's commands that arrived (non-blocking)
for each substep:
    ego.before_substep(dt)          # the onboard computers' torques and steering, to the wheels
    world.step(render=last substep and a rendering sensor)
    ego.after_substep(physx, dt)    # pose; velocity, yaw rate, acceleration by difference
stamp = ros.publish_clock(t_end)    # once per frame, by the host
ego.publish(t_end, stamp)           # vehicle status, IMU, GNSS (at its rate), lidar (one frame late)
```

Lockstep, as acb follows CARLA's ticks. Two choices, with why (scenario-world-bridge
roadmaps 038 and 051):

- **`/clock`**: stamps are `sim_ns(t) = nanos(t) + BASE_NS` (`aib.sim_time`: `BASE_NS` is
  1000 s, `nanos` rounds half away from zero as Rust's `f64::round` does), the integer every
  part of a run computes from the same time, the world protocol's `sim_time_ns` among them.
  `BASE_NS` keeps ROS time clear of 0 ("no clock yet") and of the small values for which
  `now() - duration` goes negative. Time is monotonic within an episode only. A restarted
  runner or world server starts it again; a long-lived Autoware stack would clear its TF
  buffers and EKF for good, so it ends instead (autoware_sim_profile's `clock_guard`) and
  comes back on the new episode. swb_isaac_world keeps its time running over its lifetime
  (a world load starts a new `episode_id` but not a new time), so on Isaac a world load does
  not end the stack; on CARLA it does. (Until 051 a wall-clock offset, `clock_offset: wall`,
  hid a restart, and Isaac's time and SSv2's differed by it.)
- **The lidar one frame late**: a scan sent with its own frame's `/clock` races the EKF pose
  of that stamp to NDT, which then cannot interpolate and flaps MRM.
- **The lidar stamped at its render**: the RTX annotator's data is two renders old when read
  (`isaac.latency_frames`, default 2), so a scan takes the stamp of the frame it was rendered
  in; stamped when read it lagged the car by 0.13 s, 2.6 m at 20 m/s (swb docs/issues/015).

`EgoRos` is made once per process and kept across egos (a new DDS participant among ~200
Autoware ones takes seconds to be discovered).

## Vehicle

PhysX vehicle with no engine model: every substep each wheel gets a drive torque, a brake
torque and a steer angle. base_link is the rear axle centre on the ground; wheels sit at
`(L, +-t/2)` and `(0, +-t/2)`. The suspension is tuned per axle to one frequency on that
axle's sprung mass, so both axles sag `g / w^2` and the car sits level; the suspension frame
is placed so that the sagged wheel centre is at `z = r`. The chassis collider is the box,
lifted clear of the ground so only the wheels touch it.

The physics numbers, their sources and which are assumptions are in the spec; the first
spec is the "plausible real Model 3" of scenario-world-bridge roadmap 043 ("As built:
vehicle").

## The onboard computers

Between the vehicle interface and the wheels sit the car's onboard computers: the drive
unit's controller, the brake module (brake distribution, ABS, traction control, ESC) and the
power steering, with the actuators PhysX's drive-none vehicle lacks. Since
scenario-world-bridge roadmap 055 they are not aib's: they are
[sim_vehicle_ecu](https://github.com/NEWSLabNTU/sim_vehicle_ecu) (`deps/sim_vehicle_ecu`),
which knows no simulator and no ROS. `Vehicle.apply` steps its `Ecu` every physics step (1/60
s; PhysX substeps the vehicle 8 times inside each) with Autoware's latest command as the
by-wire request and what the car's own sensors read after the previous step (`Vehicle.sensors`:
the four wheel speeds, the body's yaw rate and accelerations), and writes the drive and brake
torques and the Ackermann angles of its tire angle to the wheels. The car's speed, each wheel's
slip, the sideslip and the road's friction are the brake module's own estimates; aib hands it
nothing PhysX knows that a car's sensors do not.

Its calibration comes from the spec the rig names (`Rig.ecu`, `sim_vehicle_ecu.Calibration`),
refused when the spec's `spec_hash` is not the rig's; an ego config's `ecu:` overrides fields by
name. The rest of this section is the drive unit as it moved (aib `1d49569`); the slip it
names is the estimate since then.

### Powertrain

`sim_vehicle_ecu.powertrain` is the seam between the longitudinal request and the wheels:
`torques(req, speed, dt, wheels) -> Torques(drive[4], brake[4])`, per wheel in `WHEELS`
order, chosen by the spec's `powertrain.model`.

| model | status | what |
|---|---|---|
| `ideal` | implemented | acb's contract on CARLA: the force `m a + F_resist(v)` (wheel inertia included) as equal drive torque on four wheels, capped; a negative force is braking only, split by the dynamic axle load (ideal distribution); hold brake when standing with no request or in park. No lag, no limits but the caps |
| `ev` | implemented, default | an electric drive unit: motor torque/power envelope, gear reduction, torque lag; regeneration blended with the friction brakes; brake hydraulics (dead time, first-order); ABS per wheel and traction control on slip |
| `physx` | reserved | PhysX DriveStandard (engine torque curve, gears, clutch, differential); the command mapped to throttle/brake through accel and brake maps, as a real vehicle interface does |
| `fmu` | reserved | an FMI 2.0 co-simulation unit (a supplier's powertrain model) stepped each substep |

A spec naming a reserved model fails at load (`NotImplementedError`), so nothing silently
drives a different car. A new model is a `Powertrain` subclass registered in `make`; it may
keep state (`reset()` is called on teleport).

#### Road loads

Rolling resistance (`Crr m g`, ramped in over the first 0.5 m/s) and aerodynamic drag
(`rho Cd A v^2 / 2`) are a force on the body at its centre of mass (PhysxForceAPI, body
frame), set every step from the body's speed by `Vehicle.apply` (`aib.params.road_load`),
whichever model drives. They are physics, not part of the command. Both models' feedforward
(`sim_vehicle_ecu.powertrain.feedforward`) adds the car's estimate of them from its
calibration (today the same coefficients) plus PhysX's wheel damping (`4 c v / r^2`), so `ideal` still realizes
the request exactly and `ev` starts from the right torque. Before, `ideal` added a
`rolling_resistance` force (N per m/s, 0 in the spec) to the command with nothing in the
physics to cancel it; that key is gone (a spec naming it is refused), and nothing counts a
load twice.

#### `ev`

Per physics step, along the gear's direction:

1. **Feedforward.** `F = (m + 2 m_wheel) a_cmd + F_load(v) + F_wheel_damping(v)`.
   Autoware's longitudinal controller closes the loop on top; there is no speed loop here.
2. **Drive** (`F >= 0`): motor torque request `F r / G`, clamped to the envelope at the
   motor speed (driven wheels' mean spin x G): peak torque to base speed (`P / T`), then
   `P / omega`, tapering to 0 over the last 5 % below `motor_max_rpm`.
3. **Braking** (`F < 0`): regeneration first, up to `min(regen_max_torque,
   regen_max_power / omega, envelope)`, faded linearly to 0 between `regen_fade_high` and
   `regen_fade_low` m/s; none while ABS is working any wheel. The friction brakes take the
   rest: the total is split by the dynamic axle load, the regenerating axle's share goes to
   regen first, and regen beyond that share (mild braking) takes the other axle's friction
   off (series blending).
4. **Motor**: the request through a first-order lag (`motor_time_constant`), re-clamped to
   the envelope at the present speed. **Traction control** multiplies it by a factor that
   falls while the driven wheels' largest slip is past `tc.slip_limit` and recovers below
   `tc.slip_reapply`. The axle torque `T G` is halved over the driven axle's wheels (an
   open differential); `all` splits it by `front_drive_share` first.
5. **Friction brakes**: per axle, the request through a dead time (`brakes.dead_time`) and
   a first-order pressure lag (`brakes.time_constant`); the hold brake goes the same way.
   **ABS** multiplies each wheel's pressure by a factor that falls at `release_rate` while
   that wheel's slip against the travel is past `slip_release`, holds between, and climbs at
   `apply_rate` once within `slip_reapply`; off below `abs_min_speed` (where PhysX's slip
   denominators make the slip meaningless). Each wheel's torque is capped at
   `max_brake_torque`.

What the car cannot do (past the envelope, inside the lags, past the tires) it does not do,
and the controller sees the error, as on a real car. The parameters are the Model 3 RWD's
from published data, with their sources and which are estimates in the spec's provenance.

### Validation

`just isaac-steptest` (`aib.steptest`) builds the rig's vehicle in Isaac on a flat ground
plane, no ROS, and runs open-loop tests (0-100 km/h, 100-0 braking, regen at 60 km/h,
acceleration and braking steps, scenario-world-bridge roadmap 043's step-response sequences), writing a CSV per test
and `summary.json` (`set:=field=value` overrides, repeatable). The shipped spec, measured
2026-10-19 (flat ground, friction 1.0; "flat" is the earlier flat grip curve, "realistic"
the spec's grip falling to 0.8 at lock):

| | `ev` | `ideal` | published (Model 3 RWD) |
|---|---|---|---|
| 0-100 km/h | 6.45 s (0-60 mph 6.16 s; 0.27 s of it the hold brake releasing) | 3.16 s | 6.1 s (Tesla EU), 5.8 s 0-60 mph |
| 100-0 km/h, ABS, realistic / flat grip | 42.9 m / 42.3 m, no wheel locked | - (no ABS) | ~40 m |
| 100-0 km/h, no ABS, realistic / flat grip | 44.1 m / 40.9 m, wheels locked | 43.3 m / 39.6 m | |
| regen at ~55 km/h | -1.5 m/s^2 asked: all regen, no friction; at most ~2.0 m/s^2 (0.20 g) | friction only | ~0.2 g |
| +1.5 m/s^2 step from 10 m/s: t10 / t63 / t90 | 0.02 / 0.08 / 0.13 s | 0.01 / 0.035 / 0.05 s | |
| -3 m/s^2 step from 13 m/s: t10 / t63 / t90 | 0.01 / 0.10 / 0.19 s (regen first, friction after its dead time) | 0.005 / 0.034 / 0.048 s | |
| coast 90-70 km/h, neutral | 0.276 m/s^2 (model 0.276: the road load force acts as modelled) | 0.271 | |

ABS's thresholds were tuned with this test on the realistic curve: of five settings, release
past 0.25 / reapply below 0.15 at 8 / 4 per s stopped shortest, 1.2 m inside the locked-wheel
stop. On the flat curve a locked wheel loses nothing, so ABS can only lose there (it keeps
the car steerable). The 0-100 time is 0.35 s over Tesla's: 0.27 s is the hold brake's release
(dead time and lag, the start is from a hold), the rest the PhysX tire's slip at launch
(0.16; a twice as stiff tire gives 6.31 s).

Closed loop: `just isaac-ego` on Town01 (the SimReady world) with `just autoware` and the
pilot (then acb's, now autoware_sim_pilot's `auto_drive`) driving town01_ego_drive's route (190.8, -130.1) to (120.0, -129.8): arrived; median
|commanded - delivered| acceleration 0.018 m/s^2 for `ev` (459 samples), 0.016 for `ideal`,
against acceptance.py's 0.35. One run each, a 70 m route at up to 4.3 m/s.


## Sensors

| type | Isaac | message | notes |
|---|---|---|---|
| lidar | RTX lidar (`isaacsim.sensors.experimental.rtx`), the rig's `isaac.config` / `variant` | PointCloud2 x y z intensity return_type channel | channel from the elevation's rank among `channels`; stamped at its render (`isaac.latency_frames`), sent one frame late |
| camera | USD camera + `rgb` annotator, focal length from the horizontal FOV | Image rgb8 + CameraInfo (K from the spec) | off in the shipped rig; untested on Isaac |
| imu | from the state | Imu: orientation, angular velocity, specific force (acceleration minus gravity) in the IMU frame | every frame unless `rate_hz` |
| gnss | from the state | NavSatFix, GBAS fix, lat/lon about the map origin (TransverseMercator at 0, 0) | `rate_hz` (10) |

Frames and topics are the spec's; for the Model 3 they are acb's (`top`, `imu_link`,
`gnss_link`, on `/sensing/imu/imu_link/imu_raw` and `/sensing/gnss/gnss_link/nav_sat_fix`),
so Autoware's sensing launch and swb's runs see what they see on CARLA. The IMU and
GNSS frames were `carla/imu_link` and `carla/gnss_link` until 2026-10-19; they carry no
simulator name now, on either bridge. The camera publishes in `camera6/camera_optical_link` (acb publishes in
`camera6/camera_link`).

## Single source of truth

The spec (sim_vehicle_spec) is the only file a person edits to change the vehicle, on
CARLA and on Isaac. The rig starts with `Generated from <spec> (spec_hash ...) by aib.gen
...: do not edit` and carries the `spec_hash`; `just check-generated` (run by `just build`
and `just test`, and by the unit test `test_gen.CheckTest`) fails when it differs from what
the spec gives. `test_gen.AgreesWithAutowareTest` checks the rig against Autoware's packages
for the same car (autoware_sim_vehicle): the same spec_hash, geometry and box, and every
sensor at the same transform.

`just vehicle-check` checks the running car against Autoware's side without SSv2 or
Autoware: the runner on a ground plane, headless, then autoware_sim_vehicle's
`vehicle_check` (`autoware_sim_check`): the contract (topics, types, rates, frames, stamps,
`/robot_description`) and the dynamics run (acceleration and brake steps against 0.35 m/s^2,
a steering sweep against the bicycle model). PASS or FAIL, with `report.json`.

## Standalone runner

`aib.runner` (`just isaac-ego`) is the smallest host: Isaac Sim, a world (a USD stage with
its own colliders, or a ground plane), one ego at a pose, fixed frames forever. It is what
acb's standalone mode is to acb: Autoware drives a car with no scenario around it. No NPCs
and no scenario protocol: those are swb_isaac_world's. A host that has signals hands them to
`EgoRos.publish_signals` each frame, which publishes them to Autoware as acb does
(swb_isaac_world sends SSv2's, scenario-world-bridge roadmap 044). The runner has none
commanded, so with `signals:=<map dir>` it runs CARLA's own cycle (`aib.signals.Cycle`): the
lights of a junction, CARLA's light group, turn green one at a time, 10 s green and 3 s amber
each (`green_s:=`, `amber_s:=`), the rest red, every group from sim time 0. The groups are
CARLA's, written once per town into `<map dir>/carla/signal_groups.yaml` by
scenario-world-bridge's `scripts/generate_signal_groups.py` (CARLA holding the town); which
regulatory elements a light is comes from the map's signal table, as for acb. Only the V2X
input moves: a world's lamp prims stay as they are. Without `signals:=`, nothing publishes,
and Autoware stops at every signalised stop line with its light unknown.
