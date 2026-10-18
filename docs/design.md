# Design

## Pieces

```
vehicles/<name>.yaml ──aib.gen──┬─> aib_vehicle_description/config/vehicle_info.param.yaml, mirror.param.yaml
   (the vehicle spec)           ├─> aib_sensor_kit_description/config/sensors_calibration.yaml,
                                │   sensor_kit_calibration.yaml, urdf/sensor_kit.xacro
                                ├─> aib_individual_params/config/default/aib_sensor_kit/ (same calibration)
                                └─> src/aib/config/<name>.rig.yaml  (the Isaac rig)

Autoware (ROS domain N) <── /vehicle/status/*, /clock, sensors ── Ego (aib.isaac) in the
   aib_launch isaac_simulator.launch.xml   ── /control/command/* ──>   process stepping the world:
   = acb's carla_simulator.launch.xml                                    csb's world server, or
     + aib_vehicle, aib_sensor_kit                                       aib.runner
     - vehicle interface
```

- **The spec** is the vehicle: one YAML per vehicle ([spec-format.md](spec-format.md)).
- **The generator** (`aib.gen`) writes every file the spec owns and has a check mode that
  fails when any is stale. Autoware reads the description and calibration as for any other
  vehicle; the Isaac side reads the rig.
- **The rig** (`<name>.rig.yaml`) is the spec flattened for the Isaac side: `VehicleParams`
  (one flat field per physics number), the chassis box, each sensor with its base_link
  transform resolved. A host never reads Autoware's files to build the ego.
- **The ego config** (`config/ego.yaml`) is a run of a rig: the ROS domain, the clock
  offset, and hand overrides of vehicle fields and sensor options (`enabled`, `topic`,
  `frame`, `rate_hz`, `isaac`), applied last. csb's `acb_ego.yaml` held all of this plus
  the vehicle; the vehicle moved to the spec.

## Library layout

| module | needs | what |
|---|---|---|
| `aib.spec` | yaml | load and validate a spec; derive VehicleParams, the box, sensors in base_link, camera K |
| `aib.gen` | yaml | the generator and its `--check` |
| `aib.params` | - | `VehicleParams`; axle loads, brake split, suspension rates, inertia; geometry check |
| `aib.control` | - | `Command`, `State`, gears, the steering actuator, Ackermann |
| `aib.powertrain` | - | the powertrain seam; `ideal` |
| `aib.frames` | - | quaternions, URDF rpy, transforms, lat/lon about the map origin |
| `aib.sensing` | numpy | IMU and GNSS readings from the state, the PointCloud2 layout, periodic timers |
| `aib.rig` | yaml | `Rig`, `SensorRig`, `EgoConfig` |
| `aib.ros` | Isaac's rclpy + Autoware msgs | `EgoRos`: the vehicle interface and the sensor publishers on one domain |
| `aib.isaac.vehicle` | Isaac | the PhysX vehicle ("drive none"): build, apply, sample, teleport |
| `aib.isaac.sensors` | Isaac | RTX lidar, RGB camera, IMU, GNSS |
| `aib.isaac.ego` | Isaac | `Ego`: one vehicle with its sensors and ROS side |
| `aib.runner` | Isaac | one ego in a world, no csb |

The pure modules are what the unit tests cover, under the system Python (3.10) and Isaac's
(3.12) alike.

## The frame

A host owns the stage and the clock. Per frame (20 Hz by default, 3 physics substeps of
1/60 s):

```
ego.take_commands()                 # Autoware's commands that arrived (non-blocking)
for each substep:
    ego.before_substep(dt)          # powertrain torques, steering, written to the wheels
    world.step(render=last substep and a rendering sensor)
    ego.after_substep(physx, dt)    # pose; velocity, yaw rate, acceleration by difference
stamp = ros.publish_clock(t_end)    # once per frame, by the host
ego.publish(t_end, stamp)           # vehicle status, IMU, GNSS (at its rate), lidar (one frame late)
```

Lockstep, as acb follows CARLA's ticks. Two choices kept from csb (roadmap 038), with why:

- **`/clock` offset**: stamps are sim time plus the wall clock at process start, so a
  restarted simulator never sends a long-lived Autoware stack's time backwards (which clears
  its TF buffers and EKF for good).
- **The lidar one frame late**: a scan sent with its own frame's `/clock` races the EKF pose
  of that stamp to NDT, which then cannot interpolate and flaps MRM.

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
spec is csb's "plausible real Model 3" (csb roadmap 043, "As built: vehicle").

## Powertrain

`aib.powertrain` is the seam between Autoware's longitudinal command and the wheels:
`torques(cmd, speed, dt) -> (drive per wheel, (brake front, brake rear))`, chosen by the
spec's `powertrain.model`.

| model | status | what |
|---|---|---|
| `ideal` | implemented, default | acb's contract on CARLA: the force `m a + F_resist(v)` (wheel inertia included) as equal drive torque on four wheels, capped; a negative force is braking only, split by the dynamic axle load (ideal distribution); hold brake when standing with no request or in park |
| `physx` | reserved | PhysX DriveStandard (engine torque curve, gears, clutch, differential); the command mapped to throttle/brake through accel and brake maps, as a real vehicle interface does |
| `ev` | reserved | an electric drive unit per axle with a torque-speed envelope, regeneration blended with the friction brakes, actuator lag |
| `fmu` | reserved | an FMI 2.0 co-simulation unit (a supplier's powertrain model) stepped each substep |

A spec naming a reserved model fails at load (`NotImplementedError`), so nothing silently
drives a different car. A new model is a `Powertrain` subclass registered in `make`; it may
keep state (`reset()` is called on teleport). Models other than `ideal` will need more spec
keys (maps, torque curves, an FMU path) under `powertrain:`; the spec format grows with them.

## Sensors

| type | Isaac | message | notes |
|---|---|---|---|
| lidar | RTX lidar (`isaacsim.sensors.experimental.rtx`), the rig's `isaac.config` / `variant` | PointCloud2 x y z intensity return_type channel | channel from the elevation's rank among `channels`; one frame late |
| camera | USD camera + `rgb` annotator, focal length from the horizontal FOV | Image rgb8 + CameraInfo (K from the spec) | off in the shipped rig; untested on Isaac |
| imu | from the state | Imu: orientation, angular velocity, specific force (acceleration minus gravity) in the IMU frame | every frame unless `rate_hz` |
| gnss | from the state | NavSatFix, GBAS fix, lat/lon about the map origin (TransverseMercator at 0, 0) | `rate_hz` (10) |

Frames and topics are the spec's; for the Model 3 they are acb's (`top`, `carla/imu_link`,
`carla/gnss_link`), so Autoware's sensing launch and csb's existing runs see what they see on
CARLA. The camera publishes in `camera6/camera_optical_link` (acb publishes in
`camera6/camera_link`).

## Single source of truth

The spec is the only file a person edits to change the vehicle. Generated files start with
`Generated from <spec> by aib.gen ...: do not edit.`; `just check-generated` (run by
`just build` and `just test`, and by the unit test `test_gen.CheckTest`) fails when any
output differs from what the spec gives.

What is not generated, and why:

- `simulator_model.param.yaml` (Autoware's planning simulator only), the sensor kit's
  launch files (Autoware's pipeline; a unit test checks that their input topics are the
  rig's), `vehicle.xacro` and `sensors.xacro` (they read the generated yaml).
- acb's `vehicle_config.yaml` and its vehicle description are still acb's own. They can be
  generated from the same spec later (the spec already has the geometry, the sensor mounts
  and the camera/lidar intrinsics acb's config carries), which would make one spec drive the
  same car on CARLA and on Isaac. Not done yet.

## Standalone runner

`aib.runner` (`just isaac-ego`) is the smallest host: Isaac Sim, a world (a USD stage with
its own colliders, or a ground plane), one ego at a pose, fixed frames forever. It is what
acb's standalone mode is to acb: Autoware drives a car with no scenario around it. No NPCs,
no traffic lights, no scenario protocol: those are csb's.
