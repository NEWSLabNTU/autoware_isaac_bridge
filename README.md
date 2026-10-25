# Autoware Isaac Bridge

An [Autoware](https://autowarefoundation.github.io/autoware-documentation/) ego on
[Isaac Sim](https://developer.nvidia.com/isaac/sim): a PhysX vehicle Autoware drives, the
sensors it sees through, and the vehicle interface between them. The Isaac Sim counterpart of
[autoware_carla_bridge](https://github.com/NEWSLabNTU/autoware_carla_bridge) (acb).

The vehicle is described once, in a vehicle spec
([sim_vehicle_spec](https://github.com/NEWSLabNTU/sim_vehicle_spec), a submodule at
`deps/sim_vehicle_spec`: `vehicles/tesla_model3.yaml`). The Isaac rig the PhysX vehicle and
sensors are built from is generated from it here; Autoware's packages for the same car
(`tesla_model3_description`, `tesla_model3_sensor_kit_launch`, ...) are generated from it in
[autoware_sim_vehicle](https://github.com/NEWSLabNTU/autoware_sim_vehicle) (a submodule at
`deps/autoware_sim_vehicle`), and CARLA's files and the SSv2 catalog in sim_vehicle_spec
itself. Every generated file names the spec's `spec_hash`.

## What is in it

| | |
|---|---|
| `deps/sim_vehicle_spec/` | The vehicle specs (submodule): geometry, mass and centre of mass, wheels, tires, brakes, powertrain, steering, suspension, PhysX settings (`isaac.physx`), sensor mounts and intrinsics, and where each number comes from (its `docs/spec-format.md`) |
| `deps/autoware_sim_vehicle/` | Autoware's packages per car and the Autoware profile (submodule) |
| `deps/sim_vehicle_ecu/` | The car's onboard computers (submodule): the drive unit's controller, the brake module (EBD, ABS, traction control, ESC) and the power steering, on the car's own sensors; `aib.isaac.vehicle` steps them every physics step |
| `src/aib/` | The Python library. Pure (no Isaac, no ROS): the rig generator (`aib.gen`), the PhysX vehicle's parameters, the steering linkage, sensor maths, simulation time (`aib.sim_time`). `aib.ros`: Autoware's vehicle interface and the sensor topics (rclpy). `aib.isaac`: the PhysX vehicle, the RTX lidar, camera, IMU and GNSS, and the `Ego` that ties them to a world. `aib.runner`: one ego in an Isaac world, standalone. `aib.steptest`: open-loop step tests of the vehicle in Isaac |
| `src/aib/config/` | `tesla_model3.rig.yaml`, the Isaac rig (generated from the spec: do not edit), and `ego.yaml`, an ego run of it (the rig, the ROS domain, the node name, hand overrides) |
| `src/aib_launch/` | `isaac_simulator.launch.xml`: Autoware for the Isaac ego. It is autoware_sim_vehicle's profile with nothing added (`vehicle_model:=tesla_model3`, `sensor_model:=tesla_model3_sensor_kit`) |

### How it relates to acb and swb

- **acb** is the same thing on CARLA: CARLA owns the car, acb_bridge (Rust) publishes its
  sensors and status and applies Autoware's control. aib is the Isaac side's equivalent.
  Both run the same Autoware profile, `autoware_sim_profile` from
  [autoware_sim_vehicle](https://github.com/NEWSLabNTU/autoware_sim_vehicle), on the same
  car's packages: acb's `carla_simulator.launch.xml` is the profile plus `acb_bridge`, aib's
  `isaac_simulator.launch.xml` is the profile with nothing added. Topic names and frames are
  the same, so the same Autoware pipeline consumes them.
- **[scenario-world-bridge](https://github.com/NEWSLabNTU/scenario-world-bridge)** (swb)
  runs scenario_simulator_v2 scenarios on CARLA or Isaac Sim. Each simulator is a world
  server behind swb's world protocol; the Isaac one,
  [swb_isaac_world](https://github.com/NEWSLabNTU/swb_isaac_world), is a lockstep world
  (the world, the NPCs) whose ego is this library. It takes the ego's body for its spawn
  reply (box, wheels, `spec_hash`) from the rig.
- Unlike acb, the bridge cannot be a separate ROS node: the sensors render in the process
  that steps the world, and that process runs Isaac Sim's own Python (3.12, its bundled
  rclpy). So the library is imported by the host that owns the world -- swb_isaac_world, or
  `aib.runner`.

## Quick start

Prerequisites: Ubuntu 22.04, ROS 2 Humble, Autoware 1.5.0 (`/opt/autoware/1.5.0`), Isaac Sim
6.1 installed with pip into a venv (`ISAAC_ROOT`, default `/mnt/disk1/isaac-sim`), and
`git submodule update --init --recursive` for `deps/sim_vehicle_spec`, `deps/sim_vehicle_ecu` and
`deps/autoware_sim_vehicle` (the spec; the car's Autoware packages, the profile, the pilot and
the vehicle check, which `just build` builds).

```bash
just setup      # once: Autoware's messages rebuilt for Isaac's Python
just build      # colcon build --symlink-install against Autoware (checks the generated files first)
just test       # unit tests (no Isaac needed), the generated-file check, the URDF check
just test-isaac # the same pure tests under Isaac's Python 3.12
```

Drive it, standalone (no scenario runner):

```bash
# terminal 1: the Isaac side -- the world, the ego, its sensors and vehicle interface
DISPLAY=:8 just isaac-ego world:=/path/to/Town01_physics.usda pose:=x,y,z,yaw \
    signals:=/path/to/maps/Town01
# terminal 2: Autoware on the ego's domain (1)
ROS_DOMAIN_ID=1 just autoware map_path:=/path/to/maps/Town01
# terminal 3: drive to a goal (or set one in RViz)
ROS_DOMAIN_ID=1 ros2 run autoware_sim_pilot auto_drive --ros-args -p poses_file:=<poses.yaml>
```

Without `world:=` the ego stands on a flat ground plane, which is enough to check the
vehicle and sensors but not to localize against a map. `isaac-ego` also takes `ego:=<yaml>`
(default `src/aib/config/ego.yaml`), `realtime:=true` and `duration:=<s>`.

`signals:=<map dir>` gives the drive traffic lights: nothing commands them without a
scenario, so the runner cycles them as CARLA does, a junction's lights green in turn (10 s
green, 3 s amber: `green_s:=`, `amber_s:=`), and publishes them to Autoware over V2X. It needs
CARLA's light groups in `<map dir>/carla/signal_groups.yaml`, written once per town from
CARLA holding it by scenario-world-bridge's `scripts/generate_signal_groups.py <Town> --map
<map dir> --port <CARLA port>`. Without `signals:=` Autoware stops at every signalised stop
line, its light unknown (docs/design.md, Standalone runner).

`just autoware` passes every `name:=value` to the profile, after its `check_launch_args`
refuses the arguments roadmap 051 renamed (`carla_control:=`, ... are now `sim_control:=`,
...; `vehicle_model:=aib_vehicle` is now `tesla_model3`), naming the new one: ROS launch would
otherwise drop an argument no file declares without a word.

Check the car without Autoware or a scenario:

```bash
just vehicle-check [ego:=<yaml>] [out:=<dir>]
```

It starts the runner headless on a ground plane (`realtime:=true`), waits for the vehicle
interface on the ego config's ROS domain, then runs autoware_sim_vehicle's `vehicle_check`
(`autoware_sim_check`): the contract (topics, types, rates, frames, stamps,
`/robot_description`) and the dynamics run (acceleration and brake steps against
0.35 m/s^2, a steering sweep against the bicycle model). It writes `<out>/report.json`
(default `play_log/vehicle_check/<time>/`), prints PASS or FAIL with what failed, and exits 0
only on PASS.

### Time

The ego stamps `/clock` and every message with `sim_ns = nanos(sim seconds) + BASE_NS`
(`aib.sim_time`; `BASE_NS` is 1000 s, `nanos` rounds half away from zero), the integer every
part of a run computes from the same simulation time (scenario-world-bridge roadmap 051).
There is no clock setting: the ego config's old `clock_offset` is gone, and a config naming
it is refused at load. Time is monotonic within an episode only. A restarted runner or world
server starts it again, and the profile's `clock_guard` then ends the Autoware stack, which
comes back on the new episode. swb_isaac_world keeps its time running over its whole
lifetime, so a world load there does not end an Isaac stack (on CARLA it does).

### Changing the vehicle

Change the spec in sim_vehicle_spec (and regenerate there and in autoware_sim_vehicle), move
the `deps/` pins to it, then regenerate the rig:

```bash
just generate          # rewrites src/aib/config/<name>.rig.yaml from deps/sim_vehicle_spec
just check-generated   # what CI (and `just build`, `just test`) runs: fails when stale
```

The rig carries a `Generated from sim_vehicle_spec/vehicles/... (spec_hash ...) do not edit`
header. A hand edit, or a spec pin moved without `just generate`, makes `check-generated`
fail; a unit test checks the rig places the car and every sensor as Autoware's packages do.

## Status

- The pure library and the rig generator are tested here (`just test`).
  `isaac_simulator.launch.xml` is the profile acb's `carla_simulator.launch.xml` runs, without
  acb's bridge.
- The Isaac code (`aib.isaac`, `aib.runner`) is the ego swb_isaac_world drives in swb's Isaac
  runs. The camera is untested on Isaac (off in the shipped rig, as on CARLA).
- Powertrain: `ev`, the default (the Model 3 RWD drive unit: motor envelope and lag, regen
  blended with lagged friction brakes, ABS, traction control; road loads through the
  physics), and `ideal` (acb's contract: the requested acceleration is realized). Both
  measured in Isaac (`just isaac-steptest`) and driven by Autoware on Town01 through
  `just isaac-ego` + `just autoware` (docs/design.md, Validation). `physx` and `fmu` are
  reserved.
- The standalone runner has driven: `just isaac-ego` on Town01 with `just autoware`
  (CYCLONEDDS_URI and RMW_IMPLEMENTATION the same on both; `realtime:=true`, or the world
  outruns Autoware), autoware_sim_pilot's `auto_drive` to a goal, arrived.

## Documentation

- [docs/design.md](docs/design.md) -- the pieces, the frame loop, time, the SSoT flow, the onboard computers
- [docs/issues/](docs/issues/) -- defects found, how they were confirmed, their fixes
- sim_vehicle_spec's `docs/spec-format.md` -- the vehicle spec, key by key

## License

Apache License 2.0 ([LICENSE](LICENSE)).
