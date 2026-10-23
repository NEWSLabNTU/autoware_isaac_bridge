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
| `src/aib/` | The Python library. Pure (no Isaac, no ROS): the rig generator, the vehicle parameters, the powertrain seam, the steering, sensor maths. `aib.ros`: Autoware's vehicle interface and the sensor topics (rclpy). `aib.isaac`: the PhysX vehicle, the RTX lidar, camera, IMU and GNSS, and the `Ego` that ties them to a world. `aib.runner`: one ego in an Isaac world, standalone |
| `src/aib_launch/` | `isaac_simulator.launch.xml`: Autoware for the Isaac ego (`vehicle_model:=tesla_model3`, autoware_sim_vehicle's packages) |

### How it relates to acb and swb

- **acb** is the same thing on CARLA: CARLA owns the car, acb_bridge (Rust) publishes its
  sensors and status and applies Autoware's control. aib is the Isaac side's equivalent.
  Both include the same Autoware profile, `autoware_sim_profile` from
  [autoware_sim_vehicle](https://github.com/NEWSLabNTU/autoware_sim_vehicle) (a submodule at
  `deps/autoware_sim_vehicle`): `isaac_simulator.launch.xml` is that profile with aib's
  vehicle and sensor models. Topic names and frames are the same, so the same Autoware
  pipeline consumes them.
- **[scenario-world-bridge](https://github.com/NEWSLabNTU/scenario-world-bridge)** (swb,
  carla-scenario-bridge or csb until its roadmap 046) runs scenario_simulator_v2 scenarios
  on CARLA or Isaac Sim. Each simulator is a world server behind swb's world protocol; its
  Isaac world server (`src/isaac_world`) is a lockstep world (the world, the NPCs) whose ego
  is this library (swb roadmap 045).
- Unlike acb, the bridge cannot be a separate ROS node: the sensors render in the process
  that steps the world, and that process runs Isaac Sim's own Python (3.12, its bundled
  rclpy). So the library is imported by the host that owns the world -- swb's Isaac world
  server, or `aib.runner`.

## Quick start

Prerequisites: Ubuntu 22.04, ROS 2 Humble, Autoware 1.5.0 (`/opt/autoware/1.5.0`), Isaac Sim
6.1 installed with pip into a venv (`ISAAC_ROOT`, default `/mnt/disk1/isaac-sim`), and
`git submodule update --init --recursive` for `deps/autoware_sim_vehicle` (the Autoware
profile and pilot, which `just build` builds).

```bash
just setup      # once: Autoware's messages rebuilt for Isaac's Python
just build      # colcon build --symlink-install against Autoware (checks the generated files first)
just test       # unit tests (no Isaac needed), the generated-file check, the URDF check
```

Drive it, standalone (no scenario runner):

```bash
# terminal 1: the Isaac side -- the world, the ego, its sensors and vehicle interface
DISPLAY=:8 just isaac-ego world:=/path/to/Town01_physics.usda pose:=x,y,z,yaw
# terminal 2: Autoware on the ego's domain (1)
ROS_DOMAIN_ID=1 just autoware map_path:=/path/to/maps/Town01
# terminal 3: drive to a goal (or set one in RViz)
ROS_DOMAIN_ID=1 ros2 run autoware_sim_pilot auto_drive --ros-args -p poses_file:=<poses.yaml>
```

Without `world:=` the ego stands on a flat ground plane, which is enough to check the
vehicle and sensors but not to localize against a map.

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

- The pure library and the rig generator are tested here (`just test`;
  `play_launch dump` of `isaac_simulator.launch.xml` resolves the same 205 nodes as acb's
  `carla_simulator.launch.xml` without its bridge).
- The Isaac code (`aib.isaac`, `aib.runner`) is csb_isaac's ego (swb's `isaac_world` now), which drives swb's Isaac
  exit runs, restructured. The camera is new
  and untested on Isaac (off in the shipped rig, as on csb).
- Powertrain: `ev`, the default (the Model 3 RWD drive unit: motor envelope and lag, regen
  blended with lagged friction brakes, ABS, traction control; road loads through the
  physics), and `ideal` (acb's contract: the requested acceleration is realized). Both
  measured in Isaac (`just isaac-steptest`) and driven by Autoware on Town01 through
  `just isaac-ego` + `just autoware` (docs/design.md, Validation). `physx` and `fmu` are
  reserved.
- The standalone runner has driven: `just isaac-ego` on Town01 with `just autoware`
  (CYCLONEDDS_URI and RMW_IMPLEMENTATION the same on both; `realtime:=true`, or the world
  outruns Autoware), acb's pilot to a goal, arrived.

## Documentation

- [docs/design.md](docs/design.md) -- the pieces, the frame loop, the SSoT flow, the powertrain seam
- sim_vehicle_spec's `docs/spec-format.md` -- the vehicle spec, key by key

## License

Apache License 2.0 ([LICENSE](LICENSE)).
