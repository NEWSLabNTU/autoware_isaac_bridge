# Autoware Isaac Bridge

An [Autoware](https://autowarefoundation.github.io/autoware-documentation/) ego on
[Isaac Sim](https://developer.nvidia.com/isaac/sim): a PhysX vehicle Autoware drives, the
sensors it sees through, and the vehicle interface between them. The Isaac Sim counterpart of
[autoware_carla_bridge](https://github.com/NEWSLabNTU/autoware_carla_bridge) (acb).

The vehicle is described once, in a vehicle spec (`vehicles/tesla_model3.yaml`), and
everything else about it -- Autoware's vehicle description, the sensor kit calibration and
URDF, the Isaac rig the PhysX vehicle and sensors are built from -- is generated from that
file and checked against it.

## What is in it

| | |
|---|---|
| `vehicles/` | Vehicle specs: geometry, mass and centre of mass, wheels, tires, brakes, powertrain, steering, suspension, PhysX settings, sensor mounts and intrinsics, and where each number comes from ([docs/spec-format.md](docs/spec-format.md)) |
| `src/aib/` | The Python library. Pure (no Isaac, no ROS): the spec, the generator, the vehicle parameters, the powertrain seam, the steering, sensor maths. `aib.ros`: Autoware's vehicle interface and the sensor topics (rclpy). `aib.isaac`: the PhysX vehicle, the RTX lidar, camera, IMU and GNSS, and the `Ego` that ties them to a world. `aib.runner`: one ego in an Isaac world, standalone |
| `src/aib_vehicle_launch/` | `aib_vehicle_description` (vehicle_info, mirror, an RViz body) and `aib_vehicle_launch`, Autoware's `vehicle_model:=aib_vehicle` |
| `src/aib_sensor_kit_launch/` | `aib_sensor_kit_description` (calibration, URDF) and `aib_sensor_kit_launch` (Autoware's sensing pipeline for the kit), `sensor_model:=aib_sensor_kit` |
| `src/aib_individual_params/` | The kit's calibration as `individual_params` would hold it (named so it can share a workspace with acb's `individual_params`) |
| `src/aib_launch/` | `isaac_simulator.launch.xml`: Autoware for the Isaac ego |

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

Edit the spec, then regenerate:

```bash
$EDITOR vehicles/tesla_model3.yaml
just generate          # rewrites the files the spec owns
just check-generated   # what CI (and `just build`, `just test`) runs: fails when stale
```

The generated files carry a `Generated from vehicles/... do not edit` header. A hand edit to
one makes `check-generated` fail.

## Status

- The pure library, the generator and the Autoware packages are tested here (`just test`;
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
- [docs/spec-format.md](docs/spec-format.md) -- the vehicle spec, key by key

## License

Apache License 2.0 ([LICENSE](LICENSE)).
