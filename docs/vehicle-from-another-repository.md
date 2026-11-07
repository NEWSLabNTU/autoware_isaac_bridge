# A vehicle from another repository

aib builds an ego from files that name each other. None of them has to live in aib: a
repository that uses swb can keep its own car next to its own Autoware, and aib reads it from
there. Every path below is relative to the file that names it, unless it is absolute;
environment variables in a path expand.

```
<your repository>/
  vehicle/<car>.yaml              # the vehicle spec (sim_vehicle_spec's schema, spec v2)
  isaac/
    <car>.rig.yaml                # generated from the spec by aib.gen; commit it
    ego.yaml                      # rig: <car>.rig.yaml, body: body/<model>/entry.json
    lidars/*.usda                 # the rig's sensors' isaac.usd, if any
    body/<model>/entry.json       # the body the ego wears in the stand-alone runner
```

## The files

- **The spec.** sim_vehicle_spec's schema (`deps/sim_vehicle_spec`). It is the car's single
  source: geometry, mass, wheels, powertrain, the sensors and their mounts.
- **The rig**, generated from the spec:
  ```bash
  python3 -m aib.gen --spec vehicle/<car>.yaml -o isaac/<car>.rig.yaml
  python3 -m aib.gen --spec vehicle/<car>.yaml -o isaac/<car>.rig.yaml --check   # in CI
  ```
  Run it with aib on `PYTHONPATH` (`src/aib` of an aib checkout, or a sourced swb install).
  The rig's `spec:` is the spec's path relative to the rig. Loading the rig refuses a spec
  whose `spec_hash` is not the one the rig was generated from: regenerate after every change
  to the spec. A sensor's `isaac.usd` (an RTX lidar profile of your own) is relative to the
  rig too.
- **The ego config** (aib's `EgoConfig`): `rig:`, and optionally `body:`, `domain:`, the
  overrides (`vehicle`, `ecu`, `sensors`), `control_topic`, `lockstep`, `ground_truth`.
  Without `domain:` the ego's nodes join `ROS_DOMAIN_ID` (0 when unset), so the terminal that
  starts it chooses, as it does for any ROS node.

## Running it

Stand-alone (aib's runner; no SSv2):

```bash
play_launch launch aib_launch isaac_vehicle.launch.xml ego:=<your repository>/isaac/ego.yaml \
    world:=<map>/isaac/world.yaml map_pose:=x,y,z,yaw
```

Under SSv2 (swb's Isaac world server): its `ego_config` launch argument takes the same ego
config. The bridge's spec check then also needs the car's SSv2 catalog entry and the vehicle
agent's spec_hash; scenario-world-bridge's `docs/design/vehicle-side.md` lists those slots.
