# Vehicle spec format

One YAML file per vehicle under `vehicles/`, validated strictly by `aib.spec` (an unknown
key, a missing required key or a malformed mount is an error naming the file and the key).
Units: metres, kilograms, seconds, radians, newton-metres, unless a key says otherwise.
Frame: base_link, the rear axle centre on the ground, x forward, y left, z up.

## Top level

| key | required | |
|---|---|---|
| `spec_version` | yes | `1` |
| `name` | yes | the rig is written as `src/aib/config/<name>.rig.yaml` |
| `description` | no | one line |
| `autoware` | yes | `vehicle_model`, `sensor_model`: Autoware's package names (`<vehicle_model>_description`, `<sensor_model>_launch`, ...); the generator writes into `src/<vehicle_model>_launch/<vehicle_model>_description` and `src/<sensor_model>_launch/<sensor_model>_description` |
| `body` | no | `catalogue_model`: the visual body a host may put on the vehicle (a CARLA blueprint id in csb's vehicle catalogue) |
| `geometry` | yes | below |
| `mirror` | yes | Autoware's `mirror.param.yaml` keys, as is |
| `inertia` | yes | below |
| `wheels`, `tires`, `brakes`, `powertrain`, `resistance`, `steering`, `suspension`, `physx` | see below | |
| `sensor_kit` | yes | `mount`: base_link -> sensor_kit_base_link |
| `sensors` | no | name -> sensor, below |
| `provenance` | no | where numbers come from, below |

## Geometry -> vehicle_info

All required, written to `vehicle_info.param.yaml` as is: `wheel_base`, `wheel_tread`,
`wheel_radius`, `wheel_width`, `front_overhang`, `rear_overhang`, `left_overhang`,
`right_overhang`, `vehicle_height`, `max_steer_angle` (the bicycle-model tire angle limit
Autoware plans with; the PhysX vehicle clamps its request to it). The chassis box is the
footprint and height these give.

## Physics -> the rig's `vehicle`

| key | rig field | |
|---|---|---|
| `inertia.mass` | `mass` | kg |
| `inertia.com_front_fraction` | `com[0] = fraction x wheel_base` | share of the static load on the front axle, (0, 1) |
| `inertia.com_lateral` | `com[1]` | default 0 |
| `inertia.com_height` | `com[2]` | above the ground |
| `inertia.inertia_box` | `inertia_box` | the body's inertia as a uniform box of these l w h |
| `wheels.mass`, `wheels.damping_rate` | `wheel_mass`, `wheel_damping_rate` | moment of inertia 0.5 m r^2 |
| `tires.friction` | `tire_friction` | each wheel's friction table default (a host's surface layer may write another per wheel) |
| `tires.lat_stiff_max_load`, `lat_stiff_value`, `long_stiff_value`, `friction_vs_slip` | same | PhysX's tire model: lateral stiffness graph (rest loads, N/rad per N of rest load), longitudinal stiffness per g, friction vs slip points |
| `brakes.max_brake_torque`, `hold_brake_torque` | same | per wheel |
| `brakes.distribution` | `brake_distribution` | `dynamic_axle_load` (ideal / EBD) or `static` |
| `brakes.dead_time`, `time_constant` | `brake_dead_time`, `brake_time_constant` | `ev`: the hydraulics, a pure delay then a first-order pressure lag, s |
| `brakes.abs.{enabled, slip_release, slip_reapply, release_rate, apply_rate, min_speed}` | `abs_*` | `ev`: per wheel, the pressure factor falls at `release_rate` (1/s) while the wheel's slip is past `slip_release`, climbs at `apply_rate` once within `slip_reapply`; off below `min_speed` m/s |
| `powertrain.model` | `powertrain` | `ideal` or `ev`; `physx`, `fmu` reserved (design.md) |
| `powertrain.max_drive_torque` | same | `ideal`'s cap, per wheel, N m |
| `powertrain.drive_axle`, `front_drive_share` | same | `ev`: `rear`, `front` or `all` (then the front share of motor torque); each axle an open differential |
| `powertrain.gear_ratio` | same | `ev`, required: motor to wheel, single speed |
| `powertrain.motor_peak_torque`, `motor_peak_power`, `motor_max_rpm` | same | `ev`, required: the envelope, N m at the motor to base speed, then W; torque tapers to 0 over the last 5 % below the max rpm |
| `powertrain.motor_time_constant` | same | `ev`: first-order torque lag, s |
| `powertrain.regen_max_torque`, `regen_max_power` | same | `ev`: regeneration's limits, N m at the motor and W |
| `powertrain.regen_fade_speeds` | `regen_fade_low`, `regen_fade_high` | `ev`: `[low, high]` m/s, no regen at or below `low`, full at or above `high` |
| `powertrain.traction_control.{enabled, slip_limit, slip_reapply, release_rate, apply_rate}` | `tc_*` | `ev`: the motor torque factor on the driven wheels' largest slip, as ABS |
| `resistance.rolling_coefficient`, `drag_coefficient`, `frontal_area`, `air_density` | same | the road load, applied to the body by the physics (all models): Crr m g, rho Cd A v^2 / 2. No section: no road load |
| `steering.ackermann` | `ackermann` | split the bicycle angle into Ackermann wheel angles (default true) |
| `steering.time_constant`, `rate_limit` | `steer_time_constant`, `steer_rate_limit` | actuator lag, rate limit; 0: none |
| `suspension.frequency`, `damping_ratio`, `travel` | `suspension_*` | per axle on its sprung mass |
| `physx.substeps` | `physx_substeps` | vehicle substeps per physics step, at every speed |
| `physx.*` (others) | same name | slip denominators, sticky tire thresholds and damping |

An `ev` spec must give `gear_ratio`, `motor_peak_torque`, `motor_peak_power` and
`motor_max_rpm`; the other `ev` keys default to the shipped spec's. `ideal` ignores them, so
one spec carries both and `powertrain.model` (or an ego config's `vehicle: {powertrain: ev}`)
picks. `powertrain.rolling_resistance` (a force folded into `ideal`'s command, with no
physical counterpart) is gone: a spec naming it is refused with a pointer to `resistance`.

Every rig field is a `VehicleParams` field (`src/aib/aib/params.py`), so an ego config's
`vehicle:` override uses the same names.

## Sensors

```yaml
sensors:
  <name>:
    type: lidar | camera | imu | gnss
    calibration_key: <key>   # its key in sensor_kit_calibration.yaml (default: link)
    link: <urdf link>        # the link the mount places, child of sensor_kit_base_link
    frame: <frame_id>        # the messages' frame; a child of link (identity, or the
                             # optical rotation for a camera) when it differs
    topic: <topic>
    mount: {x, y, z, roll, pitch, yaw}   # in sensor_kit_base_link, URDF rpy
    isaac: {enabled: true, ...}          # Isaac options; enabled: false keeps it out of the rig's run
    # lidar:  channels, rate_hz, range, vertical_fov
    # camera: width, height, horizontal_fov (deg), rate_hz, info_topic
    # imu, gnss: rate_hz
```

The generator writes the mounts to `sensor_kit_calibration.yaml` (under the calibration
key), a link and frame per sensor to `sensor_kit.xacro`, and the rig's sensors with
base_link transforms composed from the kit's mount and the sensor's. A camera's rig entry
also carries `k`, the pinhole matrix from its width, height and horizontal FOV.

## Provenance

```yaml
provenance:
  <section or section.key>: {basis: "<where it comes from>", assumed: true}
```

`basis` is required; `assumed: true` marks an estimate to replace with a measurement. The
rig lists the assumed keys under `assumptions`, so a host can log what is not measured.
