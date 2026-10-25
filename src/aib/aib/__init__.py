"""autoware_isaac_bridge: an Autoware ego on Isaac Sim.

Pure modules (no Isaac, no ROS): gen, params, control, frames, sensing, rig, deps. The car's
onboard computers (powertrain, brakes, ESC, steering actuator) are sim_vehicle_ecu's
(deps/sim_vehicle_ecu).
`aib.ros` needs rclpy; `aib.isaac` and `aib.runner` need Isaac Sim.
"""
__version__ = "0.1.0"
