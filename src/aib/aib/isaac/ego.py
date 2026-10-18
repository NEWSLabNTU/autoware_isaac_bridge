"""One ego on Isaac: its PhysX vehicle, its sensors and its ROS 2 side. Import after
`SimulationApp`.

What acb_bridge is to a CARLA vehicle, this is to an Isaac one, in the process that steps
the world (the sensors render there). A host -- csb's lockstep world server, or aib's own
runner (`aib.runner`) -- owns the stage and the clock and calls, every frame:

    ego.take_commands()                    # Autoware's commands that arrived
    for each physics substep:
        ego.before_substep(dt)             # apply them (powertrain, steering)
        world.step(...)                    # the host's physics (and the last substep's render)
        ego.after_substep(physx, dt)       # read the body back
    stamp = ros.publish_clock(end)         # the host's /clock, once per frame
    ego.publish(end, stamp)                # vehicle status and sensors

Lockstep, as acb follows CARLA's ticks: the commands that arrived before a frame drive it.
"""
import logging

from ..control import State
from . import sensors as sensor_kinds
from .vehicle import Vehicle

log = logging.getLogger("aib")

WARM_UP_FRAMES = 60


class Ego:
    def __init__(self, rig, ros, stage, path, pose, name="ego", box_center=None, box_dims=None,
                 params=None):
        """`rig`: an aib.rig.Rig; `ros`: an aib.ros.EgoRos (made once per process); `path`:
        the prim path of the vehicle; `pose`: ((x, y, z), (w, x, y, z)) of base_link.
        `box_center`/`box_dims` override the rig's chassis box (a scenario's declared box),
        `params` its VehicleParams."""
        self.rig = rig
        self.name = name
        self.path = path
        self.ros = ros
        self.vehicle = Vehicle(stage, path, params or rig.vehicle,
                               box_center or rig.box_center, box_dims or rig.box_dims, pose)
        self.state = State()
        self.sensors = [sensor_kinds.make(s, ros, stage, path)
                        for s in rig.sensors.values() if s.enabled]
        ros.reset()

    @property
    def renders(self):
        """Whether a frame must render for this ego's sensors."""
        return any(s.renders for s in self.sensors)

    def warm_up(self, step, frames=WARM_UP_FRAMES):
        """Step (physics and a render) until the rendering sensors deliver: their first
        frames set up the RTX pipeline, seconds on a loaded host, which belong in the spawn
        rather than in the first frame's step. The vehicle stands on its hold brake meanwhile
        (no command yet), which also lets its suspension settle."""
        if not self.renders:
            return 0
        for i in range(frames):
            step()
            if all(s.ready() for s in self.sensors):
                return i + 1
        log.warning("ego %r: sensors not ready after %d warm-up frames", self.name, frames)
        return frames

    # --- the frame -------------------------------------------------------------------------

    def take_commands(self):
        self.ros.take_commands()

    def before_substep(self, dt):
        self.vehicle.apply(self.ros.cmd, dt, self.state)

    def after_substep(self, physx, dt):
        self.vehicle.sample(physx, dt, self.state)

    def teleport(self, pose, physx):
        self.vehicle.teleport(pose, physx)

    def publish(self, sim_time, stamp):
        """Everything Autoware gets for this frame but `/clock` (the host's), at `stamp`."""
        self.ros.publish_status(self.state, stamp)
        for s in self.sensors:
            s.publish(self.state, sim_time, stamp)

    def close(self):
        for s in self.sensors:
            s.close()
