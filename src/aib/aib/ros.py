"""The ego's ROS 2 side: Autoware's vehicle interface and the sensors' topics.

Its own rclpy context on the ego's domain (an explicit `domain_id`, so one process can serve
several domains), one node. Made once per process and kept across egos: a new DDS
participant among some 200 Autoware ones takes seconds to be discovered, and `/clock` stays
continuous between episodes instead of starting at spawn.

| topic | type | frame |
|---|---|---|
| the rig's lidar topics | PointCloud2 (x y z intensity return_type channel) | the sensor's |
| the rig's IMU topics | Imu | the sensor's |
| the rig's GNSS topics | NavSatFix | the sensor's |
| the rig's camera topics (when enabled) | Image, CameraInfo | the sensor's |
| `/vehicle/status/{velocity,steering,gear,control_mode,turn_indicators,hazard_lights}_status` | Autoware | `base_link` |
| `/clock` | Clock | |
| `/perception/traffic_light_recognition/external/traffic_signals` (when the host has signals) | TrafficLightGroupArray | |

Takes `/control/command/{control_cmd,gear_cmd,turn_indicators_cmd,hazard_lights_cmd}` and
serves `/control/control_mode_request` (AUTONOMOUS only, as acb).

Stamps are sim time plus BASE_NS (aib.sim_time; scenario-world-bridge roadmap 051): the
integer every part of a run computes from the same time, clear of zero, where Autoware
misbehaves. Time is monotonic within an episode only; a stack that sees `/clock` go back (a
restarted runner or world server) ends (autoware_sim_profile's clock_guard) and comes back on
the new episode, rather than living on with cleared TF buffers and an EKF that never drives
again (csb roadmap 038).

Runs inside Isaac Sim's Python (3.12, its bundled rclpy) with Autoware's message packages
rebuilt for it (scripts/build_msgs.sh); nothing here is imported by the pure modules.
"""
import logging

from .control import Command, gear_from_command

log = logging.getLogger("aib")

MAX_CALLBACKS_PER_FRAME = 16
DEFAULT_TOPICS = {
    "lidar": "/sensing/lidar/top/pointcloud_before_sync",
    "imu": "/sensing/imu/imu_link/imu_raw",
    "gnss": "/sensing/gnss/gnss_link/nav_sat_fix",
}


class EgoRos:
    def __init__(self, domain=1, rig=None, node_name="isaac_vehicle"):
        import rclpy
        from autoware_control_msgs.msg import Control
        from autoware_vehicle_msgs.msg import (ControlModeReport, GearCommand, GearReport,
                                               HazardLightsCommand, HazardLightsReport,
                                               SteeringReport, TurnIndicatorsCommand,
                                               TurnIndicatorsReport, VelocityReport)
        from autoware_vehicle_msgs.srv import ControlModeCommand
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import (CameraInfo, Image, Imu, NavSatFix, NavSatStatus,
                                     PointCloud2, PointField)

        self.m = dict(Clock=Clock, VelocityReport=VelocityReport, SteeringReport=SteeringReport,
                      GearReport=GearReport, ControlModeReport=ControlModeReport,
                      TurnIndicatorsReport=TurnIndicatorsReport,
                      HazardLightsReport=HazardLightsReport, Imu=Imu, NavSatFix=NavSatFix,
                      NavSatStatus=NavSatStatus, PointCloud2=PointCloud2, PointField=PointField,
                      Image=Image, CameraInfo=CameraInfo)
        self.domain = domain
        self.ctx = Context()
        rclpy.init(context=self.ctx, domain_id=domain)
        self.node = rclpy.create_node(node_name, context=self.ctx)
        self.executor = SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(self.node)
        self.reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.sensor_qos = qos_profile_sensor_data
        n, reliable = self.node, self.reliable
        self.pub = {
            "clock": n.create_publisher(Clock, "/clock", 10),
            "velocity": n.create_publisher(VelocityReport, "/vehicle/status/velocity_status", reliable),
            "steering": n.create_publisher(SteeringReport, "/vehicle/status/steering_status", reliable),
            "gear": n.create_publisher(GearReport, "/vehicle/status/gear_status", reliable),
            "mode": n.create_publisher(ControlModeReport, "/vehicle/status/control_mode", reliable),
            "turn": n.create_publisher(TurnIndicatorsReport, "/vehicle/status/turn_indicators_status", reliable),
            "hazard": n.create_publisher(HazardLightsReport, "/vehicle/status/hazard_lights_status", reliable),
        }
        self._sensor_pubs = {}  # topic -> publisher, shared by egos of the same rig
        self._signals_pub = None  # made on the first signals (aib.signals)
        if rig is not None:
            for s in rig.sensors.values():
                if s.enabled:
                    self.sensor_publisher(s)
        self.reset()

        def on_control(msg):
            self.cmd.acceleration = float(msg.longitudinal.acceleration)
            self.cmd.velocity = float(msg.longitudinal.velocity)
            self.cmd.tire_angle = float(msg.lateral.steering_tire_angle)
            self.cmd.received = True

        def on_gear(msg):
            gear = gear_from_command(msg.command)
            if gear is not None:
                self.cmd.gear = gear

        def on_turn(msg):
            # NO_COMMAND(0) keeps the last; DISABLE 1, LEFT 2, RIGHT 3 as the report's
            if msg.command:
                self.turn = int(msg.command)

        def on_hazard(msg):
            if msg.command:
                self.hazard = int(msg.command)

        def on_mode(req, res):
            res.success = req.mode == 1  # AUTONOMOUS, as acb (manual has no driver here)
            return res

        n.create_subscription(Control, "/control/command/control_cmd", on_control, reliable)
        n.create_subscription(GearCommand, "/control/command/gear_cmd", on_gear, reliable)
        n.create_subscription(TurnIndicatorsCommand, "/control/command/turn_indicators_cmd", on_turn, reliable)
        n.create_subscription(HazardLightsCommand, "/control/command/hazard_lights_cmd", on_hazard, reliable)
        n.create_service(ControlModeCommand, "/control/control_mode_request", on_mode)
        log.info("ego ROS 2 side on domain %d", domain)

    @classmethod
    def from_config(cls, cfg, rig=None):
        return cls(domain=cfg.domain, rig=rig,
                   node_name=cfg.node_name)

    def sensor_publisher(self, s):
        """The publisher(s) for a SensorRig, made on first use: (main, info or None)."""
        kind = {"lidar": "PointCloud2", "imu": "Imu", "gnss": "NavSatFix", "camera": "Image"}[s.type]
        qos = self.sensor_qos if s.type in ("lidar", "camera") else self.reliable
        topic = s.topic or DEFAULT_TOPICS.get(s.type, "")
        if topic not in self._sensor_pubs:
            self._sensor_pubs[topic] = self.node.create_publisher(self.m[kind], topic, qos)
        info = None
        if s.type == "camera":
            it = s.extra.get("info_topic") or topic.rsplit("/", 1)[0] + "/camera_info"
            if it not in self._sensor_pubs:
                self._sensor_pubs[it] = self.node.create_publisher(self.m["CameraInfo"], it, qos)
            info = self._sensor_pubs[it]
        return self._sensor_pubs[topic], info

    def reset(self):
        """A new ego starts from no command: standing, in drive."""
        self.cmd = Command()
        self.turn = 1  # TurnIndicatorsReport.DISABLE
        self.hazard = 1  # HazardLightsReport.DISABLE

    def publish_clock(self, sim_time):
        """`/clock` at `sim_time` (s); returns that stamp."""
        stamp = self.stamp(sim_time)
        self.pub["clock"].publish(self.m["Clock"](clock=stamp))
        return stamp

    def take_commands(self):
        """Run the callbacks that are ready, without waiting: one per `spin_once`, and a frame
        sees a handful (control at 30 Hz, the rest at 10 Hz, against 20 frames a second)."""
        for _ in range(MAX_CALLBACKS_PER_FRAME):
            self.executor.spin_once(timeout_sec=0)

    @staticmethod
    def stamp(sim_time):
        """The ROS time of `sim_time` s: nanos(sim_time) + BASE_NS."""
        from builtin_interfaces.msg import Time

        from .sim_time import sim_ns

        ns = sim_ns(sim_time)
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    def publish_status(self, state, stamp):
        """Autoware's vehicle status for one frame."""
        m, pub = self.m, self.pub
        v = m["VelocityReport"]()
        v.header.stamp = stamp
        v.header.frame_id = "base_link"
        v.longitudinal_velocity = float(state.velocity[0])
        v.lateral_velocity = float(state.velocity[1])
        v.heading_rate = float(state.angular_velocity[2])
        pub["velocity"].publish(v)
        st = m["SteeringReport"]()
        st.stamp = stamp
        st.steering_tire_angle = float(state.tire_angle)
        pub["steering"].publish(st)
        g = m["GearReport"]()
        g.stamp = stamp
        g.report = self.cmd.gear
        pub["gear"].publish(g)
        cm = m["ControlModeReport"]()
        cm.stamp = stamp
        cm.mode = 1  # AUTONOMOUS
        pub["mode"].publish(cm)
        tr = m["TurnIndicatorsReport"]()
        tr.stamp = stamp
        tr.report = self.turn
        pub["turn"].publish(tr)
        hz = m["HazardLightsReport"]()
        hz.stamp = stamp
        hz.report = self.hazard
        pub["hazard"].publish(hz)

    def publish_signals(self, groups, stamp):
        """Autoware's V2X traffic signals (`aib.signals`): `groups`, `[(regulatory element id,
        colour name)]` as the host holds them, at `stamp`."""
        from autoware_perception_msgs.msg import (TrafficLightElement, TrafficLightGroup,
                                                  TrafficLightGroupArray)
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        from . import signals

        if self._signals_pub is None:
            # acb's publisher's QoS (rclrs' default): reliable, depth 10.
            self._signals_pub = self.node.create_publisher(
                TrafficLightGroupArray, signals.TOPIC,
                QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        msg = TrafficLightGroupArray()
        msg.stamp = stamp
        for reg, colour in groups:
            color, shape, status, confidence = signals.element(colour)
            msg.traffic_light_groups.append(TrafficLightGroup(
                traffic_light_group_id=int(reg),
                elements=[TrafficLightElement(color=color, shape=shape, status=status,
                                              confidence=confidence)]))
        self._signals_pub.publish(msg)

    def close(self):
        import rclpy

        self.node.destroy_node()
        rclpy.shutdown(context=self.ctx)
