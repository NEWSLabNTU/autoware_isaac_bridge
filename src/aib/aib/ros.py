"""The ego's ROS 2 side: Autoware's vehicle interface and the sensors' topics.

Its own rclpy context on the ego's domain (an explicit `domain_id`, so one process can serve
several domains), one node. Made once per process and kept across egos: a new DDS
participant among some 200 Autoware ones takes seconds to be discovered. `/clock` is the
host's simulation time plus `BASE_NS` (`aib.sim_time`); it is monotonic only within an
episode, and the Autoware profile's `clock_guard` ends the stack if it ever goes back.

| topic | type | frame |
|---|---|---|
| the rig's lidar topics | PointCloud2 (x y z intensity return_type channel) | the sensor's |
| the rig's IMU topics | Imu | the sensor's |
| the rig's GNSS topics | NavSatFix | the sensor's |
| the rig's camera topics (when enabled) | Image, CameraInfo | the sensor's |
| `/vehicle/status/{velocity,steering,gear,control_mode,turn_indicators,hazard_lights}_status` | Autoware | `base_link` |
| `/clock` | Clock | |
| `/perception/traffic_light_recognition/external/traffic_signals` (when the host has signals) | TrafficLightGroupArray | |

Takes `/control/command/{control_cmd,gear_cmd,turn_indicators_cmd,hazard_lights_cmd}` (the
Control from the ego config's `control_topic`, by default `/control/command/control_cmd`) and
serves `/control/control_mode_request` (AUTONOMOUS only, as acb).

With `lockstep` (the ego config's; swb roadmap 057) the world waits for the ego's localizer:
before a frame takes its commands, `take_commands` blocks until the localizer has published a
pose for the last lidar scan sent (a `PoseWithCovarianceStamped` on `lockstep.topic` stamped at
the scan, as Autoware's NDT stamps its pose), at most `lockstep.timeout_s`. The simulation then
runs as fast as the host lets the localizer keep up, as on a car whose computer is sized for its
speed; on a busy host it slows instead of the localizer skipping scans. The wait arms on the
localizer's first pose (no waiting before it localizes) and disarms after `max_misses` waits
in a row time out (a localizer that stopped), until the next pose.

Stamps are sim time plus BASE_NS (aib.sim_time; scenario-world-bridge roadmap 051): the
integer every part of a run computes from the same time, clear of zero, where Autoware
misbehaves. Time is monotonic within an episode only; a stack that sees `/clock` go back (a
restarted runner or world server) ends (autoware_sim_profile's clock_guard) and comes back on
the new episode, rather than living on with cleared TF buffers and an EKF that never drives
again (swb roadmap 038).

Runs inside Isaac Sim's Python (3.12, its bundled rclpy) with Autoware's message packages
rebuilt for it (scripts/build_msgs.sh); nothing here is imported by the pure modules.
"""
import logging

from .control import Command, gear_from_command

log = logging.getLogger("aib")

MAX_CALLBACKS_PER_FRAME = 16
LOCKSTEP_DEFAULTS = {"topic": "/localization/pose_estimator/pose_with_covariance",
                     "timeout_s": 1.0, "max_misses": 3}
LOCKSTEP_REPORT_S = 30.0  # wall seconds between the lockstep's log lines
DEFAULT_TOPICS = {
    "lidar": "/sensing/lidar/top/pointcloud_before_sync",
    "imu": "/sensing/imu/imu_link/imu_raw",
    "gnss": "/sensing/gnss/gnss_link/nav_sat_fix",
}


class EgoRos:
    def __init__(self, domain=1, rig=None, node_name="isaac_vehicle",
                 control_topic="/control/command/control_cmd", lockstep=None):
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

        n.create_subscription(Control, control_topic, on_control, reliable)
        n.create_subscription(GearCommand, "/control/command/gear_cmd", on_gear, reliable)
        n.create_subscription(TurnIndicatorsCommand, "/control/command/turn_indicators_cmd", on_turn, reliable)
        n.create_subscription(HazardLightsCommand, "/control/command/hazard_lights_cmd", on_hazard, reliable)
        n.create_service(ControlModeCommand, "/control/control_mode_request", on_mode)
        self.lockstep = None
        if lockstep:
            unknown = set(lockstep) - set(LOCKSTEP_DEFAULTS)
            if unknown:
                raise ValueError(f"lockstep: unknown key(s) {sorted(unknown)}")
            self.lockstep = {**LOCKSTEP_DEFAULTS, **lockstep}
            from geometry_msgs.msg import PoseWithCovarianceStamped

            def on_pose(msg):
                ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                self.localized_ns = max(self.localized_ns or 0, ns)
                if not self.armed:
                    self.armed = True
                    log.info("lockstep: armed (the localizer's pose at %d ns)", ns)

            n.create_subscription(PoseWithCovarianceStamped, self.lockstep["topic"], on_pose,
                                  reliable)
            log.info("lockstep: each frame waits up to %.2f s for a pose on %s",
                     self.lockstep["timeout_s"], self.lockstep["topic"])
        self._reset_lockstep()
        log.info("ego ROS 2 side on domain %d", domain)

    @classmethod
    def from_config(cls, cfg, rig=None):
        return cls(domain=cfg.domain, rig=rig, node_name=cfg.node_name,
                   control_topic=cfg.control_topic, lockstep=cfg.lockstep)

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
        self._reset_lockstep()

    def _reset_lockstep(self):
        """A new ego, or a new episode (time may start again): nothing sent, nothing matched."""
        self.scan_ns = None  # the stamp of the last lidar scan sent
        self.localized_ns = None  # the newest stamp the localizer published a pose for
        self.armed = False
        self.misses = 0
        self.stats = {"frames": 0, "waited": 0.0, "max": 0.0, "timeouts": 0}
        self.reported_at = None

    def note_scan(self, stamp):
        """A lidar scan went out at `stamp` (builtin_interfaces Time)."""
        self.scan_ns = stamp.sec * 1_000_000_000 + stamp.nanosec

    def wait_localized(self):
        """The lockstep: block until the localizer has a pose for the last scan sent, at most
        `timeout_s`. Returns the seconds waited."""
        import time

        ls = self.lockstep
        if not ls or not self.armed or self.scan_ns is None:
            return 0.0
        start = time.perf_counter()
        deadline = start + ls["timeout_s"]
        while (self.localized_ns or 0) < self.scan_ns:
            left = deadline - time.perf_counter()
            if left <= 0:
                self.misses += 1
                self.stats["timeouts"] += 1
                if self.misses >= ls["max_misses"]:
                    self.armed = False
                    log.warning("lockstep: no pose for %d frames in a row; disarmed until the "
                                "localizer publishes again", self.misses)
                break
            self.executor.spin_once(timeout_sec=min(left, 0.005))
        else:
            self.misses = 0
        waited = time.perf_counter() - start
        st = self.stats
        st["frames"] += 1
        st["waited"] += waited
        st["max"] = max(st["max"], waited)
        now = time.perf_counter()
        if self.reported_at is None:
            self.reported_at = now
        elif now - self.reported_at >= LOCKSTEP_REPORT_S:
            log.info("lockstep: %d frames, waited %.1f ms a frame (max %.0f ms), %d timed out",
                     st["frames"], 1e3 * st["waited"] / st["frames"], 1e3 * st["max"],
                     st["timeouts"])
            self.stats = {"frames": 0, "waited": 0.0, "max": 0.0, "timeouts": 0}
            self.reported_at = now
        return waited

    def publish_clock(self, sim_time):
        """`/clock` at `sim_time` (s); returns that stamp."""
        stamp = self.stamp(sim_time)
        self.pub["clock"].publish(self.m["Clock"](clock=stamp))
        return stamp

    def take_commands(self):
        """Run the callbacks that are ready, without waiting: one per `spin_once`, and a frame
        sees a handful (control at 30 Hz, the rest at 10 Hz, against 20 frames a second). With
        `lockstep`, first wait for the localizer (`wait_localized`); with several egos the
        later calls find it done."""
        self.wait_localized()
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
