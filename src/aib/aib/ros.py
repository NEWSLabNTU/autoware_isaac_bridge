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
| `ground_truth.topic` (the ego config's, when on; scensim roadmap 063) | Odometry, base_link in the map | `map` -> `base_link` |
| NDT's pose topic (with `ground_truth.pose_estimate`) | PoseWithCovarianceStamped, the same pose | `map` |
| `/perception/traffic_light_recognition/external/traffic_signals` (when the host has signals) | TrafficLightGroupArray | |

Takes `/control/command/{control_cmd,gear_cmd,turn_indicators_cmd,hazard_lights_cmd}` (the
Control from the ego config's `control_topic`, by default `/control/command/control_cmd`) and
serves `/control/control_mode_request` (AUTONOMOUS only, as acb).

With `lockstep` (the ego config's; scensim roadmap 057) the world waits for the ego's localizer:
before a frame takes its commands, `take_commands` blocks until the localizer has published a
pose for the last lidar scan sent (a `PoseWithCovarianceStamped` on `lockstep.topic` stamped at
the scan, as Autoware's NDT stamps its pose), at most `lockstep.timeout_s`. The simulation then
runs as fast as the host lets the localizer keep up, as on a car whose computer is sized for its
speed; on a busy host it slows instead of the localizer skipping scans. The wait arms on the
localizer's first pose (no waiting before it localizes) and disarms after `max_misses` waits
in a row time out (a localizer that stopped), until the next pose.

Stamps are sim time plus BASE_NS (aib.sim_time; scensim roadmap 051): the
integer every part of a run computes from the same time, clear of zero, where Autoware
misbehaves. Time is monotonic within an episode only; a stack that sees `/clock` go back (a
restarted runner or world server) ends (autoware_sim_profile's clock_guard) and comes back on
the new episode, rather than living on with cleared TF buffers and an EKF that never drives
again (scensim roadmap 038).

Runs inside Isaac Sim's Python (3.12, its bundled rclpy) with Autoware's message packages
rebuilt for it (scripts/build_msgs.sh); nothing here is imported by the pure modules.
"""
import logging

from .control import Command, gear_from_command
from .interfaces import interface_settings

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
                 control_topic="/control/command/control_cmd", lockstep=None, ground_truth=None,
                 vehicle_interface=None):
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
        from sensor_msgs.msg import (CameraInfo, Image, Imu, LaserScan, NavSatFix, NavSatStatus,
                                     PointCloud2, PointField)

        self.m = dict(Clock=Clock, VelocityReport=VelocityReport, SteeringReport=SteeringReport,
                      GearReport=GearReport, ControlModeReport=ControlModeReport,
                      TurnIndicatorsReport=TurnIndicatorsReport,
                      HazardLightsReport=HazardLightsReport, Imu=Imu, NavSatFix=NavSatFix,
                      NavSatStatus=NavSatStatus, PointCloud2=PointCloud2, PointField=PointField,
                      Image=Image, CameraInfo=CameraInfo, LaserScan=LaserScan)
        self.domain = domain
        self.ctx = Context()
        rclpy.init(context=self.ctx, domain_id=domain)
        self.node = rclpy.create_node(node_name, context=self.ctx)
        self.executor = SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(self.node)
        self.reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.sensor_qos = qos_profile_sensor_data
        n, reliable = self.node, self.reliable
        self.interface = interface_settings(vehicle_interface)
        self.profile = self.interface["profile"]
        self.now = 0.0  # the host's simulation time, s: the commands' arrival (aib.interfaces)
        if self.profile != "autoware":
            if lockstep:
                raise ValueError(f"lockstep waits for Autoware's localizer; the {self.profile} "
                                 "vehicle interface has none: drop the lockstep")
            self.pub = {"clock": n.create_publisher(Clock, "/clock", 10)}
            self._sensor_pubs = {}
            self._signals_pub = None
            if rig is not None:
                for s in rig.sensors.values():
                    if s.enabled and s.type != "imu":  # the IMU is the profile's (imu_sender)
                        self.sensor_publisher(s)
            self._make_interface(rig)
            self.lockstep = None
            self.reset()
            self.ground_truth = {}
            if ground_truth:
                self._make_ground_truth(ground_truth)
            log.info("ego ROS 2 side on domain %d, the %s vehicle interface", domain, self.profile)
            return
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
                self.misses = 0  # a pose: the run of timeouts is over
                if not self.armed:
                    self.armed = True
                    log.info("lockstep: armed (the localizer's pose at %d ns)", ns)

            n.create_subscription(PoseWithCovarianceStamped, self.lockstep["topic"], on_pose,
                                  reliable)
            log.info("lockstep: each frame waits up to %.2f s for a pose on %s",
                     self.lockstep["timeout_s"], self.lockstep["topic"])
        self._reset_lockstep()
        self.ground_truth = {}
        if ground_truth:
            self._make_ground_truth(ground_truth)
        log.info("ego ROS 2 side on domain %d", domain)

    def _make_ground_truth(self, ground_truth):
        """base_link's ground truth publishers (the ego config's `ground_truth`)."""
        n, reliable = self.node, self.reliable
        self.ground_truth = dict(ground_truth)
        if self.ground_truth:
            from geometry_msgs.msg import PoseWithCovarianceStamped
            from nav_msgs.msg import Odometry

            from .rig import POSE_ESTIMATE_TOPIC
            from .sensing import pose_covariance

            self.m.update(Odometry=Odometry, PoseWithCovarianceStamped=PoseWithCovarianceStamped)
            self.pub["gt"] = n.create_publisher(Odometry, self.ground_truth["topic"], reliable)
            self._gt_cov = pose_covariance(self.ground_truth["covariance"])
            if self.ground_truth["pose_estimate"]:
                self.pub["gt_pose"] = n.create_publisher(PoseWithCovarianceStamped,
                                                         POSE_ESTIMATE_TOPIC, reliable)
            log.info("ground truth on %s%s", self.ground_truth["topic"],
                     f", and as the pose estimate on {POSE_ESTIMATE_TOPIC}"
                     if self.ground_truth["pose_estimate"] else "")

    def _topic(self, name):
        """A profile's topic: a relative name under the interface's namespace (as the VESC
        driver's), then the interface's remap."""
        ns = self.interface["namespace"].strip("/")
        full = name if name.startswith("/") else "/" + "/".join(x for x in (ns, name) if x)
        return self.interface["remap"].get(full, full)

    def _make_interface(self, rig):
        """The `ackermann` or `vesc` profile's topics (aib.interfaces)."""
        from . import interfaces as itf

        n, reliable = self.node, self.reliable
        if self.profile == "ackermann":
            from ackermann_msgs.msg import AckermannDriveStamped
            from nav_msgs.msg import Odometry
            from tf2_msgs.msg import TFMessage
            from geometry_msgs.msg import TransformStamped

            self.m.update(Odometry=Odometry, TFMessage=TFMessage, TransformStamped=TransformStamped)
            self.input = itf.AckermannInput()
            wheelbase = self.interface.get("wheelbase") or (rig.vehicle.wheel_base if rig else None)
            self.odometer = itf.Odometer(wheelbase)
            self._radius = rig.vehicle.wheel_radius if rig else None

            def on_drive(msg):
                d = msg.drive
                self.input.on_drive(self.now, d.speed, d.acceleration, d.jerk, d.steering_angle,
                                    d.steering_angle_velocity)

            n.create_subscription(AckermannDriveStamped, self._topic("/drive"), on_drive, reliable)
            self.pub["odom"] = n.create_publisher(Odometry, self._topic("/odom"), 10)
            self.pub["tf"] = n.create_publisher(TFMessage, "/tf", 10)
        else:
            from std_msgs.msg import Float64
            try:
                from vesc_msgs.msg import VescStateStamped
            except ImportError as e:
                raise ImportError("the vesc vehicle interface needs vesc_msgs built for Isaac's "
                                  "Python: src/aib/scripts/build_msgs.sh vesc_msgs, with "
                                  "AUTOWARE_PREFIX the vesc_msgs install of the stack's workspace "
                                  f"(<ws>/install/vesc_msgs) ({e})") from e
            self.m.update(Float64=Float64, VescStateStamped=VescStateStamped)
            self.input = itf.VescInput(self.interface.get("limits"))
            for topic in itf.MOTOR_TOPICS:
                def on_motor(msg, topic=topic):
                    self.input.on_motor(self.now, topic, msg.data)

                n.create_subscription(Float64, self._topic(f"commands/motor/{topic}"), on_motor,
                                      10)
            echo = n.create_publisher(Float64, self._topic("sensors/servo_position_command"), 10)

            def on_servo(msg):
                echo.publish(Float64(data=self.input.on_servo(msg.data)))

            n.create_subscription(Float64, self._topic("commands/servo/position"), on_servo, 10)
            self.pub["core"] = n.create_publisher(VescStateStamped, self._topic("sensors/core"),
                                                  10)
            self.core_clock = itf.TelemetryClock(self.interface["core_rate_hz"])
        self.telemetry = None

    @classmethod
    def from_config(cls, cfg, rig=None):
        return cls(domain=cfg.domain, rig=rig, node_name=cfg.node_name,
                   control_topic=cfg.control_topic, lockstep=cfg.lockstep,
                   ground_truth=cfg.ground_truth, vehicle_interface=cfg.vehicle_interface)

    def sensor_publisher(self, s):
        """The publisher(s) for a SensorRig, made on first use: (main, info or None)."""
        kind = {"lidar": "PointCloud2", "imu": "Imu", "gnss": "NavSatFix", "camera": "Image"}[s.type]
        qos = self.sensor_qos if s.type in ("lidar", "camera") else self.reliable
        topic = s.topic or DEFAULT_TOPICS.get(s.type, "")
        if s.type == "lidar" and s.isaac.get("laser_scan"):
            # a planar lidar as its driver publishes it (aib.sensing.laser_scan_settings):
            # urg_node's QoS, reliable, depth 20, which a default subscriber matches
            from rclpy.qos import QoSProfile

            kind, topic = "LaserScan", s.isaac["laser_scan"].get("topic", "/scan")
            qos = QoSProfile(depth=20)
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
        if self.profile != "autoware":
            self.input.reset()
            if self.profile == "ackermann":
                self.odometer.reset()
            else:
                self.core_clock.reset()
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
                    log.warning("lockstep: no pose for %d frames in a row (the last scan at %d ns, "
                                "the last pose at %d ns); disarmed until the localizer publishes "
                                "again", self.misses, self.scan_ns, self.localized_ns or 0)
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

    def command(self, now):
        """What the vehicle takes at simulation time `now` (s): Autoware's `Command`, or the
        profile's `AckermannCommand` / `VescCommand` with its age."""
        return self.cmd if self.profile == "autoware" else self.input.command(now)

    @staticmethod
    def stamp(sim_time):
        """The ROS time of `sim_time` s: nanos(sim_time) + BASE_NS."""
        from builtin_interfaces.msg import Time

        from .sim_time import sim_ns

        ns = sim_ns(sim_time)
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    def note_telemetry(self, t, telemetry):
        """The component's telemetry after the physics step ending at `t` (s): the vesc
        profile's `sensors/core` samples it at its own rate (`core_rate_hz`)."""
        if self.profile == "vesc":
            self.core_clock.note(t, telemetry)

    def _publisher(self, kind, topic, qos):
        """One publisher per topic for the process (egos come and go; EgoRos stays)."""
        if topic not in self._sensor_pubs:
            self._sensor_pubs[topic] = self.node.create_publisher(kind, topic, qos)
        return self._sensor_pubs[topic]

    def imu_sender(self, rig):
        """How the rig's IMU `rig` publishes a reading ((wxyz, gyro rad/s, specific force m/s^2),
        stamp): on its own topic for Autoware; on `/sensors/imu/raw` (remappable) for the
        ackermann profile; as the VESC driver for the vesc profile, `sensors/imu`
        (VescImuStamped) and `sensors/imu/raw` (Imu), both in the VESC's units and with no
        frame_id (aib.interfaces.vesc_imu)."""
        m = self.m
        if self.profile == "vesc":
            from vesc_msgs.msg import VescImuStamped

            from .interfaces import vesc_imu

            vimu = self._publisher(VescImuStamped, self._topic("sensors/imu"), 10)
            raw = self._publisher(m["Imu"], self._topic("sensors/imu/raw"), 10)

            def send(reading, stamp):
                ypr, acc, gyro, q = vesc_imu(reading)
                v = VescImuStamped()
                v.header.stamp = stamp
                v.imu.ypr.x, v.imu.ypr.y, v.imu.ypr.z = ypr
                v.imu.linear_acceleration.x, v.imu.linear_acceleration.y, \
                    v.imu.linear_acceleration.z = acc
                v.imu.angular_velocity.x, v.imu.angular_velocity.y, v.imu.angular_velocity.z = gyro
                v.imu.orientation.w, v.imu.orientation.x, v.imu.orientation.y, \
                    v.imu.orientation.z = q
                vimu.publish(v)
                r = m["Imu"]()
                r.header.stamp = stamp
                r.linear_acceleration.x, r.linear_acceleration.y, r.linear_acceleration.z = acc
                r.angular_velocity.x, r.angular_velocity.y, r.angular_velocity.z = gyro
                r.orientation.w, r.orientation.x, r.orientation.y, r.orientation.z = q
                raw.publish(r)
            return send
        if self.profile == "ackermann":
            pub = self._publisher(m["Imu"], self._topic("/sensors/imu/raw"), self.reliable)
        else:
            pub, _ = self.sensor_publisher(rig)

        def send(reading, stamp):
            q, gyro, acc = reading
            msg = m["Imu"]()
            msg.header.stamp = stamp
            msg.header.frame_id = rig.frame
            msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = (
                float(c) for c in q)
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (
                float(c) for c in gyro)
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (
                float(c) for c in acc)
            pub.publish(msg)
        return send

    def publish_status(self, state, stamp, wheel_speeds=None, driven=None):
        """The vehicle interface's status for one frame: Autoware's reports, the `ackermann`
        profile's odometry (from the driven wheels' `wheel_speeds`, rad/s, and the tire angle),
        or the `vesc` profile's `sensors/core` (the component's last telemetry, `telemetry`)."""
        if self.profile == "ackermann":
            return self._publish_odom(state, stamp, wheel_speeds, driven)
        if self.profile == "vesc":
            t = stamp.sec + stamp.nanosec * 1e-9
            return self._publish_core(t)
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

    def _publish_odom(self, state, stamp, wheel_speeds, driven):
        """`/odom` and tf odom -> base_link, by vesc_to_odom's kinematics on the driven wheels'
        rolling speed and the tire angle (aib.interfaces.Odometer)."""
        import math

        from .interfaces import rolling_speed

        speed = (rolling_speed(wheel_speeds, self._radius, driven or (True,) * 4)
                 if wheel_speeds is not None and self._radius else float(state.velocity[0]))
        t = stamp.sec + stamp.nanosec * 1e-9
        x, y, yaw, v, w = self.odometer.update(t, speed, state.tire_angle)
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        frames = self.interface["frames"]
        o = self.m["Odometry"]()
        o.header.stamp = stamp
        o.header.frame_id = frames["odom"]
        o.child_frame_id = frames["base"]
        o.pose.pose.position.x, o.pose.pose.position.y = x, y
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = qz, qw
        o.pose.covariance[0] = o.pose.covariance[7] = 0.2  # as vesc_to_odom
        o.pose.covariance[35] = 0.4
        o.twist.twist.linear.x = v
        o.twist.twist.angular.z = w
        self.pub["odom"].publish(o)
        if self.interface["publish_tf"]:
            tf = self.m["TransformStamped"]()
            tf.header.stamp = stamp
            tf.header.frame_id = frames["odom"]
            tf.child_frame_id = frames["base"]
            tf.transform.translation.x, tf.transform.translation.y = x, y
            tf.transform.rotation = o.pose.pose.orientation
            self.pub["tf"].publish(self.m["TFMessage"](transforms=[tf]))

    def _publish_core(self, t):
        """`sensors/core` at its own rate (`core_rate_hz`, the driver's 50 Hz by default): one
        message per tick that fell in the frame ending at `t` (the ROS time, s), each from the
        component's telemetry at or before its tick, stamped at it (aib.interfaces)."""
        from .interfaces import vesc_state
        from .sim_time import BASE_NS

        base = BASE_NS * 1e-9
        for tick, telemetry in self.core_clock.due(t - base):
            msg = self.m["VescStateStamped"]()
            msg.header.stamp = self.stamp(tick)
            for k, v in vesc_state(telemetry).items():
                setattr(msg.state, k, v)
            self.pub["core"].publish(msg)

    def publish_ground_truth(self, gt, stamp):
        """base_link's ground truth (aib.sensing.ground_truth) for one frame, when on."""
        if not self.ground_truth:
            return
        (px, py, pz), (qw, qx, qy, qz), v, w = gt
        o = self.m["Odometry"]()
        o.header.stamp = stamp
        o.header.frame_id = "map"
        o.child_frame_id = "base_link"
        pose = o.pose.pose
        pose.position.x, pose.position.y, pose.position.z = px, py, pz
        pose.orientation.w, pose.orientation.x = qw, qx
        pose.orientation.y, pose.orientation.z = qy, qz
        o.pose.covariance = self._gt_cov
        tw = o.twist.twist
        tw.linear.x, tw.linear.y, tw.linear.z = v
        tw.angular.x, tw.angular.y, tw.angular.z = w
        self.pub["gt"].publish(o)
        if "gt_pose" in self.pub:
            p = self.m["PoseWithCovarianceStamped"]()
            p.header = o.header
            p.pose = o.pose
            self.pub["gt_pose"].publish(p)

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
