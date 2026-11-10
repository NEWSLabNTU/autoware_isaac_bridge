"""The ego's sensors on Isaac: RTX lidar, RGB camera, IMU and GNSS. Import after
`SimulationApp` (and after enabling `isaacsim.sensors.experimental.rtx` and
`isaacsim.sensors.rtx.nodes` for the lidar).

Each takes its SensorRig (frame, topic, base_link transform, Isaac options) and publishes
through the ego's EgoRos once per frame, after the frame's physics and render. The IMU and
GNSS are computed from the vehicle's state (`aib.sensing`); the lidar and camera render.
"""
import array
import collections
import logging

import os

from ..frames import advance, compose, qrot, rpy_quat
from ..sensing import (APERTURE_MM, POINT_FORMATS, USD_CAMERA_IN_LINK, Periodic, ScanGate, Ticks,
                       gnss_reading, imu_reading, motion_distort, pack_cloud, point_times,
                       sample_index)

log = logging.getLogger("aib")


def _mount(stage, path, transform):
    from pxr import Gf, UsdGeom

    (t, q) = transform
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*t))
    x.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*q))
    return x


def _matrix(q):
    """The rotation matrix of unit quaternion q (w, x, y, z): columns are the rotated axes."""
    import numpy as np

    return np.array([qrot(q, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]).T


# The lidars: beside the vehicles, not under them (`Lidar`)
SENSORS = "/World/aib_sensors"


def _pose_after(vehicle, state, dt):
    """Where `vehicle`'s base_link will be after a substep of `dt`: its body after the last
    substep moved on at its velocities, or its pose as set (a spawn, a teleport) before the
    first."""
    if vehicle.last is not None:
        pos, q, vel = vehicle.last
        return advance((pos, q), vel, state.angular_velocity, dt)
    t, q = vehicle.translate.Get(), vehicle.orient.Get()
    return (tuple(t), (q.GetReal(), *q.GetImaginary()))


class Lidar:
    """An RTX lidar. Its data is `latency` frames old when read (the annotator's: two renders,
    the rig's `isaac.latency_frames`), so a scan takes the stamp of the frame it was rendered in,
    not the one it is read in (swb docs/issues/015: 0.13 s of lag, 2.6 m at 20 m/s). It goes out
    one frame late with that stamp: sent in the same
    instant as `/clock` of its own frame, a scan races the EKF pose of that stamp to
    ndt_scan_matcher, whose pose buffer then cannot interpolate ("Couldn't interpolate
    pose" -> localization not OK -> MRM for a cycle, swb roadmap 038). On CARLA the scan
    arrives after the tick's /clock; on a vehicle, 50-100 ms after capture.

    The lidar is not under its vehicle. Isaac never gives back what an RTX lidar's render
    product cost (~105 MB): not destroyed, not paused, not pointed at a lidar at a new path
    (swb docs/issues/003); and pointing a kept product at a new lidar prim at the old path,
    which does give it back, once hung swb's world server's renderer. So a lidar prim and its
    render product are made once per vehicle path and sensor, under `SENSORS`, and kept for
    the next vehicle at that path (swb's world server spawns its egos at the same paths).
    Before each substep the host calls `before_substep`, which puts the lidar where its mount
    will be after that substep.

    As a driver publishes it (rig `isaac:` keys, each defaulting to what a real spinning lidar
    does when the point format carries per-point times, `xyzircaedt`):
    - `publish: scan` sends each revolution once, at the profile's scan rate; `frame` sends
      what the annotator holds every frame (a 10 Hz profile under 20 Hz frames: each scan
      twice). Default `scan`.
    - `point_times`: each point's firing time after the header's stamp (the annotator's
      timestamps), the header at the revolution's first point. Default on for `xyzircaedt`.
    - `motion_distortion`: each point as the sensor saw it from where it was at that time,
      not from the one pose the revolution was rendered from (Autoware's distortion corrector
      undoes it, as on the vehicle). Default `point_times`."""

    renders = True
    kept = {}  # its mount's prim path -> (lidar prim path, render product, annotator, ops)

    def __init__(self, rig, ros, stage, vehicle_path):
        import omni.replicator.core as rep
        from isaacsim.sensors.experimental.rtx import Lidar as RtxLidar

        self.rig, self.ros = rig, ros
        slug = rig.name.replace("/", "_")
        root = f"{SENSORS}/{vehicle_path.strip('/').replace('/', '_')}_{slug}"
        kept = Lidar.kept.get(root)
        # The annotator holds the frame the last vehicle's lidar saw: two renders of its own
        self.fresh = 2
        # The emitter in the message frame: an Isaac lidar emits along its prim's +x with +z up;
        # a sensor whose driver publishes in its own axes (the golf cart's Seyond Falcon: x up,
        # y right, z forward) names the emitter's rotation in that frame, `emitter_rpy`, and
        # its points are turned back into the frame before they go out.
        self.q_emit = rpy_quat(*(float(a) for a in rig.isaac.get("emitter_rpy", (0, 0, 0))))
        self.mount = compose(rig.transform, ((0.0, 0.0, 0.0), self.q_emit))
        if kept is None or not stage.GetPrimAtPath(kept[0]):
            ops = _mount(stage, root, self.mount).GetOrderedXformOps()
            usd = rig.isaac.get("usd")
            if usd:
                # An OmniLidar profile of our own (a lidar Isaac does not ship), by path
                lidar = RtxLidar.create(f"{root}/lidar", usd_path=os.path.expandvars(usd))
            else:
                lidar = RtxLidar.create(f"{root}/lidar", config=rig.isaac.get("config", "OS1"),
                                        variant=rig.isaac.get("variant", "OS1_REV7_128ch20hz1024res"))
            rp = rep.create.render_product(lidar.paths[0], resolution=(128, 128))
            annotator = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloud")
            annotator.attach([rp.path])
            kept = Lidar.kept[root] = (lidar.paths[0], rp, annotator, ops)
            self.fresh = 0
        path, self.render_product, self.annotator, (self.translate, self.orient) = kept
        self.channels = int(rig.extra.get("channels", 128))
        # `xyzirc` (default) or `xyzircaedt` (Nebula's, as a real Velodyne driver publishes)
        self.point_format = rig.isaac.get("point_format", "xyzirc")
        if self.point_format not in POINT_FORMATS:
            raise ValueError(f"lidar {rig.name}: point_format {self.point_format!r}, "
                             f"not one of {sorted(POINT_FORMATS)}")
        self.rings = bool(rig.isaac.get("rings_by_elevation", False))
        self.pub, _ = ros.sensor_publisher(rig)
        self.pending = None
        self.latency = int(rig.isaac.get("latency_frames", 2))
        self.stamps = collections.deque(maxlen=self.latency + 1)  # the last frames' stamps
        aedt = self.point_format == "xyzircaedt"
        self.per_scan = rig.isaac.get("publish", "scan") == "scan"
        self.point_times = bool(rig.isaac.get("point_times", aedt)) and aedt
        self.distort = bool(rig.isaac.get("motion_distortion", self.point_times)) and self.point_times
        self.gate = ScanGate()
        # The world's clock (the annotator's timestamps run on it) and the sensor's trajectory
        # on it, a substep apart, for the motion a revolution spans
        self.clock, self.frame_dt, self.frame_start = 0.0, 0.0, 0.0
        self.track = collections.deque(maxlen=256)
        log.info("lidar %s: %s at %s in base_link%s", rig.name, path, rig.translation,
                 " (kept from the last vehicle at this path)" if self.fresh else "")

    def before_substep(self, vehicle, state, dt):
        from pxr import Gf

        (x, y, z), q = compose(_pose_after(vehicle, state, dt), self.mount)
        self.translate.Set(Gf.Vec3d(x, y, z))
        self.orient.Set(Gf.Quatd(*q))
        self.clock += dt
        self.track.append((self.clock, (x, y, z), tuple(q)))

    def ready(self):
        if self.fresh > 0:
            self.fresh -= 1
            return False
        d = self.annotator.get_data()
        return isinstance(d, dict) and len(d.get("data", [])) > 0

    def publish(self, state, sim_time, stamp):
        if self.pending is not None:
            self.pub.publish(self.pending)
            self.ros.note_scan(self.pending.header.stamp)  # what the lockstep waits for
        self.stamps.append(stamp)
        self.frame_dt, self.frame_start = self.clock - self.frame_start, self.clock
        # the data read now was rendered `latency` frames ago: until then there is no stamp for it
        self.pending = (self._msg(self.stamps[0], sim_time - self.clock)
                        if len(self.stamps) > self.latency else None)

    def _msg(self, stamp, to_sim):
        """The scan the annotator holds, or None (nothing yet, or `publish: scan` and this
        revolution already sent). `to_sim`: the host's sim time less the world's clock."""
        import numpy as np

        d = self.annotator.get_data()
        if not isinstance(d, dict) or not len(d.get("data", [])):
            return None
        info = d.get("info", {})
        xyz = d["data"]
        n_pts = len(np.asarray(xyz).reshape(-1, 3))
        ts = np.asarray(info.get("timestamp", ())).reshape(-1)
        ts = ts if len(ts) == n_pts else None
        if self.per_scan and ts is not None and not self.gate.new(ts):
            return None
        offsets = None
        if self.point_times and ts is not None:
            t0, offsets = point_times(ts)
            if not getattr(self, "_told", False):
                self._told = True
                log.info("lidar %s: %d points a revolution over %.1f ms, first at world %.3f s "
                         "(clock %.3f s)", self.rig.name, n_pts, int(offsets.max()) * 1e-6,
                         t0 * 1e-9, self.clock)
            stamp = self.ros.stamp(t0 * 1e-9 + to_sim)
            if self.distort and len(self.track) > 1:
                # rendered from the pose `latency` frames back; fired over the revolution
                t_ref = self.clock - self.latency * self.frame_dt
                tr = list(self.track)
                xyz = motion_distort(xyz, ts.astype(np.float64) * 1e-9, t_ref,
                                     [a for a, _, _ in tr], [b for _, b, _ in tr],
                                     [c for _, _, c in tr])
        if self.q_emit != (1.0, 0.0, 0.0, 0.0):
            xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3) @ _matrix(self.q_emit).T
        n, data = pack_cloud(xyz, info.get("intensity"), info.get("elevation"), self.channels,
                             fmt=self.point_format, rings=self.rings, time_stamp=offsets)
        fields, step = POINT_FORMATS[self.point_format]
        m = self.ros.m
        PF = m["PointField"]
        msg = m["PointCloud2"]()
        msg.header.stamp = stamp
        msg.header.frame_id = self.rig.frame
        msg.height = 1
        msg.width = n
        types = {"f4": PF.FLOAT32, "u1": PF.UINT8, "u2": PF.UINT16, "u4": PF.UINT32}
        msg.fields = [PF(name=name, offset=off, datatype=types[t], count=1)
                      for name, off, t in fields]
        msg.is_bigendian = False
        msg.point_step = step
        msg.row_step = step * n
        msg.is_dense = True
        # array.array, not bytes: rclpy copies an array in C, bytes element by element (40 ms)
        msg.data = array.array("B", data)
        return msg

    def close(self):
        """Nothing to tear down: the lidar stays, live, for the next vehicle at this path."""


class Camera:
    """A pinhole RGB camera: Image (rgb8) and CameraInfo at the rig's rate. Not exercised
    by swb's runs yet (its traffic lights reach Autoware by V2X): off in the shipped rig.

    `auto_exposure: true` (the rig's, or an ego config's override) gives it RTX's auto-exposure,
    as a real camera has: without it a fixed exposure leaves the shade under trees nearly
    black next to a sunlit road (swb roadmap 059). On Isaac 6.1 that is the camera prim's
    OmniRtxCameraAutoExposureAPI_1, not the global /rtx/post/histogram setting."""

    renders = True

    def __init__(self, rig, ros, stage, vehicle_path):
        import math

        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        self.rig, self.ros = rig, ros
        slug = rig.name.replace("/", "_")
        # The mount is the camera link; the USD camera turned to look down the link's x.
        _mount(stage, f"{vehicle_path}/{slug}_mount", rig.transform)
        cam = UsdGeom.Camera.Define(stage, f"{vehicle_path}/{slug}_mount/camera")
        cam.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*USD_CAMERA_IN_LINK))
        self.width, self.height = int(rig.extra["width"]), int(rig.extra["height"])
        hfov = math.radians(float(rig.extra["horizontal_fov"]))
        cam.CreateHorizontalApertureAttr(APERTURE_MM)
        cam.CreateVerticalApertureAttr(APERTURE_MM * self.height / self.width)
        cam.CreateFocalLengthAttr(APERTURE_MM / (2 * math.tan(hfov / 2)))
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
        if rig.extra.get("auto_exposure"):
            from pxr import Sdf

            prim = cam.GetPrim()
            prim.AddAppliedSchema("OmniRtxCameraAutoExposureAPI_1")
            prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(True)
        self.render_product = rep.create.render_product(str(cam.GetPath()),
                                                        resolution=(self.width, self.height))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach([self.render_product.path])
        self.pub, self.info_pub = ros.sensor_publisher(rig)
        self.timer = Periodic(rig.period)
        self.k = list(rig.extra.get("k") or [])

    def ready(self):
        return True

    def publish(self, state, sim_time, stamp):
        if not self.timer.due(sim_time):
            return
        import numpy as np

        d = self.annotator.get_data()
        if d is None or not hasattr(d, "shape") or d.size == 0:
            return
        rgb = np.ascontiguousarray(np.asarray(d)[..., :3], dtype=np.uint8)
        m = self.ros.m
        img = m["Image"]()
        img.header.stamp = stamp
        img.header.frame_id = self.rig.frame
        img.height, img.width = rgb.shape[0], rgb.shape[1]
        img.encoding = "rgb8"
        img.step = 3 * img.width
        img.data = array.array("B", rgb.tobytes())
        self.pub.publish(img)
        info = m["CameraInfo"]()
        info.header = img.header
        info.height, info.width = img.height, img.width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = self.k
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [self.k[0], 0.0, self.k[2], 0.0, 0.0, self.k[4], self.k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info_pub.publish(info)

    def before_substep(self, vehicle, state, dt):
        pass  # under the vehicle: PhysX moves it

    def close(self):
        try:
            self.annotator.detach([self.render_product.path])
            self.render_product.destroy()
        except Exception:  # noqa: BLE001
            log.exception("camera %s teardown", self.rig.name)


class Imu:
    """From the vehicle's state: orientation, angular velocity, specific force."""

    renders = False

    def __init__(self, rig, ros, stage=None, vehicle_path=None):
        self.rig, self.ros = rig, ros
        self.pub, _ = ros.sensor_publisher(rig)
        self.timer = Periodic(rig.period)
        # Faster than the frames (a ZED's 100 Hz under 20 Hz frames): every tick of its own
        # clock, each from the last substep's state at or before it, as the hardware samples
        self.ticks = Ticks(rig.period) if rig.period > 0 else None
        self.clock = 0.0
        self.samples = collections.deque(maxlen=512)  # (world time, reading), a substep apart

    def ready(self):
        return True

    def publish(self, state, sim_time, stamp):
        if self.ticks is None:
            if self.timer.due(sim_time):
                self._send(imu_reading(state, self.rig.quat), stamp)
            return
        self.samples.append((self.clock, imu_reading(state, self.rig.quat)))
        to_sim = sim_time - self.clock
        times = [t for t, _ in self.samples]
        for tick in self.ticks.due(sim_time):
            reading = self.samples[sample_index(times, tick - to_sim)][1]
            self._send(reading, self.ros.stamp(tick))

    def _send(self, reading, stamp):
        q, gyro, acc = reading
        msg = self.ros.m["Imu"]()
        msg.header.stamp = stamp
        msg.header.frame_id = self.rig.frame
        msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = (float(c) for c in q)
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (float(c) for c in gyro)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (float(c) for c in acc)
        self.pub.publish(msg)

    def before_substep(self, vehicle, state, dt):
        # `state` is the body after the last substep, at the clock before this one
        if self.ticks is not None:
            self.samples.append((self.clock, imu_reading(state, self.rig.quat)))
        self.clock += dt

    def close(self):
        pass


class Gnss:
    """From the vehicle's pose: a GBAS fix at the antenna, by the map's projector (`projector`,
    set by Ego; none: TransverseMercator at (0, 0))."""

    renders = False

    def __init__(self, rig, ros, stage=None, vehicle_path=None):
        self.rig, self.ros = rig, ros
        self.projector = None
        self.pub, _ = ros.sensor_publisher(rig)
        self.timer = Periodic(rig.period or 0.1)

    def ready(self):
        return True

    def publish(self, state, sim_time, stamp):
        if not self.timer.due(sim_time):
            return
        lat, lon, alt = gnss_reading(state, self.rig.translation, self.projector)
        msg = self.ros.m["NavSatFix"]()
        msg.header.stamp = stamp
        msg.header.frame_id = self.rig.frame
        msg.status.status = 2  # GBAS_FIX, as acb
        msg.status.service = 1
        msg.latitude, msg.longitude, msg.altitude = lat, lon, alt
        msg.position_covariance_type = 0
        self.pub.publish(msg)

    def before_substep(self, vehicle, state, dt):
        pass  # read from the vehicle's state

    def close(self):
        pass


KINDS = {"lidar": Lidar, "camera": Camera, "imu": Imu, "gnss": Gnss}


def make(rig, ros, stage, vehicle_path):
    return KINDS[rig.type](rig, ros, stage, vehicle_path)
