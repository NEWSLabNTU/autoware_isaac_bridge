"""The ego's sensors on Isaac: RTX lidar, RGB camera, IMU and GNSS. Import after
`SimulationApp` (and after enabling `isaacsim.sensors.experimental.rtx` and
`isaacsim.sensors.rtx.nodes` for the lidar).

Each takes its SensorRig (frame, topic, base_link transform, Isaac options) and publishes
through the ego's EgoRos once per frame, after the frame's physics and render. The IMU and
GNSS are computed from the vehicle's state (`aib.sensing`); the lidar and camera render.
"""
import array
import logging

from ..sensing import (APERTURE_MM, USD_CAMERA_IN_LINK, Periodic, gnss_reading, imu_reading,
                       pack_cloud)

log = logging.getLogger("aib")


def _mount(stage, path, transform):
    from pxr import Gf, UsdGeom

    (t, q) = transform
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*t))
    x.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*q))
    return x


class Lidar:
    """An RTX lidar. Its scan goes out one frame late with its own stamp: sent in the same
    instant as `/clock` of its own frame, a scan races the EKF pose of that stamp to
    ndt_scan_matcher, whose pose buffer then cannot interpolate ("Couldn't interpolate
    pose" -> localization not OK -> MRM for a cycle, csb roadmap 038). On CARLA the scan
    arrives after the tick's /clock; on a vehicle, 50-100 ms after capture."""

    renders = True

    def __init__(self, rig, ros, stage, vehicle_path):
        import omni.replicator.core as rep
        from isaacsim.sensors.experimental.rtx import Lidar as RtxLidar

        self.rig, self.ros = rig, ros
        slug = rig.name.replace("/", "_")
        _mount(stage, f"{vehicle_path}/{slug}_mount", rig.transform)
        lidar = RtxLidar.create(f"{vehicle_path}/{slug}_mount/lidar",
                                config=rig.isaac.get("config", "OS1"),
                                variant=rig.isaac.get("variant", "OS1_REV7_128ch20hz1024res"))
        self.render_product = rep.create.render_product(lidar.paths[0], resolution=(128, 128))
        self.annotator = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloud")
        self.annotator.attach([self.render_product.path])
        self.channels = int(rig.extra.get("channels", 128))
        self.pub, _ = ros.sensor_publisher(rig)
        self.pending = None
        log.info("lidar %s: %s at %s in base_link", rig.name, lidar.paths[0], rig.translation)

    def ready(self):
        d = self.annotator.get_data()
        return isinstance(d, dict) and len(d.get("data", [])) > 0

    def publish(self, state, sim_time, stamp):
        if self.pending is not None:
            self.pub.publish(self.pending)
        self.pending = self._msg(stamp)

    def _msg(self, stamp):
        d = self.annotator.get_data()
        if not isinstance(d, dict) or not len(d.get("data", [])):
            return None
        info = d.get("info", {})
        n, data = pack_cloud(d["data"], info.get("intensity"), info.get("elevation"),
                             self.channels)
        m = self.ros.m
        PF = m["PointField"]
        msg = m["PointCloud2"]()
        msg.header.stamp = stamp
        msg.header.frame_id = self.rig.frame
        msg.height = 1
        msg.width = n
        msg.fields = [PF(name="x", offset=0, datatype=PF.FLOAT32, count=1),
                      PF(name="y", offset=4, datatype=PF.FLOAT32, count=1),
                      PF(name="z", offset=8, datatype=PF.FLOAT32, count=1),
                      PF(name="intensity", offset=12, datatype=PF.UINT8, count=1),
                      PF(name="return_type", offset=13, datatype=PF.UINT8, count=1),
                      PF(name="channel", offset=14, datatype=PF.UINT16, count=1)]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n
        msg.is_dense = True
        # array.array, not bytes: rclpy copies an array in C, bytes element by element (40 ms)
        msg.data = array.array("B", data)
        return msg

    def close(self):
        try:
            self.annotator.detach([self.render_product.path])
            self.render_product.destroy()
        except Exception:  # noqa: BLE001 - the stage is going anyway
            log.exception("lidar %s teardown", self.rig.name)


class Camera:
    """A pinhole RGB camera: Image (rgb8) and CameraInfo at the rig's rate. Not exercised
    by csb's runs yet (its traffic lights reach Autoware by V2X): off in the shipped rig."""

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

    def ready(self):
        return True

    def publish(self, state, sim_time, stamp):
        if not self.timer.due(sim_time):
            return
        q, gyro, acc = imu_reading(state, self.rig.quat)
        msg = self.ros.m["Imu"]()
        msg.header.stamp = stamp
        msg.header.frame_id = self.rig.frame
        msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = (float(c) for c in q)
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (float(c) for c in gyro)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (float(c) for c in acc)
        self.pub.publish(msg)

    def close(self):
        pass


class Gnss:
    """From the vehicle's pose: a GBAS fix at the antenna, about the map origin."""

    renders = False

    def __init__(self, rig, ros, stage=None, vehicle_path=None):
        self.rig, self.ros = rig, ros
        self.pub, _ = ros.sensor_publisher(rig)
        self.timer = Periodic(rig.period or 0.1)

    def ready(self):
        return True

    def publish(self, state, sim_time, stamp):
        if not self.timer.due(sim_time):
            return
        lat, lon, alt = gnss_reading(state, self.rig.translation)
        msg = self.ros.m["NavSatFix"]()
        msg.header.stamp = stamp
        msg.header.frame_id = self.rig.frame
        msg.status.status = 2  # GBAS_FIX, as acb
        msg.status.service = 1
        msg.latitude, msg.longitude, msg.altitude = lat, lon, alt
        msg.position_covariance_type = 0
        self.pub.publish(msg)

    def close(self):
        pass


KINDS = {"lidar": Lidar, "camera": Camera, "imu": Imu, "gnss": Gnss}


def make(rig, ros, stage, vehicle_path):
    return KINDS[rig.type](rig, ros, stage, vehicle_path)
