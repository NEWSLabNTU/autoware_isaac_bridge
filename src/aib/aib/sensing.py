"""Sensor readings from the vehicle's state: pure maths (numpy for the lidar), no Isaac.

What the Isaac sensors give is raw (an RTX lidar's points, the body's pose); these turn it
into what Autoware's messages carry, so the conversions are testable without a simulator.
"""
from .frames import lat_lon, qconj, qmul, qrot

GRAVITY = 9.81
# PointCloud2 layout Autoware's preprocessor takes (acb's): x y z intensity return_type channel
CLOUD_FIELDS = (("x", 0, "f4"), ("y", 4, "f4"), ("z", 8, "f4"), ("intensity", 12, "u1"),
                ("return_type", 13, "u1"), ("channel", 14, "u2"))
POINT_STEP = 16
# A USD camera looks down its -Z with +Y up; a rig's camera link is x forward, z up: the
# camera prim's orientation in the link.
USD_CAMERA_IN_LINK = (0.5, 0.5, -0.5, -0.5)
# A USD camera's default horizontal aperture, mm (the focal length is set from the FOV).
APERTURE_MM = 20.955


def imu_reading(state, q_base_imu):
    """(orientation wxyz in the world, angular velocity, specific force), the last two in the
    IMU frame: the specific force is acceleration minus gravity, as an accelerometer reads."""
    g_body = qrot(qconj(state.quat), (0.0, 0.0, -GRAVITY))
    f = tuple(a - g for a, g in zip(state.acceleration, g_body))
    acc = qrot(qconj(q_base_imu), f)
    gyro = qrot(qconj(q_base_imu), state.angular_velocity)
    return qmul(state.quat, q_base_imu), gyro, acc


def gnss_reading(state, t_base_gnss, projector=None):
    """(lat, lon, altitude) of the antenna at `t_base_gnss` in base_link, by the map's
    projector (an aib.frames.Projector; none: TransverseMercator at (0, 0))."""
    p = tuple(a + b for a, b in zip(state.position, qrot(state.quat, t_base_gnss)))
    if projector is not None:
        return projector.lat_lon_alt(*p)
    lat, lon = lat_lon(p[0], p[1])
    return lat, lon, float(p[2])


def pack_cloud(xyz, intensity=None, elevation=None, channels=128):
    """Points (N x 3, sensor frame) to PointCloud2 bytes in CLOUD_FIELDS' layout. Intensity
    in [0, 1] becomes 0-255; the channel is the elevation's rank among `channels` rings."""
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = len(xyz)
    cloud = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "u1"),
                               ("return_type", "u1"), ("channel", "<u2")])
    cloud["x"], cloud["y"], cloud["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if intensity is not None:
        # Flat: the RTX lidar's annotator hands a 0-d array when it has nothing per point, which
        # len() refused, and the exception ended the runner (vehicle_check, roadmap 051).
        inten = np.asarray(intensity, dtype=np.float32).reshape(-1)
        if len(inten) == n:
            cloud["intensity"] = np.clip(inten * 255.0, 0, 255).astype(np.uint8)
    if elevation is not None:
        elev = np.asarray(elevation, dtype=np.float32).reshape(-1)
        if len(elev) == n and n:
            lo, hi = float(elev.min()), float(elev.max())
            if hi > lo:
                cloud["channel"] = np.rint((elev - lo) / (hi - lo) * (channels - 1)).astype(np.uint16)
    cloud["return_type"] = 1
    return n, cloud.tobytes()


class Periodic:
    """Fires at most every `period` s of sim time (0: every frame)."""

    def __init__(self, period):
        self.period = float(period)
        self.next = 0.0

    def due(self, t):
        if self.period <= 0:
            return True
        if t + 1e-9 >= self.next:
            self.next = t + self.period
            return True
        return False

    def reset(self):
        self.next = 0.0
