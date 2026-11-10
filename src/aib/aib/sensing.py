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
# Nebula's PointXYZIRCAEDT (autoware_point_types), what a real Velodyne driver publishes and
# what Autoware's distortion corrector and ring outlier filter read: the above, then azimuth,
# elevation and distance (rad, rad, m) and time_stamp (ns after the header's stamp).
CLOUD_FIELDS_AEDT = CLOUD_FIELDS + (("azimuth", 16, "f4"), ("elevation", 20, "f4"),
                                    ("distance", 24, "f4"), ("time_stamp", 28, "u4"))
POINT_FORMATS = {"xyzirc": (CLOUD_FIELDS, 16), "xyzircaedt": (CLOUD_FIELDS_AEDT, 32)}
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


def ground_truth(state, map_origin=(0.0, 0.0, 0.0)):
    """base_link's ground truth for one frame: (position in the map, orientation wxyz, body
    velocity, body angular velocity). The map is the world plus `map_origin`, as the GNSS is
    projected (aib.frames.Projector); the velocities are base_link's, as the velocity report."""
    p = tuple(float(a) + float(o) for a, o in zip(state.position, map_origin))
    return (p, tuple(float(c) for c in state.quat), tuple(float(c) for c in state.velocity),
            tuple(float(c) for c in state.angular_velocity))


def pose_covariance(variances):
    """A 6x6 row-major covariance, diagonal: (position variance, m^2; angle variance, rad^2)."""
    vp, va = (float(v) for v in variances)
    c = [0.0] * 36
    for i, v in enumerate((vp, vp, vp, va, va, va)):
        c[i * 7] = v
    return c


def pack_cloud(xyz, intensity=None, elevation=None, channels=128, fmt="xyzirc", rings=False,
               time_stamp=None):
    """Points (N x 3, sensor frame) to PointCloud2 bytes in `fmt`'s layout (POINT_FORMATS).
    Intensity in [0, 1] becomes 0-255. The channel is the elevation's place among `channels`
    evenly spaced rings, or with `rings` its rank among the elevations the scan has (a lidar
    whose lasers are not evenly spaced, as a VLP-32C's: each laser its own channel).
    `xyzircaedt` adds azimuth (atan2(y, x) in [0, 2 pi)), elevation and distance from the
    point, and each point's time_stamp (ns after the header's stamp, `point_times`), 0 when
    `time_stamp` is not given: a frame's points then share its stamp."""
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    n = len(xyz)
    fields, _ = POINT_FORMATS[fmt]
    cloud = np.zeros(n, dtype=[(name, "<" + t if t != "u1" else t) for name, _, t in fields])
    cloud["x"], cloud["y"], cloud["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if intensity is not None:
        # Flat: the RTX lidar's annotator hands a 0-d array when it has nothing per point, which
        # len() refused, and the exception ended the runner (vehicle_check, roadmap 051).
        inten = np.asarray(intensity, dtype=np.float32).reshape(-1)
        if len(inten) == n:
            cloud["intensity"] = np.clip(inten * 255.0, 0, 255).astype(np.uint8)
    if elevation is not None:
        elev = np.asarray(elevation, dtype=np.float32).reshape(-1)
        if len(elev) == n and n and rings:
            _, rank = np.unique(np.round(elev, 4), return_inverse=True)
            cloud["channel"] = rank.reshape(-1).astype(np.uint16)
        elif len(elev) == n and n:
            lo, hi = float(elev.min()), float(elev.max())
            if hi > lo:
                cloud["channel"] = np.rint((elev - lo) / (hi - lo) * (channels - 1)).astype(np.uint16)
    cloud["return_type"] = 1
    if fmt == "xyzircaedt" and n:
        x, y, z = (xyz[:, i].astype(np.float64) for i in range(3))
        horiz = np.hypot(x, y)
        cloud["azimuth"] = np.mod(np.arctan2(y, x), 2.0 * np.pi)
        cloud["elevation"] = np.arctan2(z, horiz)
        cloud["distance"] = np.hypot(horiz, z)
        if time_stamp is not None:
            ts = np.asarray(time_stamp).reshape(-1)
            if len(ts) == n:
                cloud["time_stamp"] = ts.astype(np.uint32)
    return n, cloud.tobytes()


class ScanGate:
    """Lets each lidar revolution through once. The RTX annotator hands the last finished
    revolution every frame, so a 10 Hz lidar under 20 Hz frames shows each one twice; a driver
    publishes it once. A revolution is known by its first point's time."""

    def __init__(self):
        self.last = None

    def new(self, timestamps):
        import numpy as np

        ts = np.asarray(timestamps).reshape(-1)
        if not len(ts):
            return False
        key = int(ts.min())
        if key == self.last:
            return False
        self.last = key
        return True


def point_times(timestamps):
    """The annotator's per-point times (ns of sim time) as a driver gives them: the scan's
    start, and each point's offset after it (uint32 ns, Nebula's time_stamp)."""
    import numpy as np

    ts = np.asarray(timestamps, dtype=np.uint64).reshape(-1)
    t0 = int(ts.min()) if len(ts) else 0
    return t0, (ts - np.uint64(t0)).astype(np.uint32)


def _qrot_many(q, v):
    """Rotate vectors v (N x 3) by unit quaternions q (N x 4, w x y z)."""
    import numpy as np

    w, u = q[:, :1], q[:, 1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _pose_at(t, hist_t, hist_p, hist_q):
    """Poses at times t from a sampled trajectory: positions linearly, orientations by
    normalised linear interpolation (the samples are a substep apart)."""
    import numpy as np

    hist_t = np.asarray(hist_t, dtype=np.float64)
    hist_p = np.asarray(hist_p, dtype=np.float64)
    hist_q = np.asarray(hist_q, dtype=np.float64).copy()
    for i in range(1, len(hist_q)):  # one hemisphere, so neighbours interpolate the short way
        if np.dot(hist_q[i], hist_q[i - 1]) < 0:
            hist_q[i] = -hist_q[i]
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    p = np.stack([np.interp(t, hist_t, hist_p[:, k]) for k in range(3)], axis=1)
    q = np.stack([np.interp(t, hist_t, hist_q[:, k]) for k in range(4)], axis=1)
    return p, q / np.linalg.norm(q, axis=1, keepdims=True)


def motion_distort(xyz, t_points, t_ref, hist_t, hist_p, hist_q):
    """What a spinning lidar measures while it moves. The RTX lidar renders a revolution from
    one pose, the sensor's at `t_ref`; a real one fires each point from wherever it is at that
    point's time. Each point goes to the world through the pose at `t_ref` and back through the
    pose at its own time (`t_points`, s), the sensor's trajectory sampled at `hist_t` (s),
    `hist_p` (N x 3) and `hist_q` (N x 4, w x y z). Autoware's distortion corrector undoes it."""
    import numpy as np

    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if not len(xyz):
        return xyz
    p_ref, q_ref = _pose_at([t_ref], hist_t, hist_p, hist_q)
    world = _qrot_many(np.repeat(q_ref, len(xyz), axis=0), xyz) + p_ref
    p_i, q_i = _pose_at(t_points, hist_t, hist_p, hist_q)
    q_inv = q_i * np.array([1.0, -1.0, -1.0, -1.0])
    return _qrot_many(q_inv, world - p_i)


class Ticks:
    """A sensor clock faster or slower than the frames: each call gives the ticks of `period`
    that fell in the frame just ended, the first at the first call's time."""

    def __init__(self, period):
        self.period = float(period)
        self.next = None

    def due(self, now):
        if self.next is None:
            self.next = now + self.period
            return [now]
        out = []
        while self.next <= now + 1e-9:
            out.append(self.next)
            self.next += self.period
        return out


def sample_index(times, t):
    """The index of the last sample at or before t (the first, when t precedes them all)."""
    import bisect

    return max(0, bisect.bisect_right(list(times), t + 1e-12) - 1)


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
