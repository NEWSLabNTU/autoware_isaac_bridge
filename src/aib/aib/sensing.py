"""Sensor readings from the vehicle's state: pure maths (numpy for the lidar), no Isaac.

What the Isaac sensors give is raw (an RTX lidar's points, the body's pose); these turn it
into what Autoware's messages carry, so the conversions are testable without a simulator.
"""
import math

from .frames import qconj, qmul, qrot

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


def gnss_reading(state, t_base_gnss, projector=None, offset=(0.0, 0.0, 0.0)):
    """(lat, lon, altitude) of the antenna at `t_base_gnss` in base_link, moved by `offset`
    (east, north, up in the map, m: the receiver's error, GnssNoise.offset), by the map's
    projector (an aib.frames.Projector; none: TransverseMercator at (0, 0))."""
    p = tuple(a + b + e for a, b, e in zip(state.position, qrot(state.quat, t_base_gnss), offset))
    if projector is None:
        from .frames import Projector

        projector = Projector()
    return projector.lat_lon_alt(*p)


# The GNSS receiver's error (scensim issue 195 A9): an RTK fix's, 2 cm horizontally and twice
# that vertically, by default. Kept here rather than in the vehicle spec, so a receiver default
# does not move every spec_hash; a project overrides it in the ego config's `gnss`. acb has the
# same model and the same names (its gnss_noise_* parameters).
GNSS_NOISE_DEFAULTS = {"enabled": True, "horizontal_stddev_m": 0.02, "vertical_stddev_m": 0.04,
                       "bias_stddev_m": 0.0, "seed": None,
                       # the receiver's heading and attitude (scensim issue 195 L1b): a dual-antenna
                       # RTK receiver's, published as GnssInsOrientationStamped
                       "heading_stddev_deg": 0.2, "attitude_stddev_deg": 0.05}
# sensor_msgs/NavSatFix position_covariance_type
COVARIANCE_TYPE_UNKNOWN, COVARIANCE_TYPE_DIAGONAL_KNOWN = 0, 2


def gnss_noise_settings(over):
    """GNSS_NOISE_DEFAULTS with `over` on top, checked: ValueError names the bad key."""
    over = dict(over or {})
    unknown = set(over) - set(GNSS_NOISE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown key(s) {sorted(unknown)}; the keys are "
                         f"{', '.join(GNSS_NOISE_DEFAULTS)}")
    s = {**GNSS_NOISE_DEFAULTS, **over}
    if not isinstance(s["enabled"], bool):
        raise ValueError(f"enabled must be true or false, is {s['enabled']!r}")
    for k in ("horizontal_stddev_m", "vertical_stddev_m", "bias_stddev_m", "heading_stddev_deg",
              "attitude_stddev_deg"):
        if isinstance(s[k], bool) or not isinstance(s[k], (int, float)) or not s[k] >= 0.0:
            unit = "deg" if k.endswith("_deg") else "m"
            raise ValueError(f"{k} must be >= 0 ({unit}), is {s[k]!r}")
        s[k] = float(s[k])
    seed = s["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ValueError(f"seed must be an integer >= 0, or null for a fresh one, is {seed!r}")
    return s


class GnssNoise:
    """One receiver's error, east, north and up in the map (m): a bias drawn once from
    N(0, bias_stddev_m) on each axis, plus white noise per fix from N(0, horizontal_stddev_m)
    east and north and N(0, vertical_stddev_m) up. Seeded: the same seed gives the same fixes;
    no seed draws one, kept in `seed` so the run can be replayed. Disabled: no error at all,
    and no covariance claimed (today's truth).

    The orientation (L1b, `orientation_error`): roll and pitch from N(0, attitude_stddev_deg),
    yaw from N(0, heading_stddev_deg), per fix, in gnss_link's frame, from a second generator
    spawned from the same seed (numpy SeedSequence), so a seed's position errors are those it
    gave before L1b."""

    def __init__(self, settings):
        import secrets

        import numpy as np

        self.settings = settings
        self.enabled = settings["enabled"]
        self.seed = settings["seed"] if settings["seed"] is not None else secrets.randbits(63)
        self._rng = np.random.default_rng(self.seed)
        self._ins_rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(1)[0])
        att, head = (math.radians(settings["attitude_stddev_deg"]),
                     math.radians(settings["heading_stddev_deg"]))
        self._rot_sigma = np.array([att, att, head])
        h, v, b = (settings["horizontal_stddev_m"], settings["vertical_stddev_m"],
                   settings["bias_stddev_m"])
        self._sigma = np.array([h, h, v])
        self._b = b
        self.bias = (tuple(float(x) for x in self._rng.normal(0.0, b, 3)) if self.enabled
                     else (0.0, 0.0, 0.0))

    def offset(self):
        if not self.enabled:
            return (0.0, 0.0, 0.0)
        w = self._rng.normal(0.0, self._sigma)
        return tuple(float(a + e) for a, e in zip(self.bias, w))

    def orientation_error(self):
        """The next fix's orientation error: (roll, pitch, yaw) in rad, gnss_link's frame."""
        if not self.enabled:
            return (0.0, 0.0, 0.0)
        return tuple(float(e) for e in self._ins_rng.normal(0.0, self._rot_sigma))

    def rmse_rotation(self):
        """GnssInsOrientation's rmse_rotation_x, _y, _z (rad): the stddevs; disabled, zeros."""
        if not self.enabled:
            return (0.0, 0.0, 0.0)
        return tuple(float(s) for s in self._rot_sigma)

    def covariance(self):
        """(NavSatFix position_covariance, its type): the variance of white noise plus the bias's
        spread on the diagonal, known; disabled, zeros of unknown type."""
        if not self.enabled:
            return [0.0] * 9, COVARIANCE_TYPE_UNKNOWN
        var = [float(s) ** 2 + self._b ** 2 for s in self._sigma]
        return ([var[0], 0.0, 0.0, 0.0, var[1], 0.0, 0.0, 0.0, var[2]],
                COVARIANCE_TYPE_DIAGONAL_KNOWN)


def gnss_ins_orientation(q_base, q_base_gnss, error=(0.0, 0.0, 0.0)):
    """gnss_link's orientation in the map (wxyz): base_link's composed with the mount, turned by
    the receiver's `error` (roll, pitch, yaw in rad, gnss_link's frame; GnssNoise)."""
    q = qmul(q_base, q_base_gnss)
    if any(error):
        from .frames import rpy_quat

        q = qmul(q, rpy_quat(*error))
    return q


def ground_truth(state, map_origin=(0.0, 0.0, 0.0)):
    """base_link's ground truth for one frame: (position in the map, orientation wxyz, body
    velocity, body angular velocity). The map is the world plus `map_origin`, as the GNSS is
    projected (aib.frames.Projector); the velocities are base_link's, as the velocity report.

    The pose is the body's exact pose after the physics step. A rate taken by differencing
    successive poses is not the body's, though (scensim issue 104). PhysX solves a step's
    velocity first and then moves the body with it, the heading already turned, so the chord
    between two poses leads the headings read with them by half a step: r dt / 2 for a yaw
    rate r and the scene's physics step dt (8.3 ms at 60 Hz, 2.1 ms at 240 Hz). A sideslip or
    heading rate from differenced poses is off by that much. Take rates from the velocities
    here, which have no such shift."""
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
               time_stamp=None, ring=None):
    """Points (N x 3, sensor frame) to PointCloud2 bytes in `fmt`'s layout (POINT_FORMATS).
    Intensity in [0, 1] becomes 0-255. The channel is the elevation's place among `channels`
    evenly spaced rings, or with `rings` its rank among the elevations the scan has (a lidar
    whose lasers are not evenly spaced, as a VLP-32C's: each laser its own channel).
    `xyzircaedt` adds azimuth (atan2(y, x) in [0, 2 pi)), elevation and distance from the
    point, and each point's time_stamp (ns after the header's stamp, `point_times`), 0 when
    `time_stamp` is not given: a frame's points then share its stamp. `ring`, when given,
    is each point's channel as it stands (a profile's, aib.lidar_profile.rings)."""
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
    if ring is not None and len(ring) == n:
        cloud["channel"] = np.asarray(ring, dtype=np.uint16).reshape(-1)
    elif elevation is not None:
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


class RenderPose:
    """When the pose an RTX render takes was the sensor's (scensim issue 196). The host writes a
    sensor prim's pose before every substep, but the render on the frame's last substep takes the
    pose written before its first: the pose after the first substep, (n - 1) substeps before the
    frame's end. Measured with the faces of two parked cars: 25.6 to 26.1, 33.6 to 34.4 and 42.2 to
    42.7 ms late at 40, 60 and 120 Hz, against (n - 1) dt = 25, 33.3 and 41.7 ms.

    `substep(clock)` after each substep with the world clock at its end; `frame_done()` once the
    frame is stepped: how long before the frame's end the render's pose was, in seconds."""

    def __init__(self):
        self.first = None  # the clock at the end of this frame's first substep
        self.clock = 0.0

    def substep(self, clock):
        if self.first is None:
            self.first = clock
        self.clock = clock

    def frame_done(self):
        back = 0.0 if self.first is None else self.clock - self.first
        self.first = None
        return back


def earlier(stamp, seconds):
    """A ROS time stamp `seconds` before `stamp`, of its own type (builtin_interfaces Time)."""
    if not seconds:
        return stamp
    ns = stamp.sec * 1_000_000_000 + stamp.nanosec - round(seconds * 1e9)
    return type(stamp)(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


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
    """Fires at most once per frame, every `period` s of sim time on average (0: every frame).

    The next deadline is counted from the last deadline, not from the frame that fired, so a
    period that is not a whole number of frames keeps its rate: 30 Hz on 10 ms frames fires
    3-3-4 frames apart, not every 4th frame (25 Hz; issue 002). After a gap of more than a
    period (the first frame, a reset, a stall) it restarts from the frame, with no burst."""

    def __init__(self, period):
        self.period = float(period)
        self.next = 0.0

    def due(self, t):
        if self.period <= 0:
            return True
        if t + 1e-9 >= self.next:
            nxt = self.next + self.period
            self.next = nxt if t + 1e-9 < nxt else t + self.period
            return True
        return False

    def reset(self):
        self.next = 0.0


LASER_SCAN_KEYS = ("angle_min_deg", "angle_max_deg", "beams", "range_min", "range_max", "topic")


def laser_scan_settings(ls, profile):
    """A planar lidar's LaserScan layout (the rig's `isaac.laser_scan`; scensim roadmap 091
    W19) over its one-ring profile: `angle_min_deg` and `angle_max_deg` (the scan's first and
    last beam, + left of x), `beams` across them (inclusive, as urg_node counts a UST-10LX's
    1081 over 270 deg), `range_min` and `range_max` (m; default the profile's range). The
    profile fires `columns` over 360 deg; each beam must land on a column, so the step
    (max - min) / (beams - 1) must be 360 / columns. ValueError otherwise."""
    import math

    unknown = set(ls) - set(LASER_SCAN_KEYS)
    if unknown:
        raise ValueError(f"laser_scan: unknown key(s) {sorted(unknown)}")
    if not profile or int(profile.get("channels", 0)) != 1:
        raise ValueError("laser_scan needs a one-ring lidar profile (channels: 1)")
    lo, hi, n = float(ls["angle_min_deg"]), float(ls["angle_max_deg"]), int(ls["beams"])
    if not (hi > lo and n >= 2 and hi - lo <= 360.0):
        raise ValueError(f"laser_scan: {lo}..{hi} deg in {n} beams is not a scan")
    step = (hi - lo) / (n - 1)
    column = 360.0 / float(profile["columns"])
    if abs(step - column) > 1e-6 * column:
        raise ValueError(f"laser_scan: a beam every {step:g} deg, but the profile fires a column "
                         f"every {column:g} deg (columns {profile['columns']}): make them equal")
    return {"angle_min": math.radians(lo), "angle_max": math.radians(hi),
            "angle_increment": math.radians(step), "beams": n,
            "range_min": float(ls.get("range_min", 0.0)),
            "range_max": float(ls.get("range_max", profile["range"])),
            "scan_time": 1.0 / float(profile["rate_hz"]),
            "time_increment": 1.0 / (float(profile["rate_hz"]) * float(profile["columns"])),
            "topic": ls.get("topic", "/scan")}


def laser_scan_ranges(xyz, s):
    """A one-ring scan's points (N x 3, sensor frame) as LaserScan ranges for layout `s`
    (laser_scan_settings): each point to the beam nearest its azimuth, its planar distance;
    the nearest of a beam's points; beams outside the layout's angles dropped; no return, or
    one outside [range_min, range_max]: +inf (REP 117)."""
    import numpy as np

    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    out = np.full(s["beams"], np.inf)
    if not len(p):
        return out
    r = np.hypot(p[:, 0], p[:, 1])
    a = np.arctan2(p[:, 1], p[:, 0])
    i = np.rint((a - s["angle_min"]) / s["angle_increment"]).astype(np.int64)
    ok = (i >= 0) & (i < s["beams"]) & (r >= s["range_min"]) & (r <= s["range_max"])
    np.minimum.at(out, i[ok], r[ok])
    return out
