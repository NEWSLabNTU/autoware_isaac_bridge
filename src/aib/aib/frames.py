"""Frames and transforms for the ego's sensors. Pure Python: no Isaac.

Quaternions are (w, x, y, z); a transform is (translation, quaternion), parent to child.
"""
import math

EARTH_RADIUS = 6378137.0  # m, WGS84 equatorial


def rpy_quat(roll, pitch, yaw):
    """URDF fixed-joint rpy (R = Rz(yaw) Ry(pitch) Rx(roll)) as (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qrot(q, v):
    """Rotate v by unit quaternion q."""
    w, x, y, z = q
    p = qmul(qmul(q, (0.0, *v)), (w, -x, -y, -z))
    return p[1:]


def compose(a, b):
    """Transform a then b, each (translation, quaternion): parent->a, a->b gives parent->b."""
    (ta, qa), (tb, qb) = a, b
    r = qrot(qa, tb)
    return (tuple(x + y for x, y in zip(ta, r)), qmul(qa, qb))


def advance(pose, velocity, angular_velocity, dt):
    """`pose` (translation, quaternion) moved on for `dt` s at `velocity` (world frame, m/s)
    and `angular_velocity` (body frame, rad/s): where a body will be after a step."""
    (t, q) = pose
    wx, wy, wz = (c * dt for c in angular_velocity)
    a = math.sqrt(wx * wx + wy * wy + wz * wz)
    dq = (1.0, 0.0, 0.0, 0.0) if a < 1e-12 else \
        (math.cos(a / 2), *(c / a * math.sin(a / 2) for c in (wx, wy, wz)))
    return (tuple(x + v * dt for x, v in zip(t, velocity)), qmul(q, dq))


def mount_transform(m):
    """A calibration entry ({x, y, z, roll, pitch, yaw}) as (translation, quaternion)."""
    return ((float(m["x"]), float(m["y"]), float(m["z"])),
            rpy_quat(float(m["roll"]), float(m["pitch"]), float(m["yaw"])))


def kit_transform(sensors_calibration, sensor_kit_calibration, key):
    """base_link -> `key` (a sensor_kit_base_link child) from the two calibration files."""
    import yaml

    with open(sensors_calibration) as f:
        base = yaml.safe_load(f)["base_link"]["sensor_kit_base_link"]
    with open(sensor_kit_calibration) as f:
        kit = yaml.safe_load(f)["sensor_kit_base_link"][key]
    return compose(mount_transform(base), mount_transform(kit))


def lat_lon(x, y):
    """Map x/y (east, north, m) to WGS84 degrees about origin (0, 0): the map's projector is
    TransverseMercator at (0, 0), which within a town of the origin is this to millimetres."""
    lat = math.degrees(y / EARTH_RADIUS)
    lon = math.degrees(x / (EARTH_RADIUS * math.cos(math.radians(lat))))
    return lat, lon


# WGS84 and UTM, for the inverse Krueger series (millimetres within a zone)
_F = 1 / 298.257223563
_N = _F / (2 - _F)
_A = EARTH_RADIUS / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_BETA = (_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96, _N ** 2 / 48 + _N ** 3 / 15,
         17 * _N ** 3 / 480)
_DELTA = (2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3, 7 * _N ** 2 / 3 - 8 * _N ** 3 / 5,
          56 * _N ** 3 / 15)
UTM_K0 = 0.9996
MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"  # 8 deg each from 80 S (X: 72-84 N)


def utm_lat_lon(zone, north, easting, northing):
    """UTM (zone 1-60, hemisphere) to WGS84 degrees."""
    xi = (northing - (0.0 if north else 10_000_000.0)) / (UTM_K0 * _A)
    eta = (easting - 500_000.0) / (UTM_K0 * _A)
    xi_p, eta_p = xi, eta
    for j, b in enumerate(_BETA, 1):
        xi_p -= b * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
        eta_p -= b * math.cos(2 * j * xi) * math.sinh(2 * j * eta)
    chi = math.asin(math.sin(xi_p) / math.cosh(eta_p))
    lat = chi + sum(d * math.sin(2 * j * chi) for j, d in enumerate(_DELTA, 1))
    lon0 = math.radians(6 * zone - 183)
    lon = lon0 + math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return math.degrees(lat), math.degrees(lon)


def mgrs_origin(grid):
    """(zone, north, easting, northing) of an MGRS 100 km square's south-west corner, e.g.
    "51RUH": what Autoware's MGRS projector adds to a map's x/y."""
    zone, band, col, row = int(grid[:-3]), grid[-3].upper(), grid[-2].upper(), grid[-1].upper()
    cols = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")[(zone - 1) % 3]
    rows = "ABCDEFGHJKLMNPQRSTUV"
    easting = (cols.index(col) + 1) * 100_000.0
    base = ((rows.index(row) - (5 if zone % 2 == 0 else 0)) % 20) * 100_000.0
    north = band >= "N"
    lo = -80 + 8 * MGRS_BANDS.index(band)
    for k in range(10):  # the 2000 km cycle the band falls in
        northing = base + k * 2_000_000.0
        lat, _ = utm_lat_lon(zone, north, easting + 50_000.0, northing + 50_000.0)
        if lo - 1 <= lat < lo + (12 if band == "X" else 8) + 1:
            return zone, north, easting, northing
    raise ValueError(f"MGRS grid {grid}: no northing puts square {col}{row} in band {band}")


class Projector:
    """World x/y/z (the simulator's, m) to WGS84, as the map's projector reads it back: the
    world is the map less `map_origin`, and the map is projected by its
    `map_projector_info.yaml`. MGRS adds its 100 km square's corner and inverts UTM; without
    a projector file the map is TransverseMercator at (0, 0) (`lat_lon`, CARLA's towns)."""

    def __init__(self, map_origin=(0.0, 0.0, 0.0), mgrs=None):
        self.origin = tuple(float(v) for v in map_origin)
        self.mgrs = mgrs_origin(mgrs) if mgrs else None
        self.name = f"MGRS {mgrs}" if mgrs else "TransverseMercator at (0, 0)"

    @classmethod
    def from_map(cls, map_dir, map_origin=(0.0, 0.0, 0.0)):
        """The map dir's projector: MGRS, or none (TransverseMercator at (0, 0)). Another
        projector type is refused rather than published wrong."""
        import os

        import yaml

        path = os.path.join(map_dir, "map_projector_info.yaml")
        if not os.path.exists(path):
            return cls(map_origin)
        with open(path) as f:
            info = yaml.safe_load(f) or {}
        kind = info.get("projector_type", "")
        if kind == "MGRS":
            return cls(map_origin, info["mgrs_grid"])
        if kind in ("", "Local"):
            return cls(map_origin)
        raise ValueError(f"{path}: projector_type {kind} is not supported (MGRS or none)")

    def lat_lon_alt(self, x, y, z):
        mx, my, mz = x + self.origin[0], y + self.origin[1], z + self.origin[2]
        if self.mgrs is None:
            lat, lon = lat_lon(mx, my)
        else:
            zone, north, e0, n0 = self.mgrs
            lat, lon = utm_lat_lon(zone, north, e0 + mx, n0 + my)
        return lat, lon, mz
