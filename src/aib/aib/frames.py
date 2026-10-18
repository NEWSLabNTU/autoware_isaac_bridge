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
