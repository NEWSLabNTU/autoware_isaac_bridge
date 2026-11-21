"""A lidar's OmniLidar profile from the vehicle spec (scensim roadmap 084b).

CARLA's lidar (ray_cast and ray_cast_semantic) fires `channels` rings spaced evenly from
`upper_fov` down to `lower_fov`, `points_per_second / (channels x rotation_frequency)` columns a
turn, and keeps every hit within `range`: no noise, no drop-off. Isaac's stock profiles are real
sensors' (the Ouster OS1's rings span +-21 deg, its weak returns drop beyond ~55 m), so the same
spec lidar gave ~20 % fewer points on Isaac than on CARLA (084: 88.7k against 112k a scan).

`profile(attrs, name)` takes the CARLA attributes the spec generates (sim_vehicle_spec.gen
.carla_attributes) and returns the rig's profile; `attributes(profile)` the OmniLidar prim's
`omni:sensor:Core:*` values that make Isaac fire the same pattern. Pure functions; the Isaac
side is aib.isaac.sensors.Lidar, which sets them on the prim before its render product.

Two ring orders: the profile's (`elevations`, OmniLidar's `channelId`) counts from the highest
ring, as CARLA fires them; the published PointCloud2's `channel` (`rings`) counts from the
lowest, 0 the lowest ring, as aib numbers every lidar's.
"""

import zlib

import numpy as np

CORE = "omni:sensor:Core:"
STATE = CORE + "emitterState:s001:"


def profile(attrs, name):
    """The rig's profile from a lidar's CARLA attributes (strings, as the spec generates them).
    None when they do not name a whole pattern (no points_per_second)."""
    if "points_per_second" not in attrs:
        return None
    channels = int(attrs["channels"])
    rate = float(attrs["rotation_frequency"])
    pps = float(attrs["points_per_second"])
    columns = pps / (channels * rate)
    if columns != int(columns):
        raise ValueError(f"lidar profile {name}: points_per_second {pps:g} is not a whole number "
                         f"of columns of {channels} rings at {rate:g} Hz")
    return {"name": name, "channels": channels, "upper_fov": float(attrs["upper_fov"]),
            "lower_fov": float(attrs["lower_fov"]), "columns": int(columns), "rate_hz": rate,
            "range": float(attrs["range"])}


def planar(sensor, name):
    """A planar scanner's (the spec's `message: laser_scan`; scensim 091 W19) profile and
    LaserScan layout: one ring at its elevation, a full turn of columns at the scan's step
    (sim_vehicle_spec's planar_columns), and the scan's window centred on the sensor's x, as
    `isaac.profile` and `isaac.laser_scan` (aib.sensing.laser_scan_settings)."""
    hfov, beams = float(sensor["horizontal_fov"]), int(sensor["beams"])
    columns = (beams - 1) * 360.0 / hfov
    if abs(columns - round(columns)) > 1e-6:
        raise ValueError(f"lidar {name}: {beams} beams over {hfov:g} deg do not divide a turn "
                         "into whole columns")
    elev = float(sensor["vertical_fov"][0])
    prof = {"name": name, "channels": 1, "upper_fov": elev, "lower_fov": elev,
            "columns": int(round(columns)), "rate_hz": float(sensor["rate_hz"]),
            "range": float(sensor["range"])}
    scan = {"angle_min_deg": -hfov / 2, "angle_max_deg": hfov / 2, "beams": beams,
            "range_min": float(sensor.get("min_range", 0.0)), "topic": sensor["topic"]}
    return prof, scan


TICK_RATE = "omni:sensor:tickRate"


def tick_rate(p, authored):
    """The OmniLidar's `omni:sensor:tickRate` for profile `p`, over the stock variant's
    `authored` one, or None when it is already that (scensim issue 099): the profile's scan
    rate. The variant's (20 Hz) was left in place while its scan rate was set, a mismatch
    Isaac's own lidar wrapper warns about. A profile turning faster than it ticked (the
    RoboRacer's 40 Hz on 20) handed tails of revolutions under 40 Hz frames: /scan at 33 Hz,
    a third of its beams missing."""
    rate = float(p["rate_hz"])
    if authored is not None and float(authored) == rate:
        return None
    return rate


def elevations(p):
    """Each ring's elevation in degrees, ring 0 the highest, as CARLA fires them."""
    return np.linspace(p["upper_fov"], p["lower_fov"], p["channels"]).tolist()


def attributes(p):
    """{OmniLidar attribute name: value} for profile `p`: CARLA's pattern on Isaac's rotary
    lidar. One emitter state, every ring at azimuth offset 0, fired across one column's period;
    one return, no angular or range noise, every hit within range kept (minReflectance 0)."""
    n = p["channels"]
    firing = int(round(p["columns"] * p["rate_hz"]))           # columns a second
    period_ns = 1e9 / firing
    zeros = [0.0] * n
    return {
        CORE + "numberOfChannels": n,
        CORE + "numberOfEmitters": n,
        CORE + "scanRateBaseHz": int(round(p["rate_hz"])),
        CORE + "reportRateBaseHz": firing,
        CORE + "patternFiringRateHz": firing,
        CORE + "farRangeM": p["range"],
        CORE + "minReflectance": 0.0,
        CORE + "minReflectionRangeM": p["range"],
        CORE + "maxReturns": 1,
        CORE + "azimuthErrorStd": 0.0,
        CORE + "elevationErrorStd": 0.0,
        CORE + "rangeAccuracyM": 0.0,
        STATE + "elevationDeg": elevations(p),
        STATE + "azimuthDeg": zeros,
        STATE + "channelId": list(range(1, n + 1)),
        # spread over one column, as a real sensor fires its rings: up to one column (0.35 deg
        # at 1024) of azimuth between the first ring and the last, where CARLA fires a column
        # at one instant
        STATE + "fireTimeNs": [int(i * period_ns / n) for i in range(n)],
        STATE + "distanceCorrectionM": zeros,
        STATE + "focalDistM": zeros,
        STATE + "focalSlope": zeros,
        STATE + "horOffsetM": zeros,
        STATE + "vertOffsetM": zeros,
        STATE + "reportRateDiv": zeros,
        "omni:sensor:modelName": p["name"],
    }


def rings(elevation_deg, p):
    """Each point's ring from its elevation (degrees), counted from the lowest (0) as aib
    numbers them; independent of which rings a scan happens to have returns on."""
    if p["channels"] == 1:  # a planar lidar: one ring
        return np.zeros(len(np.asarray(elevation_deg).reshape(-1)), np.uint16)
    step = (p["upper_fov"] - p["lower_fov"]) / (p["channels"] - 1)
    r = np.rint((np.asarray(elevation_deg, dtype=np.float64) - p["lower_fov"]) / step)
    return np.clip(r, 0, p["channels"] - 1).astype(np.uint16)


def check(elevation_deg, p, tol=None):
    """None when a scan's elevations (degrees) are the profile's rings, else what is wrong.
    Two tests, each enough alone to catch a stock profile left in place:
    - on the rings: 99 % of points within `tol` of a ring (default a quarter of the spacing;
      a still scan measured 1e-5 deg, 084b), and none beyond the field of view;
    - every ring: 90 % of the rings at or below -1 deg have a point. Those meet flat ground
      within ~115 m of a sensor 2 m up; the rings above may see only sky."""
    e = np.asarray(elevation_deg, dtype=np.float64).reshape(-1)
    if len(e) == 0:
        return "no points"
    if p["channels"] == 1:
        # a planar lidar (091 W19): every point on its one ring, within a hundredth of a degree
        tol = 0.01 if tol is None else tol
        bad = np.abs(e - p["upper_fov"]) > tol
        if bad.mean() > 0.01:
            return (f"{100 * bad.mean():.1f} % of points off the plane at {p['upper_fov']:g} deg "
                    f"by more than {tol:g} deg (elevations {e.min():.3f}..{e.max():.3f} deg)")
        return None
    step = (p["upper_fov"] - p["lower_fov"]) / (p["channels"] - 1)
    tol = step / 4 if tol is None else tol
    idx = (e - p["lower_fov"]) / step
    off = np.abs(idx - np.rint(idx)) * step
    bad = (off > tol) | (e < p["lower_fov"] - tol) | (e > p["upper_fov"] + tol)
    if bad.mean() > 0.01:
        return (f"{100 * bad.mean():.1f} % of points off the profile's rings by more than "
                f"{tol:.3f} deg (elevations {e.min():.2f}..{e.max():.2f} deg; rings every "
                f"{step:.3f} deg from {p['lower_fov']:g} to {p['upper_fov']:g})")
    rings = np.array(elevations(p))
    low = np.flatnonzero(rings <= -1.0)
    hit = np.zeros(p["channels"], bool)
    hit[np.clip(np.rint(idx[~bad]).astype(int), 0, p["channels"] - 1)] = True
    # idx counts from the lowest ring; elevations() from the highest
    low_hit = hit[p["channels"] - 1 - low]
    if len(low) and low_hit.mean() < 0.9:
        return (f"only {low_hit.sum()} of the {len(low)} rings at or below -1 deg have a point "
                f"(the lowest point at {e.min():.2f} deg)")
    return None


class PhaseJitter:
    """Column-phase drift (084e), opt-in: `phase_jitter: {seed: N}` in the rig's profile.

    A real spinning lidar is not phase-locked to the frame, so its columns fall at a new azimuth
    every turn. Isaac's rotary lidar starts every scan at the same one (and rebuilds itself when
    startAzimuthOffsetDeg changes), so a thin pole is hit in every scan or in none. The sensor
    turns the lidar prim about its own z by `delta` (degrees, drawn within one column, seeded)
    for a frame and turns that frame's points back with `unturn`: the columns land elsewhere on
    the world, the cloud's frame stays. None from `of` when the profile does not ask for it.

    One turn a revolution: the sensor refuses it unless a revolution is one frame, since the
    points of a revolution rendered over several frames would carry several turns and come back
    with one. The stream is seeded by the seed and the lidar's name (crc32, stable across
    runs), so two lidars of one rig do not drift in lockstep."""

    def __init__(self, seed, columns, lidar=""):
        self.seed = int(seed)
        self.rng = np.random.default_rng([self.seed, zlib.crc32(lidar.encode())])
        self.column = 360.0 / float(columns)

    @classmethod
    def of(cls, p, lidar=""):
        cfg = (p or {}).get("phase_jitter", "absent")
        if cfg == "absent":
            return None
        if "columns" not in p:
            raise ValueError(f"lidar profile {p.get('name', lidar)}: phase_jitter needs "
                             "`columns` (one turn is drawn within one column)")
        return cls((cfg or {}).get("seed", 0), p["columns"], lidar)

    def next(self):
        """The next frame's turn, degrees in [0, one column)."""
        return float(self.rng.uniform(0.0, self.column))


def unturn(xyz, delta):
    """Points (N x 3, the lidar's frame) seen by a lidar turned `delta` degrees about its z, in
    the unturned lidar's frame."""
    c, s = np.cos(np.radians(delta)), np.sin(np.radians(delta))
    return np.asarray(xyz, dtype=np.float64).reshape(-1, 3) @ np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).T
