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
"""

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
    step = (p["upper_fov"] - p["lower_fov"]) / (p["channels"] - 1)
    r = np.rint((np.asarray(elevation_deg, dtype=np.float64) - p["lower_fov"]) / step)
    return np.clip(r, 0, p["channels"] - 1).astype(np.uint16)


def check(elevation_deg, p, tol=0.2):
    """None when a scan's elevations (degrees) are the profile's rings, else what is wrong:
    every point within `tol` of a ring, and the lowest ring hit (the ground is always there)."""
    e = np.asarray(elevation_deg, dtype=np.float64).reshape(-1)
    if len(e) == 0:
        return "no points"
    step = (p["upper_fov"] - p["lower_fov"]) / (p["channels"] - 1)
    off = np.abs((e - p["lower_fov"]) / step - np.rint((e - p["lower_fov"]) / step)) * step
    bad = (off > tol) | (e < p["lower_fov"] - tol) | (e > p["upper_fov"] + tol)
    if bad.mean() > 0.01:
        return (f"{100 * bad.mean():.1f} % of points off the profile's rings "
                f"(elevations {e.min():.2f}..{e.max():.2f} deg, rings every {step:.3f} deg "
                f"from {p['lower_fov']:g} to {p['upper_fov']:g})")
    if e.min() > p["lower_fov"] + tol:
        return f"lowest point at {e.min():.2f} deg, the profile's lowest ring {p['lower_fov']:g}"
    return None
