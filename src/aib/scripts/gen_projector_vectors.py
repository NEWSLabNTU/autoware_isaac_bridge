#!/usr/bin/env python3
"""The GNSS projector test vectors (scensim issue 195, the GNSS offset item: the fix not inverted through
the map's projector, acb and aib). Shared by aib and acb, so both bridges are checked alike.

Each case is a map projector as Autoware 1.5.0 reads it (autoware_map_projection_loader: TransverseMercator
takes scale_factor from map_projector_info.yaml, default 0.9996; MGRS uses UTM's 0.9996) and map x/y points;
the expected latitude/longitude come from PROJ (pyproj), independent of aib's own series:
- TransverseMercator about (lat0, lon0) with scale k: +proj=tmerc +lat_0 +lon_0 +k +ellps=WGS84, so x/y
  are from the origin's projection, as lanelet2's TransverseMercatorProjector gives them;
- MGRS: UTM in the grid's zone, x/y added to the 100 km square's south-west corner (Autoware's
  MGRSProjector), the corner derived here from the lettering and checked to fall in the grid's band.
Each case's projector is the dict its map_projector_info.yaml holds, scale_factor absent where the file omits
it, so a reader applies the default itself: the CARLA towns as carla-town-autoware-maps df499322 (scensim's
pin) declares them, scale_factor 1.0, plus Town01's points without the key (the default path); the course
fixture as scensim_isaac_world.course writes it (no scale_factor).
Points: each CARLA town's lanelet2 extent (from its local_x/local_y tags) with 20 m beyond every corner,
and its centre and origin; a course fixture's span; an MGRS square's corners and centre.
Usage: gen_projector_vectors.py <carla maps dir (Town01, ...)> <out.json>   (needs pyproj)"""
import json
import re
import sys

import pyproj
import yaml

maps, out = sys.argv[1], sys.argv[2]
cases = []


def tm_case(name, info, pts):
    o = info["map_origin"]
    lat0, lon0 = o["latitude"], o["longitude"]
    k = info.get("scale_factor", 0.9996)
    t = pyproj.Transformer.from_crs(
        f"+proj=tmerc +lat_0={lat0} +lon_0={lon0} +k={k} +x_0=0 +y_0=0 +ellps=WGS84 +units=m",
        "EPSG:4326", always_xy=True)
    vec = []
    for x, y in pts:
        lon, lat = t.transform(x, y)
        vec.append({"x": x, "y": y, "lat": lat, "lon": lon})
    cases.append({"name": name, "projector": info, "points": vec})


for town in ("Town01", "Town02", "Town03", "Town05", "Town10"):
    s = open(f"{maps}/{town}/lanelet2_map.osm").read()
    xs = [float(v) for v in re.findall(r'k="local_x" v="([-\d.e]+)"', s)]
    ys = [float(v) for v in re.findall(r'k="local_y" v="([-\d.e]+)"', s)]
    x0, x1, y0, y1 = min(xs) - 20, max(xs) + 20, min(ys) - 20, max(ys) + 20
    pts = [(x0, y0), (x0, y1), (x1, y0), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2), (0.0, 0.0)]
    info = yaml.safe_load(open(f"{maps}/{town}/map_projector_info.yaml"))
    assert info["projector_type"] == "TransverseMercator" and info["scale_factor"] == 1.0, town
    pts = [(round(x, 3), round(y, 3)) for x, y in pts]
    tm_case(f"carla {town} (TransverseMercator at 0, 0, scale 1.0 as declared)", info, pts)
    if town == "Town01":
        town01 = (dict((k, v) for k, v in info.items() if k != "scale_factor"), pts)
tm_case("carla Town01's points without scale_factor (Autoware's default 0.9996)", *town01)
# a course fixture (scensim_isaac_world.course: TransverseMercator about 36, 140)
tm_case("course fixture (TransverseMercator at 36, 140, no scale_factor: 0.9996)",
        {"projector_type": "TransverseMercator", "vertical_datum": "WGS84",
         "map_origin": {"latitude": 36.0, "longitude": 140.0, "altitude": 0.0}},
        [(20.0, 1.62), (570.0, 1.62), (-1000.0, -1000.0), (1000.0, 1000.0), (0.0, 0.0)])
# a declared scale other than the default
tm_case("TransverseMercator at 35.5, 139.5, scale 0.9999",
        {"projector_type": "TransverseMercator", "vertical_datum": "WGS84",
         "map_origin": {"latitude": 35.5, "longitude": 139.5, "altitude": 0.0}, "scale_factor": 0.9999},
        [(0.0, 0.0), (500.0, -800.0), (-2000.0, 1500.0)])
# MGRS 51RUH (the NTU twin's): zone 51 (columns S..Z for zone set 3: U is the 3rd, easting 300 km),
# row H in an odd zone: 700 km within the 2000 km cycle; band R (24 to 32 N): northing 2700 km
utm = pyproj.Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
e0, n0 = 300_000.0, 2_700_000.0
lon_c, lat_c = utm.transform(e0 + 50_000, n0 + 50_000)
assert 24.0 <= lat_c < 32.0, lat_c
vec = []
for x, y in ((0.0, 0.0), (99_999.0, 0.0), (0.0, 99_999.0), (99_999.0, 99_999.0), (50_000.0, 50_000.0),
             (52_123.4, 67_890.1)):
    lon, lat = utm.transform(e0 + x, n0 + y)
    vec.append({"x": x, "y": y, "lat": lat, "lon": lon})
cases.append({"name": "MGRS 51RUH", "projector": {"projector_type": "MGRS", "vertical_datum": "WGS84",
                                                  "mgrs_grid": "51RUH"},
              "points": vec})
json.dump({"generator": "aib src/aib/scripts/gen_projector_vectors.py",
           "maps": "carla-town-autoware-maps df499322639f279f588ff5f6fd661169027eb7a8", "pyproj": pyproj.__version__,
           "proj": pyproj.proj_version_str, "tolerance_m": 0.01, "cases": cases}, open(out, "w"), indent=1)
print(f"{sum(len(c['points']) for c in cases)} points in {len(cases)} cases -> {out}")
