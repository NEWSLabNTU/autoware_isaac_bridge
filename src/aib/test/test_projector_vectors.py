"""The GNSS fix through the map's projector (scensim issue 195, the GNSS offset item: the fix not inverted
through the map's projector, acb and aib). aib.frames.Projector, built from a map_projector_info as the
map declares it, against PROJ's latitude/longitude for the same map x/y (test/data/gnss_projector_vectors.json,
shared with acb; scripts/gen_projector_vectors.py writes it). Pure: no pyproj here."""
import json
import math
import unittest

from _paths import PKG
from aib import frames

VECTORS = json.loads((PKG / "test/data/gnss_projector_vectors.json").read_text())
M_PER_DEG = 111_320.0


def error_m(lat, lon, want_lat, want_lon):
    return math.hypot((lat - want_lat) * M_PER_DEG,
                      (lon - want_lon) * M_PER_DEG * math.cos(math.radians(want_lat)))


class ProjectorVectorsTest(unittest.TestCase):
    def test_every_case_within_a_centimetre(self):
        tol = VECTORS["tolerance_m"]
        self.assertEqual(tol, 0.01)
        for case in VECTORS["cases"]:
            p = frames.Projector.from_info(case["projector"])
            for v in case["points"]:
                lat, lon, _ = p.lat_lon_alt(v["x"], v["y"], 0.0)
                e = error_m(lat, lon, v["lat"], v["lon"])
                with self.subTest(case=case["name"], x=v["x"], y=v["y"]):
                    self.assertLess(e, tol, f"{e:.4f} m")

    def test_the_vectors_cover_every_carla_town_and_the_other_kinds(self):
        names = " ".join(c["name"] for c in VECTORS["cases"])
        for town in ("Town01", "Town02", "Town03", "Town05", "Town10"):
            self.assertIn(town, names)
        self.assertIn("MGRS", names)
        self.assertIn("course fixture", names)
        # the towns as scensim's pin declares them (scale 1.0), and the default path: a TM
        # projector without scale_factor, which the reader must fill in itself
        towns = [c for c in VECTORS["cases"] if c["name"].startswith("carla Town")]
        self.assertTrue(any(c["projector"].get("scale_factor") == 1.0 for c in towns))
        self.assertTrue(any(c["projector"]["projector_type"] == "TransverseMercator"
                            and "scale_factor" not in c["projector"] for c in towns))

    def test_a_declared_scale_factor_is_used_and_the_default_is_utms(self):
        p = frames.Projector.from_info({"projector_type": "TransverseMercator",
                                        "map_origin": {"latitude": 0.0, "longitude": 0.0}})
        self.assertAlmostEqual(p.scale, 0.9996)
        p = frames.Projector.from_info({"projector_type": "TransverseMercator", "scale_factor": 1.0,
                                        "map_origin": {"latitude": 0.0, "longitude": 0.0}})
        self.assertEqual(p.scale, 1.0)

    def test_a_local_map_loads_but_a_fix_through_it_is_refused(self):
        # the golf cart's basement maps are Local: the world loads, the GNSS cannot publish
        for kind in ("LocalCartesianUTM", "LocalCartesian", "Local", "local"):
            p = frames.Projector.from_info({"projector_type": kind})
            with self.assertRaisesRegex(ValueError, kind):
                p.lat_lon_alt(1.0, 2.0, 0.0)

    def test_an_unknown_projector_is_refused_and_none_is_no_file(self):
        with self.assertRaisesRegex(ValueError, "Polar"):
            frames.Projector.from_info({"projector_type": "Polar"})
        self.assertEqual(frames.Projector.from_info({}).lat_lon_alt(10.0, 20.0, 0.0),
                         frames.Projector().lat_lon_alt(10.0, 20.0, 0.0))
