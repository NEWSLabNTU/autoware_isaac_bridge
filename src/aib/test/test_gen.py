"""The generator: the rig from the spec, its header, the staleness check, and that the rig and
Autoware's packages for the same car (autoware_sim_vehicle) agree. Pure."""
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from _paths import REPO, RIG, SPEC
from aib import gen
from aib.frames import kit_transform
from aib.params import read_vehicle_info
from aib.rig import Rig

CAR = REPO / "deps/autoware_sim_vehicle/cars/tesla_model3"
RIG_REL = "src/aib/config/tesla_model3.rig.yaml"


class CheckTest(unittest.TestCase):
    def test_the_repository_is_up_to_date(self):
        # The CI check: fails when the spec moved without `just generate`, or the rig was
        # edited by hand.
        self.assertEqual(gen.stale(), [])

    def test_the_outputs_are_one_rig_per_spec(self):
        specs = sorted((REPO / "deps/sim_vehicle_spec/vehicles").glob("*.yaml"))
        self.assertEqual(sorted(gen.outputs()),
                         sorted(f"src/aib/config/{p.stem}.rig.yaml" for p in specs))
        self.assertIn(RIG_REL, gen.outputs())

    def test_a_hand_edit_is_stale_and_generate_fixes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, specs = Path(tmp), Path(tmp) / "specs"
            specs.mkdir()
            shutil.copy(SPEC, specs / SPEC.name)
            outs = gen.outputs(specs, root)
            self.assertEqual(sorted(gen.stale(specs, root)), sorted(outs))
            self.assertEqual(sorted(gen.write(specs, root)), sorted(outs))
            self.assertEqual(gen.stale(specs, root), [])
            rig = root / RIG_REL
            rig.write_text(rig.read_text().replace("3.005", "3.1"))
            self.assertEqual(gen.stale(specs, root), [RIG_REL])
            args = ["--specs", str(specs), "--root", str(root)]
            self.assertEqual(gen.main(args + ["--check"]), 1)
            self.assertEqual(gen.main(args), 0)
            self.assertEqual(gen.main(args + ["--check"]), 0)

    def test_the_rig_names_its_spec_and_hash(self):
        text = gen.outputs()[RIG_REL]
        head = text.splitlines()[0]
        self.assertIn("Generated from sim_vehicle_spec/vehicles/tesla_model3.yaml", head)
        self.assertIn("do not edit", head)
        rig = Rig.load(RIG)
        self.assertEqual(rig.spec_hash, gen.specmod.spec_hash(gen.specmod.load(SPEC)))
        self.assertIn(rig.spec_hash, head)


class AgreesWithAutowareTest(unittest.TestCase):
    """The rig and Autoware's packages are generated from one spec, by two generators: they
    must place the car and its sensors the same."""

    def setUp(self):
        self.rig = Rig.load(RIG)

    def test_the_spec_hashes_match(self):
        text = (CAR / "tesla_model3_description/config/spec_hash.yaml").read_text()
        self.assertIn(f"spec_hash: {self.rig.spec_hash}", text)

    def test_vehicle_info_is_the_rigs_geometry(self):
        path = CAR / "tesla_model3_description/config/vehicle_info.param.yaml"
        info = yaml.safe_load(path.read_text())["/**"]["ros__parameters"]
        self.assertEqual(read_vehicle_info(path)["wheel_base"], info["wheel_base"])
        v = self.rig.vehicle
        for k in ("wheel_base", "wheel_tread", "wheel_radius", "wheel_width", "max_steer_angle"):
            self.assertAlmostEqual(info[k], getattr(v, k), msg=k)
        length = info["wheel_base"] + info["front_overhang"] + info["rear_overhang"]
        self.assertAlmostEqual(length, self.rig.box_dims[0], places=6)
        centre = (info["wheel_base"] + info["front_overhang"] - info["rear_overhang"]) / 2
        self.assertAlmostEqual(centre, self.rig.box_center[0], places=6)

    def test_the_calibration_places_every_sensor_where_the_rig_does(self):
        d = CAR / "tesla_model3_sensor_kit_description/config"
        rig_keys = gen.specmod.sensors(gen.specmod.load(SPEC))
        for name, s in self.rig.sensors.items():
            t, q = kit_transform(d / "sensors_calibration.yaml", d / "sensor_kit_calibration.yaml",
                                 rig_keys[name]["key"])
            for a, b in zip(t, s.translation):
                self.assertAlmostEqual(a, b, places=8, msg=name)
            for a, b in zip(q, s.quat):
                self.assertAlmostEqual(a, b, places=8, msg=name)


if __name__ == "__main__":
    unittest.main()
