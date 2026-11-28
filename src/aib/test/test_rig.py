"""The rig and ego config as the Isaac side loads them, and the sensor kit launch against
the spec's topics. Pure."""
import math
import re
import tempfile
import unittest
from pathlib import Path

from _paths import EGO, PKG, REPO, RIG
from aib.rig import EgoConfig, Rig

KIT = REPO / "deps/autoware_sim_vehicle/cars/tesla_model3/tesla_model3_sensor_kit_launch/launch"


class RigTest(unittest.TestCase):
    def test_the_shipped_rig(self):
        r = Rig.load(RIG)
        self.assertEqual(r.name, "tesla_model3")
        self.assertEqual(r.vehicle.mass, 1760.0)
        self.assertEqual(r.ecu.motor_peak_power, 194000.0)  # the onboard computers', from the spec
        self.assertFalse(hasattr(r.vehicle, "motor_peak_power"))
        self.assertEqual([s.name for s in r.of_type("lidar")], ["top"])
        self.assertEqual(r.of_type("camera"), [])
        self.assertEqual(len(r.of_type("camera", enabled_only=False)), 1)
        self.assertAlmostEqual(r.sensors["gnss"].period, 0.1)
        self.assertEqual(r.sensors["imu"].period, 0.0)
        self.assertIn("tires", r.assumptions)

    def test_the_mustangs_zed_left(self):
        r = Rig.load(PKG / "config/drift_mustang.rig.yaml")
        z = r.sensors["zed_left"]
        self.assertTrue(z.enabled)
        self.assertEqual(z.topic, "/sensing/camera/zed/image_raw")
        self.assertEqual(z.extra["info_topic"], "/sensing/camera/zed/camera_info")
        self.assertEqual((z.extra["width"], z.extra["height"]), (960, 600))
        self.assertAlmostEqual(z.period, 1 / 30)
        self.assertEqual(z.extra["rate_hz"], 30.0)
        fx = 480 / math.tan(math.radians(55))
        for a, b in zip(z.extra["k"], [fx, 0, 480, 0, fx, 300, 0, 0, 1]):
            self.assertAlmostEqual(a, b, places=6)
        self.assertEqual([x.name for x in r.of_type("camera")], ["zed_left"])

    def test_overrides(self):
        r = Rig.load(RIG, {"tire_friction": 0.7}, {"camera6": {"enabled": True},
                                                   "top": {"isaac": {"variant": "x"}}})
        self.assertEqual(r.vehicle.tire_friction, 0.7)
        self.assertTrue(r.sensors["camera6"].enabled)
        top = r.sensors["top"].isaac
        self.assertEqual((top["config"], top["variant"]), ("OS1", "x"))
        self.assertEqual(top["intensity"]["reflectance"]["Car"], 90)  # the spec's table stays
        with self.assertRaises(ValueError):
            Rig.load(RIG, None, {"radar": {"enabled": True}})
        with self.assertRaises(ValueError):
            Rig.load(RIG, None, {"top": {"translation": [0, 0, 0]}})
        r = Rig.load(RIG, None, None, {"esc_enabled": True})
        self.assertTrue(r.ecu.esc_enabled)
        with self.assertRaises(ValueError):  # a calibration field is not a PhysX one
            Rig.load(RIG, {"esc_enabled": True})

    def test_a_camera_without_k_gets_its_pinhole(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.yaml"
            text = RIG.read_text()
            text = re.sub(r"\n    k: \[[^\]]*\]", "", text, count=1)
            p.write_text(text)  # its spec: is relative to deps/, as the original's
            r = Rig.load(p, None, {"camera6": {"auto_exposure": True}})
        c = r.sensors["camera6"]
        self.assertTrue(c.extra["auto_exposure"])
        fx = 640 / math.tan(math.radians(74.5 / 2))
        for a, b in zip(c.extra["k"], [fx, 0, 640, 0, fx, 360, 0, 0, 1]):
            self.assertAlmostEqual(a, b, places=6)

    def test_a_stale_rig_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.yaml"
            p.write_text(RIG.read_text().replace("spec_hash: ", "spec_hash: 0", 1))
            with self.assertRaisesRegex(ValueError, "regenerate the rig"):
                Rig.load(p)

    def test_the_shipped_ego_config(self):
        cfg = EgoConfig.load(EGO)
        self.assertEqual(Path(cfg.rig), RIG)
        self.assertEqual(cfg.domain, 1)
        self.assertEqual(cfg.vehicle, {})
        self.assertEqual(cfg.ecu, {})
        self.assertEqual(cfg.load_rig().name, "tesla_model3")
        self.assertEqual(cfg.control_topic, "/control/command/control_cmd")
        self.assertEqual(cfg.lockstep, {})  # off: the world does not wait for Autoware

    def test_the_drift_cars_ego_config_takes_the_drift_controllers_command(self):
        cfg = EgoConfig.load(EGO.parent / "ego_drift_mustang.yaml")
        self.assertEqual(cfg.load_rig().name, "drift_mustang")
        self.assertEqual(cfg.control_topic, "/control/command/drift/control_cmd")
        # the world waits for NDT's pose each frame (scensim roadmap 057)
        self.assertEqual(cfg.lockstep["topic"], "/localization/pose_estimator/pose_with_covariance")

    def test_clock_offset_is_refused_by_name(self):
        # Roadmap 051: stamps are sim time + BASE_NS; the key is gone, not ignored.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.yaml"
            p.write_text(f"rig: {RIG}\nclock_offset: wall\n")
            with self.assertRaisesRegex(ValueError, "clock_offset is gone"):
                EgoConfig.load(p)

    def test_an_ego_config_needs_a_rig_and_known_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.yaml"
            p.write_text("domain: 2\n")
            with self.assertRaises(ValueError):
                EgoConfig.load(p)
            p.write_text("rig: x.yaml\nlidar: {}\n")
            with self.assertRaises(ValueError):
                EgoConfig.load(p)


class OutsideVehicleTest(unittest.TestCase):
    """A vehicle kept in another repository: its spec, rig, ego config, body and lidar
    profiles side by side, each naming the next relative to itself; the domain from
    ROS_DOMAIN_ID when the ego config names none."""

    def setUp(self):
        import os
        import shutil

        from aib import gen

        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        spec = REPO / "deps/sim_vehicle_spec/vehicles/tesla_model3.yaml"
        shutil.copy(spec, d / "car.yaml")
        (d / "isaac").mkdir()
        self.rig = d / "isaac/car.rig.yaml"
        self.assertEqual(gen.main(["--spec", str(d / "car.yaml"), "-o", str(self.rig)]), 0)
        self.assertEqual(gen.main(["--spec", str(d / "car.yaml"), "-o", str(self.rig),
                                   "--check"]), 0)
        self.d = d
        self.env = os.environ.get("ROS_DOMAIN_ID")

    def tearDown(self):
        import os

        if self.env is None:
            os.environ.pop("ROS_DOMAIN_ID", None)
        else:
            os.environ["ROS_DOMAIN_ID"] = self.env
        self.tmp.cleanup()

    def test_the_rig_names_its_spec_beside_it(self):
        self.assertIn("spec: ../car.yaml", self.rig.read_text())
        r = Rig.load(self.rig)
        self.assertEqual(r.name, "tesla_model3")
        self.assertEqual(r.ecu.motor_peak_power, 194000.0)

    def test_a_changed_spec_makes_the_rig_stale(self):
        from aib import gen

        spec = self.d / "car.yaml"
        spec.write_text(spec.read_text().replace("mass: 1760", "mass: 1761", 1))
        with self.assertRaisesRegex(ValueError, "regenerate the rig"):
            Rig.load(self.rig)
        self.assertEqual(gen.main(["--spec", str(spec), "-o", str(self.rig), "--check"]), 1)

    def test_ego_paths_and_domain(self):
        import os

        (self.d / "isaac/lidars").mkdir()
        text = self.rig.read_text()
        # the lidar's isaac block (the first sensor's): a profile of our own beside its config
        self.rig.write_text(re.sub(r"(\n    isaac:\n)", r"\1      usd: lidars/x.usda\n", text,
                                   count=1))
        ego = self.d / "isaac/ego.yaml"
        ego.write_text("rig: car.rig.yaml\nbody: body/entry.json\n")
        os.environ["ROS_DOMAIN_ID"] = "17"
        cfg = EgoConfig.load(ego)
        self.assertEqual(cfg.domain, 17)
        self.assertEqual(Path(cfg.body), (self.d / "isaac/body/entry.json").resolve())
        usds = [s.isaac.get("usd") for s in cfg.load_rig().sensors.values() if s.isaac.get("usd")]
        self.assertEqual(usds, [str((self.d / "isaac/lidars/x.usda").resolve())])
        os.environ.pop("ROS_DOMAIN_ID")
        self.assertEqual(EgoConfig.load(ego).domain, 0)
        ego.write_text("rig: car.rig.yaml\ndomain: 3\n")
        self.assertEqual(EgoConfig.load(ego).domain, 3)


class GroundTruthConfigTest(unittest.TestCase):
    """The ego config's ground_truth (scensim roadmap 063 e): off by default, defaults filled."""

    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "e.yaml"
            p.write_text(f"rig: {RIG}\n" + text)
            return EgoConfig.load(p)

    def test_off_by_default(self):
        self.assertEqual(EgoConfig.load(EGO).ground_truth, {})

    def test_on_fills_the_defaults(self):
        gt = self.load("ground_truth: {pose_estimate: true}\n").ground_truth
        self.assertEqual(gt["topic"], "/ground_truth/odom")
        self.assertTrue(gt["pose_estimate"])
        self.assertEqual(gt["covariance"], [0.01, 0.0003])
        gt = self.load("ground_truth: {topic: /gt}\n").ground_truth
        self.assertEqual(gt["topic"], "/gt")
        self.assertFalse(gt["pose_estimate"])

    def test_unknown_keys_are_refused(self):
        with self.assertRaisesRegex(ValueError, "ground_truth: unknown"):
            self.load("ground_truth: {frame: map}\n")

    def test_a_pose_estimate_and_a_lockstep_on_the_same_topic_are_refused(self):
        with self.assertRaisesRegex(ValueError, "would wait on itself"):
            self.load("ground_truth: {pose_estimate: true}\nlockstep: {timeout_s: 1.0}\n")
        # a lockstep on another topic is fine
        cfg = self.load("ground_truth: {pose_estimate: true}\nlockstep: {topic: /x}\n")
        self.assertTrue(cfg.ground_truth["pose_estimate"])


class SensorKitLaunchTest(unittest.TestCase):
    """Autoware's sensing launch consumes the topics the rig publishes."""

    def test_topics_match(self):
        r = Rig.load(RIG)
        text = {p.name: p.read_text() for p in KIT.iterdir()}
        lidar = r.sensors["top"].topic
        self.assertIn(f'("input", "{lidar}")', text["pointcloud_preprocessor.launch.py"])
        imu = r.sensors["imu"].topic
        self.assertTrue(imu.startswith("/sensing/imu/"))
        rel = imu[len("/sensing/imu/"):]
        # The topics are launch arguments (Autoware's sample kit names them imu_raw_name);
        # their defaults are the rig's, and every consumer reads the argument.
        self.assertEqual(re.findall(r'name="imu_raw_name" default="([^"]+)"', text["imu.launch.xml"]), [rel])
        self.assertEqual(re.findall(r'name="input_(?:topic|imu_raw)" value="([^"]+)"',
                                    text["imu.launch.xml"]), ["$(var imu_raw_name)"] * 2)
        gnss = r.sensors["gnss"].topic
        rel = gnss[len("/sensing/gnss/"):]
        self.assertEqual(re.findall(r'name="gnss_fix_name" default="([^"]+)"', text["gnss.launch.xml"]), [rel])
        self.assertIn('<remap from="fix" to="$(var gnss_fix_name)"/>', text["gnss.launch.xml"])


if __name__ == "__main__":
    unittest.main()


def test_the_rig_makes_its_control_component(tmp_path, monkeypatch):
    """scensim roadmap 091 step 2: a rig's car steps a control component; by default the shipped
    onboard computers on the rig's calibration (its overrides kept), else a run's own, found by
    entry point from a workspace outside aib."""
    import sys
    import textwrap

    import pytest
    from sim_vehicle_ecu import component as components

    r = Rig.load(RIG, None, None, {"esc_enabled": True})
    c = r.make_component()
    assert isinstance(c, components.EcuComponent) and c.calibration is r.ecu
    assert c.calibration.esc_enabled and c.commands == "autoware"

    site = tmp_path / "site"
    (site / "my_racer_model").mkdir(parents=True)
    (site / "my_racer_model" / "__init__.py").write_text(textwrap.dedent('''
        class Sample:
            commands = "autoware"
            driven = (False, False, True, True)

            def __init__(self, vehicle, params, seed):
                self.vehicle, self.params, self.seed = vehicle, params, seed

            def reset(self):
                pass

            def step(self, command, sensors, dt):
                raise AssertionError("not stepped here")
    '''))
    info = site / "my_racer_model-0.1.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: my-racer-model\nVersion: 0.1.0\n")
    (info / "entry_points.txt").write_text(
        "[scensim.vehicle_components]\nmy_racer.sample = my_racer_model:Sample\n")
    monkeypatch.syspath_prepend(str(site))
    try:
        r = Rig.load(RIG, component={"entry_point": "my_racer.sample", "params": {"k": 2}})
        c = r.make_component(seed=5)
        assert type(c).__name__ == "Sample" and c.params == {"k": 2} and c.seed == 5
        assert c.vehicle["name"] == r.spec["name"]
        with pytest.raises(ValueError, match="component: {entry_point"):
            Rig.load(RIG, component={"name": "my_racer.sample"})
        with pytest.raises(components.ComponentError, match="no control component 'nope'"):
            Rig.load(RIG, component={"entry_point": "nope"}).make_component()
    finally:
        sys.modules.pop("my_racer_model", None)


def test_a_component_is_made_with_the_run_s_session_record(monkeypatch):
    """scensim 091 W12: the run's session record rides in the ego config's `component.session`,
    and the rig hands it to sim_vehicle_ecu when it makes the component, which gives a class
    the session inputs it declares at construction (and refuses one the record lacks)."""
    import pytest
    from sim_vehicle_ecu import component as components

    seen = {}
    monkeypatch.setattr(components, "for_spec",
                        lambda spec, seed=0, component=None, base=None, session=None:
                        seen.update(session=session) or "c")
    rec = {"schema": "scensim.session/1", "battery": {"voltage_v": 7.6, "source": "measured"}}
    r = Rig.load(RIG, component={"entry_point": "my_racer.sample", "session": rec})
    assert r.make_component() == "c" and seen["session"] == rec
    Rig.load(RIG, component={"entry_point": "my_racer.sample"}).make_component()
    assert seen["session"] is None
    with pytest.raises(ValueError, match="session"):
        Rig.load(RIG, component={"entry_point": "my_racer.sample", "session": "7.6 V"})


def test_an_ego_config_names_a_component(tmp_path):
    cfg_path = tmp_path / "ego.yaml"
    cfg_path.write_text(f"rig: {RIG}\ncomponent: {{entry_point: my_racer.sample}}\n")
    cfg = EgoConfig.load(cfg_path)
    assert cfg.component == {"entry_point": "my_racer.sample"}


def test_truth_values_are_the_declared_inputs_only():
    """scensim 091 (the user): a component's declared ground truth, from the state the world
    left; only the names it declared."""
    import math

    import pytest
    from aib.control import State, truth_values

    st = State(position=(12.0, 5.0, 0.0), quat=(math.cos(0.25), 0.0, 0.0, math.sin(0.25)),
               velocity=(4.0, 0.4, 0.0), angular_velocity=(0.0, 0.0, 0.3))
    wheels = [type("W", (), {"slip": s})() for s in (0.1, 0.1, 0.2, 0.2)]
    got = truth_values(st, wheels, ("pose", "speed", "wheel_slip", "sideslip", "yaw_rate"),
                       map_origin=(10.0, 0.0, 0.0))
    assert set(got) == {"pose", "speed", "wheel_slip", "sideslip", "yaw_rate"}
    assert got["pose"] == pytest.approx((2.0, 5.0, 0.5))
    assert got["speed"] == 4.0 and got["yaw_rate"] == 0.3
    assert got["wheel_slip"] == (0.1, 0.1, 0.2, 0.2)
    assert got["sideslip"] == pytest.approx(math.atan2(0.4, 4.0))
    assert truth_values(st, None, ()) == {}
    with pytest.raises(ValueError, match="no ground truth 'mind'"):
        truth_values(st, wheels, ("mind",))


def test_a_component_s_paths_reach_the_interpreter(tmp_path):
    """scensim 091 W3: the ego config names where the component's package is (a workspace's
    site-packages, built for another Python); the rig puts it on sys.path before looking it up."""
    import sys

    site = tmp_path / "ws" / "install" / "my_racer_model" / "lib" / "python3.10" / "site-packages"
    (site / "my_racer_paths").mkdir(parents=True)
    (site / "my_racer_paths" / "__init__.py").write_text(
        "class Sample:\n    commands = 'autoware'\n\n    def __init__(self, vehicle, params, seed):\n"
        "        pass\n\n    def reset(self):\n        pass\n\n    def step(self, c, s, dt):\n"
        "        return None\n")
    info = site / "my_racer_paths-0.1.0.egg-info"   # what colcon's setup.py install writes
    info.mkdir()
    (info / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: my-racer-paths\nVersion: 0.1.0\n")
    (info / "entry_points.txt").write_text(
        "[scensim.vehicle_components]\nmy_racer.paths = my_racer_paths:Sample\n")
    try:
        r = Rig.load(RIG, component={"entry_point": "my_racer.paths", "paths": [str(site)]})
        assert type(r.make_component()).__name__ == "Sample"
        assert str(site) in sys.path
    finally:
        sys.modules.pop("my_racer_paths", None)
        if str(site) in sys.path:
            sys.path.remove(str(site))


def test_a_component_s_files_are_relative_to_the_spec(monkeypatch):
    """scensim 091 W4: the fmu component's unit (params.fmu) is named relative to the spec, so
    the rig hands its spec's directory to the component as `base`."""
    from pathlib import Path

    from sim_vehicle_ecu import component as components

    seen = {}
    monkeypatch.setattr(components, "for_spec",
                        lambda spec, seed=0, component=None, base=None: seen.update(base=base) or "c")
    r = Rig.load(RIG, component={"entry_point": "fmu", "params": {"fmu": "models/racer.fmu"}})
    assert Path(r.spec_file).is_file() and Path(r.spec_file).name.endswith((".yaml", ".yml"))
    assert r.make_component() == "c" and seen["base"] == str(Path(r.spec_file).parent)
