"""A run that raises exits nonzero (scensim issue 118): the helper, pure, and aib's two Isaac
entry points under aib.fake_kit, which stands in for Isaac (no Isaac needed)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_paths = __import__("_paths")  # sys.path, for direct runs
from aib import app_exit, fake_kit


class OldApp:
    """An app whose close takes no exit status (before Isaac 6.2)."""

    def __init__(self):
        self.closed = []

    def close(self):
        self.closed.append(0)


class App62(OldApp):
    """Isaac 6.2's: close(exit_code=...); this one returns rather than end the process."""

    def close(self, exit_code=0):
        self.closed.append(exit_code)


class Exited(Exception):
    pass


def fake_exit(code):
    raise Exited(code)


class CloseTest(unittest.TestCase):
    def test_a_failed_run_hands_its_status_to_close_then_exits(self):
        app = App62()
        with self.assertRaises(Exited) as e:
            app_exit.close(app, True, _exit=fake_exit)
        self.assertEqual(e.exception.args, (1,))
        self.assertEqual(app.closed, [1])

    def test_an_old_app_is_not_closed_with_0(self):
        app = OldApp()
        with self.assertRaises(Exited) as e:
            app_exit.close(app, 2, _exit=fake_exit)
        self.assertEqual(e.exception.args, (2,))
        self.assertEqual(app.closed, [])

    def test_a_passed_run_closes(self):
        for passed in (False, 0):
            app = App62()
            app_exit.close(app, passed, _exit=fake_exit)
            self.assertEqual(app.closed, [0])


class CodeOfTest(unittest.TestCase):
    def test_codes(self):
        with open(os.devnull, "w") as null:
            stderr, sys.stderr = sys.stderr, null
            try:
                codes = [app_exit.code_of(e) for e in (
                    SystemExit(), SystemExit(0), SystemExit(2), SystemExit("refused"),
                    KeyboardInterrupt(), RuntimeError("boom"))]
            finally:
                sys.stderr = stderr
        self.assertEqual(codes, [0, 0, 2, 1, app_exit.INTERRUPTED, 1])


# the ROS modules aib.runner's node imports, stood in for too
ROS = ("rclpy,rosgraph_msgs,sensor_msgs,std_msgs,geometry_msgs,nav_msgs,builtin_interfaces,"
       "diagnostic_msgs,tf2_msgs,autoware_vehicle_msgs,autoware_control_msgs,tier4_vehicle_msgs,"
       "autoware_perception_msgs,autoware_adapi_v1_msgs,autoware_sensing_msgs")


def run_fake(module, *args, extra=""):
    """The entry point under aib.fake_kit: (exit status, stdout + stderr)."""
    env = dict(os.environ, FAKE_KIT_EXTRA=extra, PYTHONPATH=os.pathsep.join(
        [str(_paths.PKG)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    env.pop("ISAAC_KIT_ARGS", None)
    r = subprocess.run([sys.executable, "-m", "aib.fake_kit", module, *args], env=env,
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


GUARDED = """
import sys
from aib import fake_kit
fake_kit.install()
from isaacsim import SimulationApp
from aib import app_exit
if sys.argv[1] == "wrapper":  # guarded before the app, holding none
    app_exit.guard()
app = SimulationApp()
if sys.argv[1] != "wrapper":
    app_exit.guard(app)
if sys.argv[1] in ("raise", "wrapper"):
    raise RuntimeError("raised at module level")


def main():
    if sys.argv[1] == "exit":
        raise SystemExit("main refused")
    if sys.argv[1] == "exit0":
        raise SystemExit(0)


app_exit.run(main)
app.close()
"""


class GuardTest(unittest.TestCase):
    def run_script(self, how):
        env = dict(os.environ, PYTHONPATH=str(_paths.PKG))
        return subprocess.run([sys.executable, "-c", GUARDED, how], env=env,
                              capture_output=True, text=True, timeout=60)

    def test_an_escaped_exception_ends_with_1(self):
        r = self.run_script("raise")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("raised at module level", r.stderr)  # Python's own traceback

    def test_a_wrapper_guarded_before_the_app_ends_with_1(self):
        r = self.run_script("wrapper")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("raised at module level", r.stderr)

    def test_a_system_exit_out_of_main_keeps_its_status(self):
        r = self.run_script("exit")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("main refused", r.stderr)
        self.assertEqual(self.run_script("exit0").returncode, 0)

    def test_a_script_that_ends_closes_with_0(self):
        r = self.run_script("pass")
        self.assertEqual(r.returncode, 0, r.stderr)


class EntryPointTest(unittest.TestCase):
    def test_steptest_that_raises_exits_1_and_says_why(self):
        with tempfile.TemporaryDirectory() as d:
            code, output = run_fake("aib.steptest", "--out", d, "--label", "fake")
            summary = json.loads((Path(d) / "fake/summary.json").read_text())
        self.assertEqual(code, 1, output)
        self.assertIn(fake_kit.FORCED, output)  # its traceback
        self.assertEqual(summary["error"], f"RuntimeError: {fake_kit.FORCED}")
        self.assertEqual(summary["tests"], {})

    def test_runner_that_raises_exits_1(self):
        code, output = run_fake("aib.runner", "--ego", str(_paths.EGO), extra=ROS)
        self.assertEqual(code, 1, output)
        self.assertIn(fake_kit.FORCED, output)

    def test_ground_check_that_raises_exits_1(self):
        code, output = run_fake("aib.ground_check")
        self.assertEqual(code, 1, output)
        self.assertIn(fake_kit.FORCED, output)
        line = next(x for x in output.splitlines() if x.startswith("GROUND-CHECK "))
        self.assertFalse(json.loads(line[len("GROUND-CHECK "):])["pass"])


if __name__ == "__main__":
    unittest.main()
