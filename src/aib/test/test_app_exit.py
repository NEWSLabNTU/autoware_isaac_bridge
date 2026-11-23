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


def run_fake(module, *args):
    """The entry point under aib.fake_kit: (exit status, stdout + stderr)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(_paths.PKG)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    env.pop("ISAAC_KIT_ARGS", None)
    r = subprocess.run([sys.executable, "-m", "aib.fake_kit", module, *args], env=env,
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


class EntryPointTest(unittest.TestCase):
    def test_steptest_that_raises_exits_1_and_says_why(self):
        with tempfile.TemporaryDirectory() as d:
            code, output = run_fake("aib.steptest", "--out", d, "--label", "fake")
            summary = json.loads((Path(d) / "fake/summary.json").read_text())
        self.assertEqual(code, 1, output)
        self.assertIn(fake_kit.FORCED, output)  # its traceback
        self.assertEqual(summary["error"], f"RuntimeError: {fake_kit.FORCED}")
        self.assertEqual(summary["tests"], {})

    def test_ground_check_that_raises_exits_1(self):
        code, output = run_fake("aib.ground_check")
        self.assertEqual(code, 1, output)
        self.assertIn(fake_kit.FORCED, output)
        line = next(x for x in output.splitlines() if x.startswith("GROUND-CHECK "))
        self.assertFalse(json.loads(line[len("GROUND-CHECK "):])["pass"])


if __name__ == "__main__":
    unittest.main()
