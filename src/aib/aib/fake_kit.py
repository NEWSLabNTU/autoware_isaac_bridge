"""Run an Isaac entry point with no Isaac, to test how it exits (scensim issue 118).

    [FAKE_KIT_EXTRA=zmq,...] [FAKE_KIT_FAIL=module.name] python -m aib.fake_kit <module> [args]

Every module under omni, pxr, carb, usdrt and isaacsim imports as a stand-in whose attributes
are MagicMocks, except: `isaacsim.SimulationApp`, whose `close()` ends the process as Isaac
6.2's does (with 0, or the `exit_code` it is given) and which closes itself at exit as 6.2's
does; `isaacsim.core.api.World`, which raises FORCED; and carb's settings, which hold none. So
the entry point starts its app, fails where it first builds Isaac's World, and the process's
exit status is what the entry point does with a failure. Nothing here imports Isaac.

An entry point that builds no World names its own failure point (FAKE_KIT_FAIL: a callable in
a faked module, `module.name`, or a faked module's import, `import:module`; the message
`forced(target)`). With isaaclab faked, `isaaclab.app.AppLauncher` gives the stand-in app;
with wheeledlab_rl faked, so does `wheeledlab_rl.startup.startup`.
"""
import atexit
import importlib.abc
import importlib.machinery
import os
import runpy
import sys
import types
from unittest import mock

FORCED = "fake_kit: forced failure building the World"
FAKED = ("omni", "pxr", "carb", "usdrt", "isaacsim")
EXTRA_ENV = "FAKE_KIT_EXTRA"  # more top-level modules to stand in for, comma separated (zmq)
FAIL_ENV = "FAKE_KIT_FAIL"  # in place of World: a faked callable (module.name) raises, or a
# faked module's import (import:module)
_FAIL_IMPORT = []


class SimulationApp:
    def __init__(self, config=None, *args, **kwargs):
        self.config = config
        self._exiting = False
        atexit.register(self._atexit_close)  # as Isaac 6.2's: an escaped exception ends with 0

    def _atexit_close(self):
        if not self._exiting:
            self.close()

    def is_running(self):
        return True

    def update(self):
        pass

    def close(self, wait_for_replicator=True, skip_cleanup=False, exit_code=0):
        # Isaac 6.2's: it ends the process, with 0 unless it is given another status
        self._exiting = True
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


def _world(*args, **kwargs):
    raise RuntimeError(FORCED)


def forced(target):
    """The message a FAKE_KIT_FAIL target raises."""
    return f"fake_kit: forced failure at {target}"


def _fail_at(target):
    def fail(*args, **kwargs):
        raise RuntimeError(forced(target))
    return fail


class AppLauncher:
    """Isaac Lab's: its `app` is the stand-in SimulationApp."""

    def __init__(self, *args, **kwargs):
        self.app = SimulationApp()

    @staticmethod
    def add_app_launcher_args(parser):
        pass


def _wheeledlab_startup(parser=None, **kwargs):
    """WheeledLab's startup: the stand-in app, and the parser's arguments."""
    return SimulationApp(), (parser.parse_args() if parser is not None else None)


class _Settings:  # carb's settings: none set, so Kit names no log file
    def get(self, path):
        return None


OVERRIDES = {"isaacsim": {"SimulationApp": SimulationApp},
             "isaacsim.core.api": {"World": _world},
             "carb.settings": {"get_settings": _Settings},
             "isaaclab.app": {"AppLauncher": AppLauncher},  # with isaaclab in FAKE_KIT_EXTRA
             "wheeledlab_rl.startup": {"startup": _wheeledlab_startup}}  # with wheeledlab_rl


class _Module(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = mock.MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, extra=()):
        self.faked = set(FAKED) | set(extra)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in self.faked:
            return None
        return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = _Module(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        if _FAIL_IMPORT and module.__name__ == _FAIL_IMPORT[0]:
            raise ImportError(forced(f"import:{module.__name__}"))
        for attr, value in OVERRIDES.get(module.__name__, {}).items():
            setattr(module, attr, value)


def install(extra=(), fail=""):
    """Fake Isaac's modules in this interpreter, and `extra` top-level ones, for the imports
    that follow; `fail` (module.name) raises where it is called, in place of World."""
    if fail:
        OVERRIDES["isaacsim.core.api"] = {}
        if fail.startswith("import:"):  # the faked module's import raises
            _FAIL_IMPORT[:] = [fail[len("import:"):]]
        else:
            module, name = fail.rsplit(".", 1)
            OVERRIDES.setdefault(module, {})[name] = _fail_at(fail)
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder(extra))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    install([m for m in os.environ.get(EXTRA_ENV, "").split(",") if m],
            os.environ.get(FAIL_ENV, ""))
    sys.argv = [argv[0]] + argv[1:]
    runpy.run_module(argv[0], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
