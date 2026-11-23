"""Run an Isaac entry point with no Isaac, to test how it exits (scensim issue 118).

    [FAKE_KIT_EXTRA=zmq,...] python -m aib.fake_kit <module> [args ...]

Every module under omni, pxr, carb, usdrt and isaacsim imports as a stand-in whose attributes
are MagicMocks, except: `isaacsim.SimulationApp`, whose `close()` ends the process as Isaac
6.2's does (with 0, or the `exit_code` it is given) and which closes itself at exit as 6.2's
does; `isaacsim.core.api.World`, which raises FORCED; and carb's settings, which hold none. So
the entry point starts its app, fails where it first builds Isaac's World, and the process's
exit status is what the entry point does with a failure. Nothing here imports Isaac.
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


class _Settings:  # carb's settings: none set, so Kit names no log file
    def get(self, path):
        return None


OVERRIDES = {"isaacsim": {"SimulationApp": SimulationApp},
             "isaacsim.core.api": {"World": _world},
             "carb.settings": {"get_settings": _Settings}}


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
        for attr, value in OVERRIDES.get(module.__name__, {}).items():
            setattr(module, attr, value)


def install(extra=()):
    """Fake Isaac's modules in this interpreter, and `extra` top-level ones, for the imports
    that follow."""
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder(extra))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    install([m for m in os.environ.get(EXTRA_ENV, "").split(",") if m])
    sys.argv = [argv[0]] + argv[1:]
    runpy.run_module(argv[0], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
