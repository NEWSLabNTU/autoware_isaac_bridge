"""aib's dependencies that are not Python packages of their own on the path: its submodules
under `deps/` (`sim_vehicle_spec`, the spec and its library; `sim_vehicle_ecu`, the car's
onboard computers). `use` puts them on `sys.path`, so every host that imports aib -- the
world server, `aib.runner`, the tests, Isaac's Python or the system's -- finds the same
checkouts, at the commits aib pins."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEPS = REPO / "deps"


def use(*names):
    for name in names:
        path = str(DEPS / name)
        if path not in sys.path:
            sys.path.insert(0, path)
