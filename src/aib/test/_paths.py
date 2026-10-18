"""Shared paths for the tests: the package's source and the repository."""
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parents[1]
SPEC = REPO / "vehicles/tesla_model3.yaml"
RIG = PKG / "config/tesla_model3.rig.yaml"
EGO = PKG / "config/ego.yaml"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))
