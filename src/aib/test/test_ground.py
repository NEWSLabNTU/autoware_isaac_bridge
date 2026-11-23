"""The flat ground aib builds itself (scensim issue 117): nothing in aib asks Isaac for its asset
root, which is NVIDIA's S3 bucket, and the plane keeps the contact values Isaac's default
plane had. Pure: reads the sources, imports no Isaac."""
import re
import unittest
from pathlib import Path

_paths = __import__("_paths")  # sys.path, for direct runs
from aib.isaac import ground

PACKAGE = Path(__file__).resolve().parents[1] / "aib"
# Isaac's calls that fetch from the asset root: the default plane, and the root itself
ASSET_ROOT_CALLS = re.compile(r"\b(add_default_ground_plane|get_assets_root_path)\s*\(")
# aib.ground_check builds Isaac's default plane on purpose, behind --isaac-default: the before
EXEMPT = {"ground_check.py"}


class NoAssetRootTest(unittest.TestCase):
    def test_no_module_fetches_isaacs_asset_root(self):
        found = [f"{p.relative_to(PACKAGE.parent)}:{i}"
                 for p in sorted(PACKAGE.rglob("*.py")) if p.name not in EXEMPT
                 for i, line in enumerate(p.read_text().splitlines(), 1)
                 if ASSET_ROOT_CALLS.search(line.split("#")[0])]
        self.assertEqual(found, [], "these reach NVIDIA's S3 at start; use aib.isaac.ground")


class GroundTest(unittest.TestCase):
    def test_contacts_are_isaacs_default_planes(self):
        # Isaac 6.2's add_default_ground_plane: static and dynamic friction 0.5, restitution 0.8.
        # The tyres never read them (aib's friction tables name no ground material).
        self.assertEqual((ground.STATIC_FRICTION, ground.DYNAMIC_FRICTION), (0.5, 0.5))
        self.assertEqual(ground.RESTITUTION, 0.8)

    def test_the_world_servers_path(self):
        self.assertEqual(ground.PATH, "/World/defaultGroundPlane")


if __name__ == "__main__":
    unittest.main()
