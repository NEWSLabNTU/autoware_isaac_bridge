"""Simulation time: the integer every part of a run computes (roadmap 051). Pure."""
import unittest

import _paths  # noqa: F401  sys.path
from aib.sim_time import BASE_NS, nanos, sim_ns

# swb_world_protocol's sim_time table (and acb's clock.rs), case for case.
NANOS_CASES = [
    (0.0, 0), (0.05000000074505806, 50_000_001), (5e-10, 1), (1.5e-9, 2), (2.5e-9, 3),
    (-2.5e-9, -3), (3.5e-9, 4), (0.10000000149011612, 100_000_001),
    (0.15000000223517418, 150_000_002), (67.0000009983778, 67_000_000_998),
    (171.60000255703926, 171_600_002_557), (1000.0000149011612, 1_000_000_014_901),
    (237000.05000000075, 237_000_050_000_001), (237171.60000255704, 237_171_600_002_557),
    (1790611998.05, 1_790_611_998_049_999_872),
]


class SimTimeTest(unittest.TestCase):
    def test_nanos_matches_the_shared_table(self):
        for secs, ns in NANOS_CASES:
            self.assertEqual(nanos(secs), ns, secs)

    def test_sim_time_is_past_the_base(self):
        self.assertEqual(BASE_NS, 1_000_000_000_000)
        self.assertEqual(sim_ns(0.05000000074505806), BASE_NS + 50_000_001)


if __name__ == "__main__":
    unittest.main()
