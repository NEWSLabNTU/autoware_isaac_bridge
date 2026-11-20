"""The bump stop at the real travel (scensim issue 077): its force from PhysX's jounce. Pure."""
import unittest

_paths = __import__("_paths")  # sys.path, for direct runs
from aib.bump_stop import BumpStop
from aib.params import VehicleParams

DT = 1.0 / 60


def stop(**kw):
    return BumpStop(**{"travel": 0.20, "bump": 0.06, "rate": 200e3, "damping": 15e3, **kw})


class ForceTest(unittest.TestCase):
    def test_nothing_short_of_the_engagement(self):
        s = stop()
        self.assertEqual(s.force(0.097, 0.0), 0.0)  # at rest
        self.assertEqual(s.force(0.14, 0.2), 0.0)  # at the engagement
        self.assertEqual(s.force(0.0, -1.0), 0.0)  # full droop

    def test_progressive_to_rate_times_bump_at_full_compression(self):
        s = stop()
        half, full = s.force(0.17, 0.0), s.force(0.20, 0.0)
        self.assertAlmostEqual(full, 200e3 * 0.06)
        self.assertAlmostEqual(half, full / 4)  # the square of the compression

    def test_never_pulls_and_is_capped(self):
        s = stop()
        self.assertEqual(s.force(0.141, -5.0), 0.0)  # rebound damping does not pull the body
        self.assertEqual(s.force(0.45, 0.0), s.cap * s.rate * s.bump)

    def test_the_damper_sees_at_most_v_max(self):
        s = stop()
        slow = s.force(0.17, s.v_max) - s.force(0.17, 0.0)
        fast = s.force(0.17, 5.0) - s.force(0.17, 0.0)
        self.assertAlmostEqual(slow, s.damping * s.v_max)
        self.assertAlmostEqual(fast, slow)

    def test_bad_settings_are_refused(self):
        for kw in ({"bump": 0.0}, {"bump": 0.25}, {"rate": 0.0}):
            with self.assertRaises(ValueError):
                stop(**kw)


class StepTest(unittest.TestCase):
    def test_acts_on_the_jounce_as_read_not_extrapolated(self):
        """The sweep's jounce jumps 8 cm in one step at a step's face (issue 077): the force is
        the one at the reading, its damper at v_max, not one extrapolated a step on."""
        s = stop()
        s.step([0.097] * 4, DT)
        f = s.step([0.179, 0.179, 0.097, 0.097], DT)
        self.assertAlmostEqual(f[0], s.force(0.179, s.v_max))
        self.assertLess(f[0], s.cap * s.rate * s.bump + 1e-9)
        self.assertEqual(f[2:], [0.0, 0.0])

    def test_a_wheel_without_a_reading_gets_nothing_and_reset_forgets(self):
        s = stop()
        self.assertEqual(s.step([None, 0.18, 0.1, 0.1], DT)[0], 0.0)
        s.reset()
        self.assertIsNone(s.prev)
        # no previous reading: no speed, the spring part only
        self.assertAlmostEqual(s.step([0.18] * 4, DT)[0], s.force(0.18, 0.0))


class ParamsTest(unittest.TestCase):
    def test_physx_travel_is_the_longer_of_the_two(self):
        p = VehicleParams()
        self.assertEqual(p.travels(), (0.2, 0.2))  # today's car: no stop
        self.assertEqual(VehicleParams(physx_travel=0.5).travels(), (0.5, 0.2))
        self.assertEqual(VehicleParams(physx_travel=0.1).travels(), (0.2, 0.2))
        self.assertEqual(p.wheel_query, "raycast")


if __name__ == "__main__":
    unittest.main()
