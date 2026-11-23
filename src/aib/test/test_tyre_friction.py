"""The friction aib writes to a PhysX tyre (scensim issue 108): the spec's mu over the load
filter's gain, on every path a friction reaches PhysX by. Pure."""
import unittest

_paths = __import__("_paths")  # sys.path, for direct runs
from aib.params import VehicleParams
from aib.tyre_friction import (FILTER_GRIP, LOAD_FILTER, WheelFriction, filtered_normalised_load,
                               physx_friction, wheel_share)

FRONTS = (True, True, False, False)  # aib.isaac.vehicle.WHEELS' order


class Attr:
    """A friction table's defaultFrictionValue attribute: what was written."""

    def __init__(self, value):
        self.writes = [value]

    def Set(self, value):
        self.writes.append(value)


def built(p):
    """What aib.isaac.vehicle.Vehicle builds: each wheel's base from its axle's tyre."""
    wf = WheelFriction([float(p.wheel_tire(front)["tire_friction"]) for front in FRONTS])
    for i in range(len(FRONTS)):
        wf.attach(Attr(wf.default(i)))
    return wf


class FilterTest(unittest.TestCase):
    def test_the_gain_is_the_filter_at_the_rest_load(self):
        self.assertEqual(LOAD_FILTER, ((0.0, 0.2308), (3.0, 3.0)))
        self.assertAlmostEqual(FILTER_GRIP, 0.2308 + (3.0 - 0.2308) / 3.0, places=12)
        self.assertAlmostEqual(FILTER_GRIP, 1.15387, places=5)

    def test_a_cars_grip_is_its_friction_at_any_load_transfer(self):
        # four wheels of one rest load each carry four rest loads, however they share them
        for loads in ((1, 1, 1, 1), (1.6, 1.6, 0.4, 0.4), (0.5, 1.5, 0.5, 1.5), (2.9, 0.1, 1, 0)):
            grip = sum(filtered_normalised_load(x) for x in loads) / FILTER_GRIP
            self.assertAlmostEqual(grip, 4.0)

    def test_what_is_left_is_per_wheel(self):
        for x, share in ((0.5, 1.20), (1.0, 1.0), (2.0, 0.90), (3.0, 0.867)):
            self.assertAlmostEqual(wheel_share(x), share, places=2)
        self.assertAlmostEqual(filtered_normalised_load(4.0), 3.0)  # clamped past three


class WritePathsTest(unittest.TestCase):
    """Each path a friction reaches a PhysX table by writes the spec's mu / FILTER_GRIP, and
    keeps the spec's mu as the wheel's value."""

    def test_the_tables_default_at_build(self):
        wf = built(VehicleParams(tire_friction=1.0))
        self.assertEqual([a.writes for a in wf.attrs], [[1.0 / FILTER_GRIP]] * 4)
        self.assertEqual(wf.values, [1.0] * 4)

    def test_a_set_tire_friction(self):
        # aib.steptest --set tire_friction=0.3: an override on the rig's vehicle
        p = VehicleParams.from_dict(VehicleParams().to_dict(), {"tire_friction": 0.3})
        wf = built(p)
        for a in wf.attrs:
            self.assertAlmostEqual(a.writes[0], 0.3 / FILTER_GRIP)
        self.assertEqual(wf.values, [0.3] * 4)

    def test_an_axles_override(self):
        p = VehicleParams(tire_friction=1.0, axle_tires={"rear": {"tire_friction": 0.8}})
        wf = built(p)
        want = [1.0, 1.0, 0.8, 0.8]
        for a, mu in zip(wf.attrs, want):
            self.assertAlmostEqual(a.writes[0], physx_friction(mu))
        self.assertEqual(wf.values, want)

    def test_a_hosts_set_wheel_friction(self):
        wf = built(VehicleParams(tire_friction=1.0))
        wf.set(2, 0.45)  # the surface layer, every substep
        wf.set(2, 0.45)  # unchanged: not written again
        wf.set(0, 1.0)  # the base: not written
        self.assertEqual(wf.attrs[2].writes, [1.0 / FILTER_GRIP, 0.45 / FILTER_GRIP])
        self.assertEqual(wf.attrs[0].writes, [1.0 / FILTER_GRIP])
        self.assertEqual(wf.values, [1.0, 1.0, 0.45, 1.0])


if __name__ == "__main__":
    unittest.main()
