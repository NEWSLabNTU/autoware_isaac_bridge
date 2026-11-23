"""The friction aib writes to a PhysX tyre (scensim issue 108). Pure.

PhysX's tyre caps its force by its friction times a *filtered* load (VhTireFunctions.cpp,
computeTireLoad): restLoad x filter(load / restLoad). omni.physx applies PhysX's default filter,
(0, 0.2308) to (3, 3), and Isaac's USD schema does not expose it. Over a car, where the
restLoads are the sprung weight and the loads carry it, that is 0.2308 W + 0.9231 W = 1.154 W
at any load transfer while every wheel carries 0 to 3 rest loads: the car had 1.154 times the
spec's grip (measured: 1.145 with restLoad doubled against 1.14 predicted; 0.955 locked against
0.953). So aib writes the spec's mu divided by FILTER_GRIP, and the car's limit is the spec's.

What is left is per wheel: a wheel's grip against mu times its own load is
filtered_normalised_load(x) / x / FILTER_GRIP, 1.20 at half its rest load and 0.87 at three.
"""

LOAD_FILTER = ((0.0, 0.2308), (3.0, 3.0))  # omni.physx's (normalised, filtered) points


def filtered_normalised_load(x, points=LOAD_FILTER):
    """PhysX's computeFilteredNormalisedTireLoad: the two points' line, clamped outside them."""
    (x0, y0), (x1, y1) = points
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


FILTER_GRIP = filtered_normalised_load(1.0)  # the car's grip over its friction: 1.1539


def physx_friction(mu):
    """The friction value that gives a PhysX tyre the grip of `mu` (the spec's)."""
    return float(mu) / FILTER_GRIP


def wheel_share(x):
    """A wheel's grip against the spec's mu times its own load, at `x` rest loads, as written."""
    return filtered_normalised_load(x) / x / FILTER_GRIP


class WheelFriction:
    """Each wheel's friction in the spec's mu (`values`), written to its PhysX friction table as
    physx_friction(mu). `attach` takes each table's defaultFrictionValue attribute in wheel
    order (anything with Set(float)); `default(i)` is the value to create it with."""

    def __init__(self, bases):
        self.bases = [float(m) for m in bases]  # each wheel's axle's tire_friction
        self.values = list(self.bases)  # each wheel's friction now, the spec's mu
        self.attrs = []

    def default(self, i):
        return physx_friction(self.bases[i])

    def attach(self, attr):
        self.attrs.append(attr)
        return attr

    def set(self, i, mu):
        """Wheel `i`'s friction from the next step on; written only when it changes."""
        if abs(mu - self.values[i]) > 1e-6:
            self.attrs[i].Set(physx_friction(mu))
            self.values[i] = mu
