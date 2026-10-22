import _paths  # noqa: F401 - puts aib on the path

from aib import signals


def test_a_colour_is_a_solid_circle_at_full_confidence():
    assert signals.element("RED") == (signals.RED, signals.CIRCLE, signals.SOLID_ON, 1.0)
    assert signals.element("GREEN")[0] == signals.GREEN
    assert signals.element("AMBER")[0] == signals.AMBER


def test_a_dark_lamp_is_unknown():
    assert signals.element("OFF") == (0, 0, 0, 0.0)
