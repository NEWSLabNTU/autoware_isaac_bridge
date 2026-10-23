"""Simulation time as every part of a run computes it (scenario-world-bridge roadmap 051):
sim_ns = nanos(seconds) + BASE_NS, the world protocol's sim_time. Pure.

BASE_NS keeps ROS time far from 0 (ROS's "no clock yet") and from the small values for which
`now() - duration` goes negative in tf2 and Autoware's monitors. Time is monotonic within an
episode and only there; a stack that sees it go back ends (autoware_sim_profile's
clock_guard) and comes back on the new episode.
"""
import math

BASE_NS = 1_000_000_000_000


def nanos(secs):
    """Seconds to integer ns, rounding half away from zero (Rust's f64::round; Python's round()
    rounds half to even, 1 ns apart on ~1.4 % of frames)."""
    return int(math.copysign(math.floor(abs(secs) * 1e9 + 0.5), secs))


def sim_ns(secs):
    return nanos(secs) + BASE_NS
