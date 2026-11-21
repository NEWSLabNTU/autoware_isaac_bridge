"""EgoRos.taken_command, the command the car runs this frame (scensim roadmap 092 K2.4):
the latest Control taken at the frame's start wins; the stamp and the count say which ran and
how many were superseded. No ROS here: the callbacks are driven by hand."""
__import__("_paths")  # sys.path, for direct runs

from types import SimpleNamespace

from aib.control import Command
from aib.ros import EgoRos


def control(acc, vel, steer, sec, nanosec):
    return SimpleNamespace(longitudinal=SimpleNamespace(acceleration=acc, velocity=vel),
                           lateral=SimpleNamespace(steering_tire_angle=steer),
                           stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def ros_with(arrivals_per_frame):
    """An EgoRos whose executor delivers `arrivals_per_frame[i]` Controls at frame i's take."""
    r = EgoRos.__new__(EgoRos)
    r.profile = "autoware"
    r.cmd = Command()
    r.cmd_stamp_ns, r.cmd_arrived, r._arrivals = 0, 0, 0
    r.wait_localized = lambda: None
    queue = []
    r.executor = SimpleNamespace(
        spin_once=lambda timeout_sec=0: queue and r._on_control(queue.pop(0)))
    return r, queue


def test_latest_wins_and_the_count_says_how_many_arrived():
    r, q = ros_with(None)
    assert r.taken_command() is None             # before the first Control
    q += [control(1.0, 3.0, 0.0, 1000, 0), control(0.5, 3.0, 0.1, 1000, 50_000_000)]
    r.take_commands()
    t = r.taken_command()
    assert t["stamp_ns"] == 1000_050_000_000 and t["acceleration"] == 0.5 and t["arrived"] == 2
    r.take_commands()                            # nothing new: the same command runs again
    t = r.taken_command()
    assert t["stamp_ns"] == 1000_050_000_000 and t["arrived"] == 0


def test_another_profile_has_none():
    r, _ = ros_with(None)
    r.profile = "ackermann"
    r.cmd.received = True
    assert r.taken_command() is None
