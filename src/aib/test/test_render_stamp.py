"""scensim issue 196: a scan is stamped at the time of the pose its render took. The lidar's prim
is written before every substep, but the render on the frame's last substep takes the pose
written before its first, so a scan stamped at the frame's end was (n - 1) substeps late
(measured: 25.6 to 26.1, 33.6 to 34.4 and 42.2 to 42.7 ms at 40, 60 and 120 Hz)."""

import collections
import unittest
from dataclasses import dataclass

from aib.isaac.sensors import Lidar
from aib.sensing import RenderPose, earlier


@dataclass
class Time:
    sec: int = 0
    nanosec: int = 0


def frame(rp, n, dt):
    """A frame of `n` substeps of `dt` through RenderPose; its offset back to the render's pose."""
    clock = rp.clock
    for _ in range(n):
        clock += dt
        rp.substep(clock)
    return rp.frame_done()


class RenderPoseTest(unittest.TestCase):
    def test_the_render_pose_is_n_minus_one_substeps_before_the_frames_end(self):
        for hz, n in ((40, 2), (60, 3), (120, 6)):
            rp = RenderPose()
            for _ in range(3):  # every frame alike, not only the first
                self.assertAlmostEqual(frame(rp, n, 1.0 / hz), (n - 1) / hz, places=12)

    def test_a_frame_of_one_substep_is_stamped_at_its_end(self):
        self.assertEqual(frame(RenderPose(), 1, 0.05), 0.0)

    def test_a_frame_without_substeps_has_no_offset(self):
        self.assertEqual(RenderPose().frame_done(), 0.0)


class EarlierTest(unittest.TestCase):
    def test_a_stamp_moved_back_borrows_a_second(self):
        t = earlier(Time(sec=1002, nanosec=10_000_000), 1.0 / 30)
        self.assertEqual((t.sec, t.nanosec), (1001, 976_666_667))
        self.assertIsInstance(t, Time)

    def test_zero_keeps_the_stamp(self):
        t = Time(sec=5, nanosec=7)
        self.assertEqual(earlier(t, 0.0), t)


@dataclass
class Header:
    stamp: Time


@dataclass
class Msg:
    header: Header


class Sent:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


class Ros:
    def note_scan(self, stamp):
        pass


def bare_lidar(latency=2):
    """A Lidar without Isaac: the stamping state only, its message the stamp it was given."""
    lidar = Lidar.__new__(Lidar)
    lidar.pub, lidar.ros, lidar.pending = Sent(), Ros(), None
    lidar.latency = latency
    lidar.stamps = collections.deque(maxlen=latency + 1)
    lidar.deltas = collections.deque(maxlen=latency + 1)
    lidar.backs = collections.deque(maxlen=latency + 1)
    lidar.jitter, lidar.delta = None, 0.0
    lidar.clock, lidar.frame_dt, lidar.frame_start = 0.0, 0.0, 0.0
    lidar.render = RenderPose()
    lidar.t_refs = []
    lidar._msg = lambda stamp, to_sim: (lidar.t_refs.append((lidar.render_t_ref(), lidar.clock)),
                                        Msg(Header(stamp)))[1]
    return lidar


class LidarStampTest(unittest.TestCase):
    def run_frames(self, hz, frames=6, frame_s=0.05):
        n = round(frame_s * hz)
        dt = frame_s / n
        lidar, sim = bare_lidar(), 1000.0
        stamps = []
        for k in range(frames):
            for _ in range(n):
                lidar.clock += dt
                lidar.render.substep(lidar.clock)
            sim += frame_s
            stamp = Time(sec=int(sim), nanosec=round((sim - int(sim)) * 1e9))
            stamps.append(stamp)
            lidar.publish(None, sim, stamp)
        return lidar, stamps, n, dt

    def test_each_scan_goes_out_with_its_render_frames_pose_time(self):
        for hz in (40, 60, 120):
            lidar, stamps, n, dt = self.run_frames(hz)
            sent = lidar.pub.msgs
            # the data read at frame k was rendered at frame k - latency and goes out a frame on
            self.assertEqual(len(sent), len(stamps) - lidar.latency - 1)
            for i, msg in enumerate(sent):
                want = earlier(stamps[i], (n - 1) * dt)
                got = msg.header.stamp
                self.assertEqual((got.sec, got.nanosec), (want.sec, want.nanosec), (hz, i))

    def test_warm_up_frames_do_not_move_the_first_stamp(self):
        # the ego's warm-up steps frames (physics and a render) without publishing; each ends
        # with skip_frame, so the first published frame counts only its own substeps
        lidar, n, dt, sim = bare_lidar(latency=0), 3, 1.0 / 60, 1000.0
        for _ in range(4):
            for _ in range(n):
                lidar.clock += dt
                lidar.render.substep(lidar.clock)
            lidar.skip_frame()
        lidar.frame_start = lidar.clock
        for _ in range(2):
            for _ in range(n):
                lidar.clock += dt
                lidar.render.substep(lidar.clock)
            sim += n * dt
            stamp = Time(sec=int(sim), nanosec=round((sim - int(sim)) * 1e9))
            lidar.publish(None, sim, stamp)
        got = lidar.pub.msgs[0].header.stamp
        want = earlier(Time(sec=1000, nanosec=50_000_000), (n - 1) * dt)
        self.assertEqual((got.sec, got.nanosec), (want.sec, want.nanosec))

    def test_motion_distortion_starts_from_the_render_pose(self):
        lidar, stamps, n, dt = self.run_frames(60)
        frame_s = n * dt
        for t_ref, clock in lidar.t_refs:
            # `latency` frames back from the frame just stepped, then (n - 1) substeps more
            self.assertAlmostEqual(t_ref, clock - lidar.latency * frame_s - (n - 1) * dt, places=9)


if __name__ == "__main__":
    unittest.main()
