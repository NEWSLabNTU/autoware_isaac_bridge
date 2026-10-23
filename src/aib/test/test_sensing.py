"""Frames and sensor readings. Pure (numpy for the lidar)."""
import math
import unittest

import numpy as np

_paths = __import__("_paths")  # sys.path, for direct runs
from aib import frames
from aib.control import State
from aib.sensing import (POINT_STEP, USD_CAMERA_IN_LINK, Periodic, gnss_reading, imu_reading,
                         pack_cloud)


def close(a, b, tol=1e-9):
    return all(abs(x - y) < tol for x, y in zip(a, b))


class FramesTest(unittest.TestCase):
    def test_advance_is_the_inverse_of_the_vehicles_sampling(self):
        # aib.isaac.vehicle.Vehicle.sample: omega from qconj(q0) * q1 over dt, body frame
        q0 = frames.rpy_quat(0.02, -0.01, 1.2)
        dt, omega, vel = 1 / 60, (0.1, -0.3, 0.5), (3.0, -1.0, 0.2)
        (t, q1) = frames.advance(((1.0, 2.0, 0.5), q0), vel, omega, dt)
        self.assertTrue(close(t, (1.05, 2.0 - 1 / 60, 0.5 + 0.2 / 60)))
        dq = frames.qmul(frames.qconj(q0), q1)
        s = math.sqrt(sum(c * c for c in dq[1:]))
        back = tuple(c / s * 2 * math.atan2(s, dq[0]) / dt for c in dq[1:])
        self.assertTrue(close(back, omega, 1e-9))
        self.assertEqual(frames.advance(((0, 0, 0), q0), (0, 0, 0), (0, 0, 0), dt)[1], q0)

    def test_rpy_is_urdf_order(self):
        q = frames.rpy_quat(math.pi / 2, 0.0, math.pi / 2)
        self.assertTrue(close(frames.qrot(q, (1, 0, 0)), (0, 1, 0)))
        self.assertTrue(close(frames.qrot(q, (0, 0, 1)), (1, 0, 0)))

    def test_compose(self):
        a = ((1.0, 0.0, 0.0), frames.rpy_quat(0, 0, math.pi / 2))
        b = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        t, _ = frames.compose(a, b)
        self.assertTrue(close(t, (1.0, 1.0, 0.0)))

    def test_lat_lon_round_trip_near_the_origin(self):
        lat, lon = frames.lat_lon(190.0, -130.1)
        y = math.radians(lat) * frames.EARTH_RADIUS
        x = math.radians(lon) * frames.EARTH_RADIUS * math.cos(math.radians(lat))
        self.assertAlmostEqual(x, 190.0, places=6)
        self.assertAlmostEqual(y, -130.1, places=6)


class ImuTest(unittest.TestCase):
    def test_at_rest_it_reads_gravity_up(self):
        q, gyro, acc = imu_reading(State(), (1.0, 0.0, 0.0, 0.0))
        self.assertTrue(close(acc, (0.0, 0.0, 9.81)))
        self.assertTrue(close(gyro, (0.0, 0.0, 0.0)))

    def test_a_flipped_mount_flips_the_reading(self):
        # csb's (CARLA's) IMU mount: roll pi, yaw pi
        q_bi = frames.rpy_quat(math.pi, 0.0, math.pi)
        s = State(acceleration=(1.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.2))
        _, gyro, acc = imu_reading(s, q_bi)
        self.assertTrue(close(acc, (-1.0, 0.0, -9.81)))
        self.assertTrue(close(gyro, (0.0, 0.0, -0.2)))

    def test_braking_reads_forward_negative(self):
        _, _, acc = imu_reading(State(acceleration=(-3.0, 0.0, 0.0)), (1.0, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(acc[0], -3.0)


class GnssTest(unittest.TestCase):
    def test_the_antenna_offset_turns_with_the_car(self):
        yaw = math.pi / 2
        s = State(position=(10.0, 0.0, 0.0), quat=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
        lat, lon, alt = gnss_reading(s, (1.0, 0.0, 1.5))
        self.assertAlmostEqual(math.radians(lat) * frames.EARTH_RADIUS, 1.0, places=6)
        self.assertAlmostEqual(alt, 1.5)


class CloudTest(unittest.TestCase):
    def test_layout(self):
        xyz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        n, data = pack_cloud(xyz, [0.0, 0.5, 2.0], [-0.2, 0.0, 0.2], channels=3)
        self.assertEqual(n, 3)
        self.assertEqual(len(data), 3 * POINT_STEP)
        rec = np.frombuffer(data, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("intensity", "u1"), ("return_type", "u1"),
                                         ("channel", "<u2")])
        self.assertEqual(rec["y"].tolist(), [2.0, 5.0, 8.0])
        self.assertEqual(rec["intensity"].tolist(), [0, 127, 255])
        self.assertEqual(rec["channel"].tolist(), [0, 1, 2])
        self.assertEqual(set(rec["return_type"].tolist()), {1})

    def test_no_points(self):
        self.assertEqual(pack_cloud(np.zeros((0, 3)))[0], 0)

    def test_a_scalar_intensity_or_elevation_is_ignored(self):
        # What the RTX lidar's annotator gave on a ground plane: a 0-d array per field.
        xyz = np.ones((4, 3))
        n, data = pack_cloud(xyz, np.float32(0.5), np.float32(0.1))
        self.assertEqual(n, 4)
        self.assertEqual(len(data), 4 * 16)
        self.assertEqual(pack_cloud(np.zeros((0, 3)), np.float32(0.5), np.float32(0.1))[0], 0)


class MiscTest(unittest.TestCase):
    def test_the_usd_camera_looks_down_the_link_x(self):
        self.assertTrue(close(frames.qrot(USD_CAMERA_IN_LINK, (0, 0, -1)), (1, 0, 0)))
        self.assertTrue(close(frames.qrot(USD_CAMERA_IN_LINK, (0, 1, 0)), (0, 0, 1)))

    def test_periodic(self):
        p = Periodic(0.1)
        fired = [t for t in (0.0, 0.05, 0.1, 0.15, 0.2) if p.due(t)]
        self.assertEqual(fired, [0.0, 0.1, 0.2])
        self.assertTrue(all(Periodic(0).due(t) for t in (0.0, 0.0, 0.01)))


if __name__ == "__main__":
    unittest.main()
