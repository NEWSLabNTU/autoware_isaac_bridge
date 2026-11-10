"""Frames and sensor readings. Pure (numpy for the lidar)."""
import math
import unittest

import numpy as np

_paths = __import__("_paths")  # sys.path, for direct runs
from aib import frames
from aib.control import State
from aib.sensing import (POINT_STEP, USD_CAMERA_IN_LINK, Periodic, ScanGate, Ticks, gnss_reading,
                         ground_truth, imu_reading, motion_distort, pack_cloud, point_times,
                         pose_covariance, sample_index)


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
        # swb's (CARLA's) IMU mount: roll pi, yaw pi
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

    def test_mgrs_square_corner(self):
        # 51RUH: zone 51 (column set STUVWXYZ, U = 300 km), odd zone (H = 700 km), band R
        # (24-32 N) puts the square in the second 2000 km cycle
        self.assertEqual(frames.mgrs_origin("51RUH"), (51, True, 300_000.0, 2_700_000.0))

    def test_mgrs_projector_matches_the_maps_own_lat_lon(self):
        # node 1 of the NTU r02 Lanelet2 map, as its MGRS 51RUH tooling wrote it; the world is
        # the map less map_origin (scenario-world-bridge roadmap 059)
        p = frames.Projector((52_500.0, 67_700.0, 0.0), "51RUH")
        lat, lon, alt = p.lat_lon_alt(52217.4936 - 52_500.0, 67651.6452 - 67_700.0, 11.8783)
        north = (lat - 25.01720437293) * 111_320.0
        east = (lon - 121.53539682005) * 111_320.0 * math.cos(math.radians(lat))
        self.assertLess(math.hypot(north, east), 0.002)
        self.assertAlmostEqual(alt, 11.8783)

    def test_transverse_mercator_about_an_origin_matches_the_maps_own_lat_lon(self):
        # node 10300004601 of NTU COSS's Lanelet2 map (AutoSDV's car A twin): Autoware's
        # TransverseMercator (GeographicLib exact, k0 0.9996) about map_projector_info's origin
        # puts its lat/lon at local_x/local_y less TWD97 of that origin
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "map_projector_info.yaml"), "w") as f:
                f.write("projector_type: TransverseMercator\nvertical_datum: WGS84\n"
                        "map_origin: {latitude: 25.0201, longitude: 121.5423, altitude: 25.0}\n")
            p = frames.Projector.from_map(d)
        x, y = 304774.8604 - 304731.3793, 2768128.0634 - 2768113.4495
        lat, lon, alt = p.lat_lon_alt(x, y, 8.9764)
        north = (lat - 25.0202319785) * 111_320.0
        east = (lon - 121.54273096174) * 111_320.0 * math.cos(math.radians(lat))
        self.assertLess(math.hypot(north, east), 0.002)
        self.assertAlmostEqual(alt, 8.9764)

    def test_the_reading_takes_the_projector(self):
        p = frames.Projector((52_500.0, 67_700.0, 0.0), "51RUH")
        s = State(position=(-282.5064, -48.3548, 11.8783), quat=(1.0, 0.0, 0.0, 0.0))
        self.assertEqual(gnss_reading(s, (0.0, 0.0, 0.0), p),
                         p.lat_lon_alt(-282.5064, -48.3548, 11.8783))

    def test_no_projector_file_is_the_old_frame(self):
        p = frames.Projector.from_map("/nonexistent")
        self.assertEqual(p.lat_lon_alt(10.0, 20.0, 1.0)[:2], frames.lat_lon(10.0, 20.0))

    def test_transverse_mercator_at_zero_is_the_old_frame(self):
        # what download_maps.sh writes for CARLA's towns (swb 063: aib refused it, so no
        # Isaac run on such a map started)
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "map_projector_info.yaml")
            with open(f, "w") as fh:
                fh.write("projector_type: TransverseMercator\nvertical_datum: WGS84\n"
                         "map_origin: {latitude: 0.0, longitude: 0.0, altitude: 0.0}\n")
            p = frames.Projector.from_map(d)
            self.assertEqual(p.lat_lon_alt(10.0, 20.0, 1.0)[:2], frames.lat_lon(10.0, 20.0))
            # about another origin it was refused; it is projected now (the COSS test above)
            with open(f, "w") as fh:
                fh.write("projector_type: TransverseMercator\n"
                         "map_origin: {latitude: 25.0, longitude: 121.5}\n")
            lat, lon, _ = frames.Projector.from_map(d).lat_lon_alt(0.0, 0.0, 0.0)
            self.assertAlmostEqual(lat, 25.0, places=7)   # the series: ~0.3 mm
            self.assertAlmostEqual(lon, 121.5, places=7)


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

    def test_nebula_layout(self):
        # A VLP-32C's lasers are not evenly spaced: each elevation its own channel (rings),
        # and Nebula's PointXYZIRCAEDT for the cart's preprocessor.
        xyz = np.array([[0, 2, 0], [-3, 0, 4], [1, -1, 0], [2, 0, 0]], dtype=np.float32)
        elev = [-0.436, -0.017, -0.017, 0.262]
        n, data = pack_cloud(xyz, None, elev, fmt="xyzircaedt", rings=True)
        self.assertEqual((n, len(data)), (4, 4 * 32))
        rec = np.frombuffer(data, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("intensity", "u1"), ("return_type", "u1"),
                                         ("channel", "<u2"), ("azimuth", "<f4"),
                                         ("elevation", "<f4"), ("distance", "<f4"),
                                         ("time_stamp", "<u4")])
        self.assertEqual(rec["channel"].tolist(), [0, 1, 1, 2])
        self.assertTrue(np.allclose(rec["azimuth"], [np.pi / 2, np.pi, 1.75 * np.pi, 0.0]))
        self.assertTrue(np.allclose(rec["elevation"], [0.0, np.arctan2(4, 3), 0.0, 0.0]))
        self.assertTrue(np.allclose(rec["distance"], [2.0, 5.0, np.sqrt(2), 2.0]))
        self.assertEqual(rec["time_stamp"].tolist(), [0, 0, 0, 0])


class MiscTest(unittest.TestCase):
    def test_the_usd_camera_looks_down_the_link_x(self):
        self.assertTrue(close(frames.qrot(USD_CAMERA_IN_LINK, (0, 0, -1)), (1, 0, 0)))
        self.assertTrue(close(frames.qrot(USD_CAMERA_IN_LINK, (0, 1, 0)), (0, 0, 1)))

    def test_periodic(self):
        p = Periodic(0.1)
        fired = [t for t in (0.0, 0.05, 0.1, 0.15, 0.2) if p.due(t)]
        self.assertEqual(fired, [0.0, 0.1, 0.2])
        self.assertTrue(all(Periodic(0).due(t) for t in (0.0, 0.0, 0.01)))




class GroundTruthTest(unittest.TestCase):
    """base_link in the map frame for the recordings and the drive on ground truth (swb 063 e)."""

    def test_the_map_is_the_world_plus_its_origin(self):
        st = State()
        st.position = (10.0, -5.0, 0.3)
        st.quat = frames.rpy_quat(0.0, 0.0, 1.0)
        st.velocity = (20.0, 0.1, 0.0)
        st.angular_velocity = (0.0, 0.0, 0.2)
        p, q, v, w = ground_truth(st, (100.0, 200.0, 1.0))
        self.assertTrue(close(p, (110.0, 195.0, 1.3)))
        self.assertTrue(close(q, st.quat))
        self.assertTrue(close(v, (20.0, 0.1, 0.0)))  # body frame, as the velocity report
        self.assertTrue(close(w, (0.0, 0.0, 0.2)))
        self.assertTrue(close(ground_truth(st)[0], st.position))  # no origin: the world

    def test_the_covariance_is_diagonal(self):
        c = pose_covariance((0.01, 0.0003))
        self.assertEqual(len(c), 36)
        self.assertEqual([c[i * 7] for i in range(6)], [0.01] * 3 + [0.0003] * 3)
        self.assertAlmostEqual(sum(c), 0.0309)

if __name__ == "__main__":
    unittest.main()


class ScanTest(unittest.TestCase):
    """A rotating lidar as its driver publishes it: one message a revolution, each point's
    firing time after the header's stamp, and the motion the sensor made while it spun."""

    def test_a_scan_goes_out_once_though_frames_render_faster_than_it_turns(self):
        # a 10 Hz lidar under 20 Hz frames: the annotator hands the same revolution twice
        gate = ScanGate()
        self.assertTrue(gate.new(np.array([0, 50, 99])))
        self.assertFalse(gate.new(np.array([0, 50, 99])))
        self.assertTrue(gate.new(np.array([100, 150, 199])))

    def test_point_times_are_nanoseconds_after_the_scan_start(self):
        t0, off = point_times(np.array([100_000_000, 100_000_500, 199_994_720], dtype=np.uint64))
        self.assertEqual(t0, 100_000_000)
        self.assertEqual(off.dtype, np.uint32)
        self.assertEqual(list(off), [0, 500, 99_994_720])

    def test_the_cloud_carries_the_point_times(self):
        n, data = pack_cloud([(1, 0, 0), (0, 1, 0)], fmt="xyzircaedt",
                             time_stamp=np.array([0, 25_000_000], dtype=np.uint32))
        rec = np.frombuffer(data, dtype=np.dtype({"names": ["time_stamp"], "formats": ["<u4"],
                                                  "offsets": [28], "itemsize": 32}))
        self.assertEqual(list(rec["time_stamp"]), [0, 25_000_000])

    def _history(self, v=(0.0, 0.0, 0.0), yaw_rate=0.0):
        t = np.linspace(-0.2, 0.3, 81)
        p = np.outer(t, v)
        q = np.stack([np.cos(yaw_rate * t / 2), 0 * t, 0 * t, np.sin(yaw_rate * t / 2)], axis=1)
        return t, p, q

    def test_a_still_sensor_sees_the_scene_as_rendered(self):
        xyz = np.array([[10.0, 0, 0], [0, 5.0, 1.0]])
        out = motion_distort(xyz, np.array([0.0, 0.08]), 0.0, *self._history())
        np.testing.assert_allclose(out, xyz, atol=1e-6)

    def test_a_sensor_moving_forward_sees_a_wall_closer_the_later_it_fires(self):
        # rendered at t=0 from the sensor's pose then; fired at 0.1 s after 0.2 m of travel
        xyz = np.array([[10.0, 0, 0], [10.0, 0, 0]])
        out = motion_distort(xyz, np.array([0.0, 0.1]), 0.0, *self._history(v=(2.0, 0, 0)))
        np.testing.assert_allclose(out, [[10.0, 0, 0], [9.8, 0, 0]], atol=2e-3)

    def test_a_turning_sensor_sees_the_scene_turn_the_other_way(self):
        xyz = np.array([[10.0, 0, 0]])
        out = motion_distort(xyz, np.array([0.1]), 0.0, *self._history(yaw_rate=0.5))
        a = -0.05  # the sensor yawed +0.05 rad, so the point lies 0.05 rad clockwise
        np.testing.assert_allclose(out, [[10 * math.cos(a), 10 * math.sin(a), 0]], atol=2e-2)


class ImuRateTest(unittest.TestCase):
    """An IMU faster than the frame rate: every tick of its own clock inside the frame."""

    def test_ticks_run_at_the_imus_rate_across_frames(self):
        ticks = Ticks(0.01)
        self.assertEqual(ticks.due(0.05), [0.05])
        np.testing.assert_allclose(ticks.due(0.10), [0.06, 0.07, 0.08, 0.09, 0.10])
        np.testing.assert_allclose(ticks.due(0.15), [0.11, 0.12, 0.13, 0.14, 0.15])

    def test_a_slower_imu_skips_frames(self):
        ticks = Ticks(0.1)
        self.assertEqual(ticks.due(0.05), [0.05])
        self.assertEqual(ticks.due(0.10), [])
        np.testing.assert_allclose(ticks.due(0.15), [0.15])

    def test_a_tick_reads_the_last_state_at_or_before_it(self):
        t = [0.0, 0.00625, 0.0125, 0.01875]
        self.assertEqual(sample_index(t, 0.01), 1)
        self.assertEqual(sample_index(t, 0.0125), 2)
        self.assertEqual(sample_index(t, 0.5), 3)
