"""The spec's lidar intensity on Isaac (084): table, ids, the id map, incidence, intensity."""
import os

import numpy as np
import yaml

from aib import intensity as im

HERE = os.path.dirname(os.path.abspath(__file__))
RIG = os.path.join(HERE, "..", "config", "tesla_model3.rig.yaml")


def rig_table():
    rig = yaml.safe_load(open(RIG))
    return rig["sensors"]["top"]["isaac"]["intensity"]


def test_the_rig_carries_the_specs_table():
    t = rig_table()
    assert t["default"] == 30
    assert t["reflectance"]["Car"] == 90 and t["reflectance"]["Roads"] == 5


def test_every_qcode_names_a_tag_the_spec_knows():
    from sim_vehicle_spec.spec import CARLA_SEMANTIC_TAGS
    assert set(im.QCODE_TAG.values()) <= set(CARLA_SEMANTIC_TAGS)


def test_a_class_reads_its_reflectance_and_the_rest_the_default():
    t = im.Table(rig_table())
    assert t.of_qcode("Q1420") == 90        # car
    assert t.of_qcode("Q34442") == 5        # road
    assert t.of_qcode("Q1392287") == 50     # road lines
    assert t.of_qcode("Q221488") == 50      # a walker
    assert t.of_qcode(None) == 30           # unlabelled
    assert t.of_qcode("Q999999") == 30      # a code we do not map


def test_object_ids_are_16_byte_little_endian():
    ids = [(1, 0), (2**64 - 1, 7), (5, 2**63)]
    raw = b"".join(lo.to_bytes(8, "little") + hi.to_bytes(8, "little") for lo, hi in ids)
    for buf in (np.frombuffer(raw, np.uint8), np.frombuffer(raw, np.uint32),
                np.frombuffer(raw, np.uint64)):
        got = im.object_ids(buf)
        assert [tuple(int(x) for x in r) for r in got] == ids


def test_the_stable_id_map_decodes_as_isaacs_own():
    labels = ["/Root/Road/A", "/World/npcs/car_1/body"]
    ids = [(11, 22), (2**40 + 3, 9)]
    head, body = b"", b""
    offset = 24 * len(ids)
    for (lo, hi), lab in zip(ids, labels):
        words = [lo & 0xFFFFFFFF, lo >> 32, hi & 0xFFFFFFFF, hi >> 32, len(lab), offset + len(body)]
        head += np.array(words, "<u4").tobytes()
        body += lab.encode()
    raw = head + body + len(ids).to_bytes(4, "little")
    assert im.stable_id_map(np.frombuffer(raw, np.uint8)) == dict(zip(ids, labels))


def test_incidence_is_the_rays_cosine_against_the_normal():
    xyz = [[10, 0, 0], [10, 0, 0], [3, 4, 0], [0, 0, 0]]
    n = [[-1, 0, 0], [0, 0, 1], [0, 0, 2], [1, 0, 0]]
    c = im.incidence_cos(xyz, n)
    assert np.allclose(c, [1.0, 0.0, 0.0, 0.0])
    c = im.incidence_cos([[1, 1, 0]], [[0, -3, 0]])
    assert np.isclose(c[0], np.sqrt(0.5))


def test_intensity_is_reflectance_times_cos_clamped():
    v = im.intensity([90, 90, 300, 5], [1.0, 0.5, 1.0, 0.1])
    assert v.dtype == np.uint8 and list(v) == [90, 45, 255, 0]


def test_per_point_reflectance_looks_each_id_up_once():
    ids = np.array([[1, 0], [2, 0], [1, 0], [1, 0]], dtype=np.uint64)
    calls = []

    def look(keys):
        calls.append(list(keys))
        return [{(1, 0): 90.0, (2, 0): 5.0}[k] for k in keys]
    r = im.per_point_reflectance(ids, look)
    assert list(r) == [90.0, 5.0, 90.0, 90.0] and calls == [[(1, 0), (2, 0)]]
    assert list(im.per_point_reflectance(ids, {(1, 0): 1.0, (2, 0): 2.0})) == [1, 2, 1, 1]


def test_id_reflectance_resolves_each_new_id_once_across_scans():
    calls = []
    table = {(1, 0): 90.0, (2, 0): 5.0, (3, 9): 50.0}

    def resolve(keys):
        calls.append(sorted(keys))
        return [table.get(k, 30.0) for k in keys], [k in table for k in keys]
    look = im.IdReflectance(resolve)
    a = np.array([[1, 0], [2, 0], [1, 0]], dtype=np.uint64)
    assert list(look(a)[0]) == [90.0, 5.0, 90.0]
    b = np.array([[2, 0], [3, 9], [1, 0], [3, 9]], dtype=np.uint64)
    assert list(look(b)[0]) == [5.0, 50.0, 90.0, 50.0]
    assert list(look(a)[0]) == [90.0, 5.0, 90.0]
    assert calls == [[(1, 0), (2, 0)], [(3, 9)]]


def test_an_id_not_placed_yet_is_asked_about_again():
    table = {}

    def resolve(keys):
        return [table.get(k, 30.0) for k in keys], [k in table for k in keys]
    look = im.IdReflectance(resolve)
    car = np.array([[7, 7], [7, 7], [1, 1]], dtype=np.uint64)
    table[(1, 1)] = 5.0
    vals, placed = look(car)
    assert list(vals) == [30.0, 30.0, 5.0] and list(placed) == [False, False, True]
    table[(7, 7)] = 90.0                         # the map caught up
    vals, placed = look(car)
    assert list(vals) == [90.0, 90.0, 5.0] and placed.all()


def test_a_point_takes_the_class_of_the_entity_box_it_falls_in():
    def at(x, y, yaw_deg=0.0):
        c, s_ = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        # USD row-vector matrix: rows are the frame's axes, the last its origin
        return [[c, s_, 0, 0], [-s_, c, 0, 0], [0, 0, 1, 0], [x, y, 0, 1]]
    car = (at(10, 0), (1.4, 0, 0.7), (4.8, 2.1, 1.5), "Q1420")
    walker = (at(10, 1.6, 90), (0, 0, 0.9), (0.5, 0.5, 1.8), "Q221488")  # beside the car
    pts = [[11.4, 0, 0.7],    # in the car
           [10, 1.6, 1.0],    # in the walker
           [10, 1.2, 0.9],    # in both (the margins): the walker's centre is the nearer
           [30, 0, 0.5]]      # in neither
    got = im.box_classes(pts, [car, walker])
    assert list(got) == [0, 1, 1, -1]
    assert list(im.box_classes([], [car])) == []
