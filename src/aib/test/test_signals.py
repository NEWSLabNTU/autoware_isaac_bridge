__import__("_paths")  # sys.path, for direct runs

from aib import signals


def test_a_colour_is_a_solid_circle_at_full_confidence():
    assert signals.element("RED") == (signals.RED, signals.CIRCLE, signals.SOLID_ON, 1.0)
    assert signals.element("GREEN")[0] == signals.GREEN
    assert signals.element("AMBER")[0] == signals.AMBER


def test_a_dark_lamp_is_unknown():
    assert signals.element("OFF") == (0, 0, 0, 0.0)


def test_a_group_turns_green_one_light_at_a_time():
    c = signals.Cycle([["1", "2", "3"]], {"1": [10], "2": [20], "3": [30]}, 10.0, 3.0)
    assert c.colours(0.0) == {"1": "GREEN", "2": "RED", "3": "RED"}
    assert c.colours(10.5) == {"1": "AMBER", "2": "RED", "3": "RED"}
    assert c.colours(13.0) == {"1": "RED", "2": "GREEN", "3": "RED"}
    assert c.colours(36.5) == {"1": "RED", "2": "RED", "3": "AMBER"}
    assert c.colours(39.0) == c.colours(0.0)


def test_an_element_shows_its_most_restrictive_light():
    c = signals.Cycle([["1", "2"]], {"1": [10], "2": [10, 20]}, 10.0, 3.0)
    assert c.groups_at(0.0) == [(10, "RED"), (20, "RED")]
    assert c.groups_at(14.0) == [(10, "RED"), (20, "GREEN")]


def test_a_light_in_no_group_cycles_on_its_own():
    c = signals.Cycle([["1"]], {"1": [10], "9": [90]}, 10.0, 3.0)
    assert c.groups_at(0.0) == [(10, "GREEN"), (90, "GREEN")]


def test_a_map_dir_needs_its_groups_file(tmp_path):
    (tmp_path / "carla").mkdir()
    (tmp_path / "carla" / "traffic_lights.yaml").write_text(
        "signals:\n- opendrive_id: '1'\n  regulatory_element_ids: [10]\n"
        "- opendrive_id: '2'\n  regulatory_element_ids: []\n")
    try:
        signals.Cycle.from_map_dir(tmp_path)
        assert False, "no groups file must fail"
    except FileNotFoundError as e:
        assert "generate_signal_groups.py" in str(e)
    (tmp_path / "carla" / "signal_groups.yaml").write_text("groups:\n- lights: ['1', '2']\n")
    c = signals.Cycle.from_map_dir(tmp_path)
    assert c.groups_at(0.0) == [(10, "GREEN")]
    assert c.groups_at(13.0) == [(10, "RED")]
