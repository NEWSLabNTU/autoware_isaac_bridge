__import__("_paths")  # sys.path, for direct runs

import math

import pytest

from aib import interfaces as itf


def test_the_profiles():
    assert itf.check_profile("vesc") == "vesc"
    with pytest.raises(ValueError, match="not one of autoware, ackermann, vesc"):
        itf.check_profile("roboracer")


def test_the_driver_s_limits_default_to_f1tenth_s_and_clip_as_the_driver():
    lim = itf.DriverLimits()
    assert lim.clip("speed", 30000) == 23250.0 and lim.clip("speed", -30000) == -23250.0
    assert lim.clip("duty_cycle", 0.5) == 0.0  # f1tenth_system turns duty commands off
    assert lim.clip("servo", 0.95) == 0.85 and lim.clip("servo", 0.0) == 0.15
    own = itf.DriverLimits({"duty_cycle": [-2.0, 0.4], "servo": [0.9, 0.1]})
    assert own.range["duty_cycle"] == (-1.0, 0.4)  # bounded by the feasible range
    assert own.range["servo"] == (0.1, 0.9)  # min above max: swapped
    with pytest.raises(ValueError, match="unknown erpm"):
        itf.DriverLimits({"erpm": [0, 1]})


def test_the_last_motor_message_wins_and_ages_in_simulation_time():
    v = itf.VescInput()
    c = v.command(1.0)
    assert c.motor is None and c.servo is None and c.age == math.inf
    v.on_motor(1.0, "speed", 4614.0)
    v.on_motor(1.5, "brake", 300000.0)
    c = v.command(1.6)
    assert (c.motor, c.value) == ("brake", 200000.0) and abs(c.age - 0.1) < 1e-12
    assert v.on_servo(0.53) == 0.53 and v.on_servo(1.2) == 0.85  # the echo is the clipped one
    assert v.command(1.6).servo == 0.85
    v.on_motor(2.0, "duty_cycle", 0.3)
    assert v.command(2.0)[:2] == ("duty", 0.0)
    v.reset()
    assert v.command(3.0).motor is None


def test_the_last_drive():
    a = itf.AckermannInput()
    assert a.command(0.0).age == math.inf and a.command(0.0).speed == 0.0
    a.on_drive(2.0, 1.5, steering_angle=0.2)
    c = a.command(2.25)
    assert (c.speed, c.steering_angle, c.acceleration, c.age) == (1.5, 0.2, 0.0, 0.25)


def test_the_telemetry_as_vesc_state():
    s = itf.vesc_state({"erpm": -4614.0, "duty": 0.1, "current_motor": 3.2,
                        "voltage_input": 7.9, "tachometer": 120.4})
    assert s["speed"] == -4614.0 and s["duty_cycle"] == 0.1 and s["displacement"] == 120
    assert s["temp_fet"] == 0.0 and s["fault_code"] == 0
    assert itf.vesc_state(None)["speed"] == 0.0
    with pytest.raises(ValueError, match="'rpm' is not a VescState field"):
        itf.vesc_state({"rpm": 1})


def test_the_odometer_is_vesc_to_odom_s():
    o = itf.Odometer(0.25)
    assert o.update(0.0, 1.0, 0.0) == (0.0, 0.0, 0.0, 1.0, 0.0)  # the first sets the time only
    x, y, yaw, v, w = o.update(1.0, 1.0, 0.0)
    assert (x, y, yaw) == (1.0, 0.0, 0.0)
    x, y, yaw, v, w = o.update(1.5, 2.0, 0.1)
    assert abs(w - 2.0 * math.tan(0.1) / 0.25) < 1e-12
    assert abs(x - 2.0) < 1e-12 and abs(yaw - 0.5 * w) < 1e-12  # forward Euler: the old yaw
    assert o.update(2.0, 0.04, 0.3)[3] == 0.0  # under 0.05 m/s: standing
    with pytest.raises(ValueError):
        itf.Odometer(0.0)


def test_the_rolling_speed_is_the_driven_wheels():
    assert itf.rolling_speed((10.0, 10.0, 20.0, 20.0), 0.05, (False, False, True, True)) == 1.0
    assert itf.rolling_speed((10.0, 10.0, 10.0, 10.0), 0.05, (False,) * 4) == 0.5


def test_the_settings_default_to_autoware_and_refuse_by_name():
    s = itf.interface_settings(None)
    assert s["profile"] == "autoware" and s["frames"] == {"odom": "odom", "base": "base_link"}
    s = itf.interface_settings({"profile": "vesc", "namespace": "car1",
                                "limits": {"speed": [-1000, 1000]}})
    assert s["namespace"] == "car1" and s["publish_tf"] is True
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['topic'\]"):
        itf.interface_settings({"topic": "/drive"})
    with pytest.raises(ValueError, match="the vesc profile only"):
        itf.interface_settings({"profile": "ackermann", "limits": {"speed": [0, 1]}})
    with pytest.raises(ValueError, match="not one of"):
        itf.interface_settings({"profile": "vesc_v2"})


def test_an_ego_config_names_its_interface(tmp_path):
    from _paths import EGO

    from aib.rig import EgoConfig

    def cfg(extra):
        p = tmp_path / "ego.yaml"
        p.write_text(f"rig: {EGO.parent / 'tesla_model3.rig.yaml'}\n{extra}")
        return EgoConfig.load(p)

    assert cfg("").vehicle_interface == {}
    assert cfg("vehicle_interface: {profile: vesc}\n").vehicle_interface == {"profile": "vesc"}
    with pytest.raises(ValueError, match="unknown erpm"):
        cfg("vehicle_interface: {profile: vesc, limits: {erpm: [0, 1]}}\n")
    with pytest.raises(ValueError, match="drop the lockstep"):
        cfg("vehicle_interface: {profile: ackermann}\nlockstep: {timeout_s: 1}\n")


def test_sensors_core_ticks_at_the_driver_s_50_hz_not_the_frames():
    c = itf.TelemetryClock()
    t = 0.0
    out = []
    for frame in range(10):  # 20 Hz frames of three 60 Hz substeps
        for _ in range(3):
            t += 1 / 60
            c.note(t, {"erpm": round(t * 1000)})
        out += c.due(t)
    ticks = [round(tick, 6) for tick, _ in out]
    assert ticks[:3] == [round(1 / 60 * 3, 6), round(0.05 + 0.02, 6), round(0.05 + 0.04, 6)]
    assert len(out) == 1 + int((t - ticks[0]) / 0.02 + 1e-9)  # 50 Hz from the first
    for tick, tel in out:  # the last telemetry at or before each tick
        assert tel["erpm"] <= round(tick * 1000) and tel["erpm"] > round((tick - 1 / 60) * 1000) - 1
    c.reset()
    assert c.due(1.0) == [(1.0, None)]
    with pytest.raises(ValueError, match="core_rate_hz"):
        itf.interface_settings({"profile": "vesc", "core_rate_hz": 0})


def test_the_vesc_imu_is_in_the_driver_s_units():
    import math as m
    yaw = 0.3
    q = (m.cos(yaw / 2), 0.0, 0.0, m.sin(yaw / 2))
    ypr, acc, gyro, quat = itf.vesc_imu((q, (0.0, 0.0, 0.5), (1.0, 0.0, 9.80665)))
    assert abs(ypr[2] - m.degrees(yaw)) < 1e-9 and ypr[0] == 0.0
    assert acc == (1.0 / 9.80665, 0.0, 1.0) and abs(gyro[2] - m.degrees(0.5)) < 1e-12
    assert quat == q
