# 002: a sensor whose period is not a whole number of frames publishes slower than its rate

**Severity:** medium. A 30 Hz camera on 100 Hz frames published at 25 Hz. That failed the speed ladder's
`camera_rate` check (minimum 25 Hz) at 24.7 Hz. **Status:** fixed 2026-11-20.

## Defect

`aib.sensing.Periodic.due(t)` set the next deadline from the frame that fired: `next = t + period`.
A frame that fires is never exactly on the deadline unless the period is a whole number of frames, so
the time the frame was late is lost at every fire. With 10 ms frames and a 33.3 ms period, the camera
fired every 4th frame: 40 ms, or 25 Hz. A frame lost now and then took it below 25 Hz.

Seen on 2026-11-20 by swb 133, in the speed ladder's rung 10 on Isaac (town05_ring_lap, physics and
frame rate 100 Hz). The camera came out at 24.7 Hz in both arms: rendering at its own rate, and
rendering every frame. Both publish through the same `Periodic`, so both lost the same time.
Diagnosed by swb 241 (NAS `home/2026-11-20 speed ladder A-B try4 (133)/`).

Sensors whose period is a whole number of frames, such as a 10 Hz lidar on 100 Hz frames, were not
affected.

## Fix

The next deadline is counted from the last deadline: `next += period`. So 30 Hz on 10 ms frames fires
3, 3 and 4 frames apart, which averages 30 Hz. If the frame is already past the following deadline (the
first frame, a reset, a stall), the timer restarts from the frame, with no burst of catch-up fires. It
still fires at most once per frame.

`test_sensing.py::test_periodic_keeps_its_rate_off_the_frame_grid` checks this: 300 or 301 fires in 10 s,
with gaps of 30 and 40 ms only, and no burst after a stall.
