# Issues

One file per defect: what was wrong, what it cost, how it was confirmed, the fix.

| # | Issue | Severity | Status |
|---|---|---|---|
| [001](001-friction-tables-removed-while-playing.md) | Per-wheel friction tables under the vehicle: PhysX refuses their removal at every despawn | low | fixed |
| [002](002-periodic-sensor-drops-to-a-whole-number-of-frames.md) | A sensor's period that is not a whole number of frames lost the remainder at every fire: a 30 Hz camera on 100 Hz frames published at 25 Hz (speed ladder camera_rate 24.7) | medium | fixed |
| [003](003-planar-lidar-silent-with-nothing-in-range.md) | A planar lidar (LaserScan) with nothing in range published no scan at all, so a stack steering by it never drove: an empty revolution now goes out with every beam +inf | high | fixed |
