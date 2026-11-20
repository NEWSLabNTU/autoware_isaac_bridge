# 003: A planar lidar with nothing in range publishes no scan at all

**Severity**: high for a LaserScan stack. A stack that steers by the scan gets no scan, so it never
drives. **Status**: fixed (this commit). **Found**: 2026-11-20, by 241 (scensim 091 W20 prework).

## What happens

The Tesla rig's top lidar ran as a UST-10LX class scanner (270 deg, 1081 beams, 10 m;
`isaac.laser_scan`). It stood at the centre of a 40 x 24 m walled hall, every wall more than 10 m
away. In 200 s of simulation, nothing went out on `/scan`. The runner logged:

```
[aib] ego 'tesla_model3': sensors not ready after 60 warm-up frames
```

f1tenth_system's chain and the stand-in driver waited for a scan that never came, and the car never
moved.

## Cause

Two places in `aib.isaac.sensors.Lidar` treat an empty revolution as no data:
- `ready()` wants at least one point;
- `_msg()` returns None when the annotator holds no points.

That holds for a 3D lidar on a road, which always sees the ground. A planar scanner at its mount
height often sees nothing. A real one still publishes every revolution, with every beam out of
range: urg_node's `+inf`, or the driver's own out-of-range value.

## Fix

For a lidar published as LaserScan (`isaac.laser_scan`), an empty revolution is a scan:
- `_msg()` builds it with every range `+inf`;
- `ready()` is true once the annotator answers, points or none.

With no points there are no firing times to gate whole revolutions by, so an empty scan goes out once
per frame. That is the rate the scan has anyway: one whole revolution per frame, at most.
