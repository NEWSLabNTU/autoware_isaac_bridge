# 001: per-wheel friction tables removed while playing

**Severity:** low (an error per wheel at every despawn and a leak in PhysX; no run failed).
**Status:** fixed 2026-10-23.

## Defect

`df94015` gave each wheel its own tire friction table (roadmap 047 S1 in
scenario-world-bridge), defined under the vehicle at `<vehicle>/tireFriction/<wheel>`. A host
despawns a vehicle by removing its prim, which takes the tables with it, and PhysX refuses
that while the simulation plays. Every ego despawn in the Isaac world server logged one error
per wheel:

```
[Error] [omni.physx.plugin] PhysX Vehicle: "/World/entities/e0/tireFriction/rearRight":
    removing tire friction tables while playing is not permitted.
```

## Impact

None visible: each later ego got new tables at its own path and drove (the world server's
S2 runs passed with 20 such errors in the log, from five despawns). PhysX keeps the tables it
refused to remove, so they leak for the server's life, and the errors bury real ones.

## Fix

The tables live outside the vehicle, at `/World/vehicleTireFriction/<vehicle path with "/"
as "_">/<wheel>`, and stay when it goes; a vehicle spawned again at the same path takes its
tables back (`Define` on the existing prim, the default value written afresh).

## Confirmed

Isaac world server on Town01, `just world-check sim:=isaac` twice (each spawns and despawns
an ego): 22/22 both times and no `removing tire friction tables` error, nor any other
omni.physx error, against four per despawn before. A scenario with the surface layer
(`town01_zones.xosc`) on the same server afterwards checks that the moved tables still take
the per-wheel friction.
