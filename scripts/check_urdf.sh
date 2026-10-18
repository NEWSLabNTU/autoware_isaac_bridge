#!/usr/bin/env bash
# Expand Autoware's vehicle URDF (tier4_vehicle_launch's vehicle.xacro) for aib's vehicle and
# sensor kit, as robot_state_publisher gets it, and check every sensor link and frame of the
# rig is in it. Needs Autoware and this workspace's install sourced.
set -euo pipefail
rig=${1:-$(dirname "$0")/../src/aib/config/tesla_model3.rig.yaml}
vm=${2:-aib_vehicle}
sm=${3:-aib_sensor_kit}
top=$(ros2 pkg prefix tier4_vehicle_launch)/share/tier4_vehicle_launch/urdf/vehicle.xacro
cfg=$(ros2 pkg prefix "${sm}_description")/share/${sm}_description/config
urdf=$(xacro "$top" vehicle_model:="$vm" sensor_model:="$sm" config_dir:="$cfg")
missing=0
links=$(python3 -c '
import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
print(" ".join(dict.fromkeys(["base_link", "sensor_kit_base_link"]
      + [v for s in d["sensors"].values() for v in (s["link"], s["frame"])])))' "$rig")
for link in $links; do
    if ! grep -q "<link name=\"$link\"" <<<"$urdf"; then
        echo "missing link $link" >&2
        missing=1
    fi
done
[ $missing = 0 ] && echo "URDF for $vm + $sm: $(grep -c '<link ' <<<"$urdf") links, all of the rig's links and frames present"
exit $missing
