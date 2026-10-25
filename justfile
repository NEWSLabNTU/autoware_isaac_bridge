# Autoware-Isaac Bridge
set dotenv-load

project := justfile_directory()
# Where Autoware's own setup.bash lives (the NEWSLabNTU 1.5.0 Debian installs under
# /opt/autoware/1.5.0). Override with AUTOWARE_SETUP when yours is elsewhere.
autoware_setup := env_var_or_default('AUTOWARE_SETUP', '/opt/autoware/1.5.0/setup.bash')
# Isaac Sim 6.1's pip install (its venv, its bundled ROS 2 Humble). Environment, not a setting.
isaac_root := env_var_or_default('ISAAC_ROOT', '/mnt/disk1/isaac-sim')

# List available recipes
default:
    @just --list

# One-time: Autoware's message packages rebuilt for Isaac's Python 3.12, and the setuptools check
setup: _check-setuptools
    ISAAC_ROOT={{isaac_root}} src/aib/scripts/build_msgs.sh

# Fail fast if a pip setuptools shadows the system one (59.6.0): colcon --symlink-install
# runs `setup.py develop --editable`, which setuptools 80 removed.
_check-setuptools:
    #!/usr/bin/env bash
    set -e
    path=$(python3 -c 'import setuptools; print(setuptools.__file__)')
    case "$path" in
        /usr/lib/python3/dist-packages/*) ;;
        *)
            echo "ERROR: setuptools resolves to $path" >&2
            echo "  colcon --symlink-install will fail with 'option --editable not recognized'." >&2
            echo "  Fix: pip uninstall -y setuptools" >&2
            exit 1
            ;;
    esac

# Build all packages (colcon, --symlink-install) against Autoware
build: _check-setuptools check-generated
    #!/usr/bin/env bash
    set -e
    source "{{autoware_setup}}"
    # deps/ (submodules): the Autoware profile and the car's packages aib_launch includes
    colcon build --base-paths src deps/autoware_sim_vehicle deps/sim_vehicle_spec --symlink-install

# Remove build artifacts
clean:
    rm -rf build install log

# Write the Isaac rig from the vehicle spec(s) in deps/sim_vehicle_spec (a submodule); Autoware's
# packages for the car are autoware_sim_vehicle's
generate:
    PYTHONPATH={{project}}/src/aib python3 -m aib.gen

# Fail when a generated file is stale (the spec changed, or a generated file was edited)
check-generated:
    PYTHONPATH={{project}}/src/aib python3 -m aib.gen --check

# Unit tests (no Isaac, no ROS), the generated-file check, and the URDF check when built
test: check-generated
    #!/usr/bin/env bash
    set -e
    python3 -m pytest -q -p no:cacheprovider src/aib/test
    if [ -f install/setup.bash ]; then
        source "{{autoware_setup}}"
        source install/setup.bash
        scripts/check_urdf.sh
    else
        echo "(not built: skipping the URDF check; run just build)"
    fi

# The pure unit tests under Isaac's Python (3.12), as the Isaac side imports them
test-isaac:
    #!/usr/bin/env bash
    set -e
    export ISAAC_ROOT={{isaac_root}}
    source "{{project}}/src/aib/scripts/isaac_env.sh"
    cd "{{project}}/src/aib/test" && python -m unittest -q

# Lint the Python (pyflakes) and run the tests
check: test
    python3 -m pyflakes src/aib

# Run one ego in an Isaac world, no swb/SSv2 (Isaac's Python). Settings as name:=value:
#   ego:=<yaml>      ego config (default src/aib/config/ego.yaml)
#   world:=<usd>     the stage to drive in, with its colliders; empty: a flat ground plane
#   pose:=x,y,z,yaw  base_link's start in the world (default 0,0,0.05,0)
#   realtime:=true   pace frames to the wall clock
#   duration:=<s>    stop after this much sim time
#   signals:=<map dir>   cycle that map's traffic lights as CARLA does and publish them to
#                    Autoware (needs <map dir>/carla/signal_groups.yaml); green_s:=, amber_s:=
#                    (default 10, 3); none: Autoware stops at every signalised stop line
# A window opens on $DISPLAY when set (ISAAC_HEADLESS=1 forces headless); DDS as the ego
# stack's (CYCLONEDDS_URI / RMW_IMPLEMENTATION pass through).
[positional-arguments]
isaac-ego *args:
    #!/usr/bin/env bash
    set -e
    ego="{{project}}/src/aib/config/ego.yaml"; world=""; pose="0,0,0.05,0"; extra=()
    for a in "$@"; do
        case "$a" in
            ego:=*) ego="${a#ego:=}" ;;
            world:=*) world="${a#world:=}" ;;
            pose:=*) pose="${a#pose:=}" ;;
            realtime:=true) extra+=(--realtime) ;;
            realtime:=*) ;;
            duration:=*) extra+=(--duration "${a#duration:=}") ;;
            signals:=?*) extra+=(--signals "${a#signals:=}") ;;
            signals:=) ;;
            green_s:=*) extra+=(--green-s "${a#green_s:=}") ;;
            amber_s:=*) extra+=(--amber-s "${a#amber_s:=}") ;;
            *) echo "unknown argument $a" >&2; exit 2 ;;
        esac
    done
    display=""
    if [ -n "${DISPLAY:-}" ] && [ "${ISAAC_HEADLESS:-0}" != 1 ]; then display="$DISPLAY"; fi
    export ISAAC_ROOT={{isaac_root}}
    source "{{project}}/src/aib/scripts/isaac_env.sh"
    export PYTHONPATH="{{project}}/src/aib:$PYTHONPATH"
    exec python -m aib.runner --ego "$ego" --world "$world" --pose="$pose" \
        --display "$display" "${extra[@]}"

# vehicle_check (autoware_sim_vehicle's autoware_sim_check) on Isaac, without SSv2 or Autoware:
# the ego on a flat ground plane (`just isaac-ego realtime:=true`, headless), then the contract
# (topics, types, rates, frames, stamps, /robot_description) and the dynamics run (acceleration
# and brake steps against 0.35 m/s^2, a steering sweep against the bicycle model), on the ego
# config's ROS domain. Writes <out>/report.json; exits 0 only if every check passed.
#
# Usage: just vehicle-check [ego:=<yaml>] [out:=<dir>]
[positional-arguments]
vehicle-check *args:
    #!/usr/bin/env bash
    set -e
    ego="{{project}}/src/aib/config/ego.yaml"; out="{{project}}/play_log/vehicle_check/$(date +%F_%H-%M-%S)"
    for a in "$@"; do
        case "$a" in
            ego:=?*) ego="${a#*:=}" ;;
            out:=?*) out="${a#*:=}" ;;
            *) echo "unknown argument $a (ego:=, out:=)" >&2; exit 2 ;;
        esac
    done
    mkdir -p "$out"
    export ROS_DOMAIN_ID=$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("domain", 1))' "$ego")
    groups=()
    stop() { for g in "${groups[@]}"; do kill -TERM -"$g" 2>/dev/null || true; done
             sleep 20; for g in "${groups[@]}"; do kill -KILL -"$g" 2>/dev/null || true; done; }
    trap stop EXIT
    setsid env -u DISPLAY ISAAC_HEADLESS=1 just --justfile "{{justfile()}}" isaac-ego ego:="$ego" \
        realtime:=true > "$out/runner.log" 2>&1 < /dev/null &
    runner=$!; groups+=($runner)
    # Autoware's setup.bash replaces CYCLONEDDS_URI when loopback multicast is on; keep the
    # caller's, the Isaac ego's, or the check never discovers it (scenario-world-bridge issue 008).
    dds_uri="${CYCLONEDDS_URI:-}"
    source "{{autoware_setup}}"; source "{{project}}/install/setup.bash"
    if [ -n "$dds_uri" ]; then export CYCLONEDDS_URI="$dds_uri"; fi
    echo "[vehicle-check] waiting for the Isaac ego (domain $ROS_DOMAIN_ID)"
    until timeout 20 ros2 topic echo --no-daemon --once /vehicle/status/velocity_status > /dev/null 2>&1; do
        kill -0 $runner 2>/dev/null || { echo "[vehicle-check] the runner exited; see $out/runner.log" >&2; exit 1; }
        sleep 5
    done
    model=$(python3 -c 'import os, sys, yaml
    ego = sys.argv[1]
    rig = os.path.join(os.path.dirname(ego), yaml.safe_load(open(ego))["rig"])
    print(yaml.safe_load(open(rig)).get("name", "tesla_model3"))' "$ego")
    play_launch launch --disable-web-ui --log-dir "$out/check_log" \
        autoware_sim_check vehicle_check.launch.xml simulator:=isaac vehicle_model:="$model" \
        out:="$out/report.json" > "$out/check.log" 2>&1 < /dev/null || true
    [ -f "$out/report.json" ] || { echo "[vehicle-check] no report; see $out/check.log" >&2; exit 1; }
    python3 - "$out/report.json" <<'PY'
    import json, sys
    r = json.load(open(sys.argv[1]))
    bad = [t for t, p in r["contract"].items() if p] + [s["step"] for s in r["dynamics"] if not s["pass"]]
    for s in r["dynamics"]:
        print(f'  {s["step"]:12} {"pass" if s["pass"] else "FAIL"}  ' + ", ".join(f"{k} {v}" for k, v in s.items() if k not in ("step", "pass")))
    print(f'[vehicle-check] {r["vehicle_model"]} on {r["simulator"]}: ' + ("PASS" if not bad else "FAIL: " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
    PY

# Open-loop step tests of the PhysX vehicle and its onboard computers in Isaac, no ROS (aib.steptest):
#   out:=<dir>          where the CSVs and summary.json go (default /mnt/disk1/aib-ev/steptest)
#   powertrain:=<model> override the calibration's (ev, ideal)
#   only:=a,b           some of: launch brake_100 regen_60 steps accel_brake hard_brake limit_brake
#   set:=k=v            a VehicleParams or Calibration override (JSON values; repeat for more);
#                       label:=<name> the
#                       run's subdirectory
[positional-arguments]
isaac-steptest *args:
    #!/usr/bin/env bash
    set -e
    out=/mnt/disk1/aib-ev/steptest; extra=()
    for a in "$@"; do
        case "$a" in
            out:=*) out="${a#out:=}" ;;
            powertrain:=*) extra+=(--powertrain "${a#powertrain:=}") ;;
            only:=*) extra+=(--only "${a#only:=}") ;;
            label:=*) extra+=(--label "${a#label:=}") ;;
            world:=*) extra+=(--world "${a#world:=}") ;;
            set:=*) extra+=(--set "${a#set:=}") ;;
            *) echo "unknown argument $a" >&2; exit 2 ;;
        esac
    done
    export ISAAC_ROOT={{isaac_root}}
    source "{{project}}/src/aib/scripts/isaac_env.sh"
    export PYTHONPATH="{{project}}/src/aib:$PYTHONPATH"
    exec python -m aib.steptest --out "$out" "${extra[@]}"

# Autoware for the Isaac ego (aib_launch isaac_simulator.launch.xml) on ROS_DOMAIN_ID
# (default 1, the ego config's). map_path:=<dir> is required; other name:=value pass through.
[positional-arguments]
autoware *args:
    #!/usr/bin/env bash
    set -e
    # Autoware's setup.bash sets its own CYCLONEDDS_URI; keep the caller's, which must be the
    # Isaac side's (`just isaac-ego`), or the two never discover each other.
    dds_uri="${CYCLONEDDS_URI:-}"
    source "{{autoware_setup}}"
    if [ -n "$dds_uri" ]; then export CYCLONEDDS_URI="$dds_uri"; fi
    source "{{project}}/install/setup.bash"
    # Arguments roadmap 051 renamed are refused by name (ROS launch would drop them).
    "{{project}}/install/autoware_sim_profile/lib/autoware_sim_profile/check_launch_args" "$@" || exit 2
    export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1}
    exec play_launch launch --web-addr 0.0.0.0:${AIB_WEB_PORT:-8086} --log-dir play_log/ego \
        aib_launch isaac_simulator.launch.xml "$@"
