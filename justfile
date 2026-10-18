# Autoware-Isaac Bridge
set dotenv-load

project := justfile_directory()
# Where Autoware's own setup.bash lives (the NEWSLabNTU 1.5.0 Debian installs under
# /opt/autoware/1.5.0). Override with AUTOWARE_SETUP when yours is elsewhere.
autoware_setup := env_var_or_default('AUTOWARE_SETUP', '/opt/autoware/1.5.0/setup.bash')
# Isaac Sim 6.1's pip install (its venv, its bundled ROS 2 Humble). Environment, not a setting.
isaac_root := env_var_or_default('ISAAC_ROOT', '/mnt/disk1/isaac-sim')
# acb's install, for its ROS packages (acb_launch, acb_pilot) when not in this workspace.
acb_setup := env_var_or_default('ACB_SETUP', '')
specs := project + '/vehicles/*.yaml'

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
    colcon build --base-paths src --symlink-install

# Remove build artifacts
clean:
    rm -rf build install log

# Write Autoware's vehicle and sensor kit files and the Isaac rig from the vehicle spec(s)
generate:
    PYTHONPATH={{project}}/src/aib python3 -m aib.gen {{specs}}

# Fail when a generated file is stale (the spec changed, or a generated file was edited)
check-generated:
    PYTHONPATH={{project}}/src/aib python3 -m aib.gen {{specs}} --check

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

# Run one ego in an Isaac world, no csb/SSv2 (Isaac's Python). Settings as name:=value:
#   ego:=<yaml>      ego config (default src/aib/config/ego.yaml)
#   world:=<usd>     the stage to drive in, with its colliders; empty: a flat ground plane
#   pose:=x,y,z,yaw  base_link's start in the world (default 0,0,0.05,0)
#   realtime:=true   pace frames to the wall clock
#   duration:=<s>    stop after this much sim time
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

# Open-loop step tests of the PhysX vehicle and powertrain in Isaac, no ROS (aib.steptest):
#   out:=<dir>          where the CSVs and summary.json go (default /mnt/disk1/aib-ev/steptest)
#   powertrain:=<model> override the rig's (ev, ideal)
#   only:=a,b           some of: launch brake_100 regen_60 steps accel_brake hard_brake limit_brake
#   set:=k=v            a VehicleParams override (JSON values; repeat for more); label:=<name> the
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
    if [ -n "{{acb_setup}}" ]; then source "{{acb_setup}}"; fi
    source "{{project}}/install/setup.bash"
    export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1}
    exec play_launch launch --web-addr 0.0.0.0:${AIB_WEB_PORT:-8086} --log-dir play_log/ego \
        aib_launch isaac_simulator.launch.xml "$@"
