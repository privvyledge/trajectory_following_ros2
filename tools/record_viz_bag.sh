#!/usr/bin/env bash
# Record everything the visualizer draws, so a run can be replayed, re-viewed and
# analysed offline without the car.
#
# This is the durable artifact: a .rrd freezes one rendering of one run, while a
# bag can be replayed through the visualizer (any backend), re-analysed with new
# scripts, or fed to a fresh controller for offline tuning.
#
# Usage:
#   tools/record_viz_bag.sh <output_dir> [namespace] [extra_topic ...]
#     tools/record_viz_bag.sh data/bags/run_$(date +%H%M%S) gosling1
#     tools/record_viz_bag.sh /tmp/simrun            # no namespace
#
# Replay with tools/replay_viz_bag.sh.
set -euo pipefail

OUT=${1:?usage: record_viz_bag.sh <output_dir> [namespace] [extra_topic ...]}
NS=${2:-}
shift $(( $# > 1 ? 2 : 1 ))
EXTRA=("$@")

prefix=''
[[ -n "$NS" ]] && prefix="/${NS#/}"

# The visualizer's subscriptions (viz/visualizer_node.py::_setup_subscriptions)
# plus the controller inputs needed to re-derive its decisions offline.
RELATIVE=(
    odometry/local
    trajectory/path
    trajectory/speed
    mpc/predicted_path
    mpc/reference_path
    mpc/goal_point
    mpc/des_yaw_rate
    mpc/des_steer
    mpc/des_speed
    mpc/solve_time
    drive
    accel/local
    obstacles/object_array
    tf
    tf_static
)

# Only record topics that actually exist: a namespace typo would otherwise
# produce a bag that is empty in exactly the way a quiet feed looks.
mapfile -t LIVE < <(ros2 topic list 2>/dev/null)
TOPICS=()
MISSING=()
for t in "${RELATIVE[@]}"; do
    full="$prefix/$t"
    if printf '%s\n' "${LIVE[@]}" | grep -qx -- "$full"; then
        TOPICS+=("$full")
    else
        MISSING+=("$full")
    fi
done
for t in "${EXTRA[@]}"; do
    TOPICS+=("$t")
done

if [[ ${#TOPICS[@]} -eq 0 ]]; then
    echo "ERROR: none of the expected topics are being published under '$prefix'." >&2
    echo "       Check the namespace and that the controller is running." >&2
    exit 1
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "Not published (skipped): ${MISSING[*]}"
fi
echo "Recording ${#TOPICS[@]} topics -> $OUT"
echo "Stop with Ctrl-C (the bag is finalized on SIGINT)."

# --include-hidden-topics is deliberately off; /tf and /tf_static are listed
# explicitly above so a namespaced stack records its own tree, not the global one.
exec ros2 bag record -o "$OUT" "${TOPICS[@]}"
