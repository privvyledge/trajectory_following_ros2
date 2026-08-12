#!/usr/bin/env bash
# Replay a bag recorded by tools/record_viz_bag.sh through the visualizer, with
# no car and no controller.
#
# Usage:
#   tools/replay_viz_bag.sh <bag_dir> [namespace]
#
# Environment knobs:
#   VIZ_BACKEND    rerun (default) | native | both
#   RRD_PATH       write a .rrd instead of opening a viewer (headless)
#   CONNECT_ADDR   stream to a running tools/rerun_tee.sh instead (e.g. 127.0.0.1:9876)
#   SERVE_WEB      1 = serve the web viewer from the visualizer itself
#   RATE           bag playback rate (default 1.0)
#   EGO_RADIUS / SAFE_DISTANCE / EGO_DISC_OFFSETS
#       keep-out geometry to draw. The live keep-out sync polls the controller's
#       parameter service, which does not exist in a replay, so these fallbacks
#       are the only thing keeping the drawn circles equal to what was enforced.
#       Set them to the values the run itself used.
set -euo pipefail

BAG=${1:?usage: replay_viz_bag.sh <bag_dir> [namespace]}
NS=${2:-}
RATE=${RATE:-1.0}
VIZ_BACKEND=${VIZ_BACKEND:-rerun}
EGO_RADIUS=${EGO_RADIUS:-0.212}
SAFE_DISTANCE=${SAFE_DISTANCE:-0.15}
EGO_DISC_OFFSETS=${EGO_DISC_OFFSETS:-[0.045,0.335]}

[[ -d "$BAG" ]] || { echo "ERROR: no such bag directory: $BAG" >&2; exit 1; }

prefix=''
[[ -n "$NS" ]] && prefix="/${NS#/}"

# The visualizer subscribes to the path and speed topics with TRANSIENT_LOCAL
# (they are latched in a live run). `ros2 bag play` republishes VOLATILE by
# default, which is QoS-incompatible with that subscription: the publisher warns
# and the route silently never draws. Override just those two.
QOS=$(mktemp /tmp/replay_qos_XXXX.yaml)
cat > "$QOS" <<EOF
$prefix/trajectory/path:
  history: keep_last
  depth: 1
  reliability: reliable
  durability: transient_local
$prefix/trajectory/speed:
  history: keep_last
  depth: 1
  reliability: reliable
  durability: transient_local
EOF

viz_args=(--ros-args -p use_sim_time:=true
          -p "viz_backend:=$VIZ_BACKEND"
          -p spawn_viewer:=false
          -p "viz_ego_radius:=$EGO_RADIUS"
          -p "viz_safe_distance:=$SAFE_DISTANCE"
          -p "viz_ego_disc_offsets:=$EGO_DISC_OFFSETS")
[[ -n "$NS" ]] && viz_args+=(-r "__ns:=$prefix")
# Never pass an empty string to -p: rcl rejects a bare `-p name:=` at startup.
[[ -n "${RRD_PATH:-}" ]]     && viz_args+=(-p "recording_path:=$RRD_PATH")
[[ -n "${CONNECT_ADDR:-}" ]] && viz_args+=(-p "connect_addr:=$CONNECT_ADDR")
[[ "${SERVE_WEB:-0}" == 1 ]] && viz_args+=(-p serve_web:=true)

cleanup() { [[ -n "${VIZ_PID:-}" ]] && kill "$VIZ_PID" 2>/dev/null || true; rm -f "$QOS"; }
# INT/TERM as well as EXIT: an EXIT trap alone does not fire when the script is
# killed (e.g. under `timeout`), which orphans the visualizer holding the topics.
trap cleanup INT TERM EXIT

ros2 run trajectory_following_ros2 trajectory_visualizer "${viz_args[@]}" &
VIZ_PID=$!

# Let the subscriptions come up before playback: the latched path is republished
# once, at the very start of the bag, and a subscriber that misses it draws no route.
sleep 4

# --clock so the visualizer's sim time matches the recorded stamps; without it
# the entries that stamp themselves from the node clock (yaw rate, solve time)
# land at wall-now and split the rerun timeline in two.
ros2 bag play "$BAG" --clock --rate "$RATE" --qos-profile-overrides-path "$QOS"

# Keep the visualizer alive after playback so the last frame stays on screen --
# except when driving this from a script, where a blocking wait is a hang.
if [[ "${KEEP_OPEN:-auto}" == 0 || ( "${KEEP_OPEN:-auto}" == auto && ! -t 0 ) ]]; then
    echo 'Playback finished; closing the visualizer (KEEP_OPEN=1 to keep it up).'
else
    echo 'Playback finished. Ctrl-C to close the visualizer.'
    wait "$VIZ_PID"
fi
