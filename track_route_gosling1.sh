#!/usr/bin/env bash
# track_route_gosling1.sh — drive a recorded route on gosling1 under the MPC.
#
# Companion to the live-run scripts: 25_drive_session.sh brings the vehicle
# stack up and records bags, this one puts the trajectory controller on top of
# an already-running stack and follows a waypoint CSV produced by
# bag_to_waypoints.py.
#
# Deploy next to 00_env.sh (scripts/live_runs/) so the shared environment,
# namespace and helpers are picked up unchanged.
#
#   ./track_route_gosling1.sh check   out/gosling1_figure8.csv
#   ./track_route_gosling1.sh launch  out/gosling1_figure8.csv
#   ./track_route_gosling1.sh status
#   ./track_route_gosling1.sh stop
#   ./track_route_gosling1.sh tmux    out/gosling1_figure8.csv   # watch panes
#
# Every subcommand returns immediately, so this is drivable from a remote shell.
#
# SAFETY: `launch` publishes drive commands to a real car. `check` does not --
# it only validates the wiring and the CSV. Run `check` first, keep the joystick
# deadman in reach, and have the estop ready.

set -uo pipefail
cd "$(dirname "$0")"
# shellcheck source=00_env.sh
source ./00_env.sh

# --------------------------------------------------------------- config ----
# Backend: acados is the only one that holds 20 Hz on the Jetson with 100%
# optimal solves. do-mpc misses the budget at every rate on this hardware.
BACKEND="${BACKEND:-acados}"
CONTROL_RATE="${CONTROL_RATE:-20.0}"

# Must equal the route CSV's frame_id; check_csv enforces that.
#
# An odom-frame route needs no TF at all. A map-frame route needs two things,
# and the failure mode if either is missing is SILENT rather than loud:
# base_tracker.odom_callback warns once on a failed lookup and then tracks the
# RAW odom pose as though it were already map-frame, i.e. offset by the whole
# map->odom correction (0.75-0.81 m by the end of these drives, and growing).
#
#   1. The controller must receive the namespaced TF. tf2 builds its listener on
#      the ABSOLUTE /tf, which PushRosNamespace does not redirect. mpc.launch.py
#      now remaps /tf and /tf_static under use_namespace (namespaced_tf, default
#      True; verified: the node subscribes to /gosling1/tf).
#   2. A localizer must actually be publishing map->odom. Nothing in the
#      controller can tell "no localizer" from "localizer at identity", so
#      check_stack probes for the transform before the car is allowed to move.
GLOBAL_FRAME="${GLOBAL_FRAME:-odom}"
ROBOT_FRAME="${ROBOT_FRAME:-base_link}"

# Speed cap for the run. Independent of the joystick MAX_SPEED in 00_env.sh --
# this one bounds what the solver may command.
TRACK_MAX_SPEED="${TRACK_MAX_SPEED:-0.8}"

# Steering cap, DEGREES at the road wheel.
#
# Deliberately NOT the platform YAML's 27 deg, which its own comment flags as a
# guess. The competing number is the servo calibration: MAX_STEERING in 00_env.sh
# was chosen for (servo = -1.4 * angle + 0.56, servo_max 0.92), so on paper
# anything beyond (0.92 - 0.56) / 1.4 = 0.257 rad = 14.72 deg clips inside the
# VESC driver, and a solver allowed more plans turns the servo silently refuses
# while tracking a predicted yaw rate the car never achieves.
#
# 23 deg is used anyway, on the evidence of the recordings rather than the
# calibration constants: the gosling1 drives sit at or near full lock for 23-52%
# of each route, and reconstructing them needs ~23 deg. Capping at 14.72 makes
# every one of those corners infeasible by construction -- measured in sim as ~3x
# worse figure-8 CTE p95 with steering saturated ~55% of the time, against ~5% at
# 23. If the servo really does clip at 14.72 the run is no worse off than the cap
# would have made it; if it does not, the cap was throwing away the only steering
# authority these routes need.
#
# UNVERIFIED ON THE CAR (2026-08-06): which of the two numbers is real has never
# been measured. The first hardware run should compare commanded steering against
# the actuator/odometry response above ~15 deg and settle it -- if the response
# flattens there, recalibrate the servo before driving these routes for tracking
# numbers, because the prediction/actual mismatch is silent.
#
# HOW the cap is applied matters, and the obvious way does not work. In
# mpc.launch.py the per-platform/per-backend overlays are applied LAST --
# "weights > platform > args > base" by its own comment -- so the
# `max_steer:=...` launch argument below is OVERRIDDEN by config/platforms/
# f1tenth.yaml's `max_steer: 27.0`. Passing the launch arg alone silently gives
# 27 deg: not the requested cap, and the very value the note above rejects.
#
# The weights overlay is the only lever that outranks the platform file, so the
# cap travels with WEIGHTS. gosling1_acados_recal is f1tenth_acados plus exactly
# `max_steer/min_steer: +/-23` (diffed -- the cost matrices are identical), which
# also makes this the configuration the sim legs were validated against.
#
# So: change the cap by pointing WEIGHTS at a file that pins it, and verify on
# the running node (assert_effective_steer_cap below does this automatically).
# After a real VESC/steering recalibration, update MAX_STEERING in 00_env.sh,
# config/platforms/f1tenth.yaml and the weights file together.
TRACK_MAX_STEER_DEG="${TRACK_MAX_STEER_DEG:-23.0}"

# Weights overlay. Defaults per backend to the one carrying the steering cap
# above; f1tenth_${BACKEND} would leave the platform's 27 deg in force.
case "$BACKEND" in
  acados) WEIGHTS="${WEIGHTS:-gosling1_acados_recal}" ;;
  *)      WEIGHTS="${WEIGHTS:-f1tenth_${BACKEND}}" ;;
esac

# Uniform arc-length resampling of the route, metres, applied by waypoint_loader
# before smoothing. On by default HERE rather than in the loader, whose global
# default stays 0.0 so no existing CARLA or f1tenth route silently changes its
# waypoint count.
#
# 0.05 m matches the ~0.045 m median spacing of the recorded gosling1 drives and
# equals one tick of travel at 1 m/s / 20 Hz. It exists because these recordings
# contain dropouts -- figure8 has a single 0.555 m hole against that median --
# and smoothing does not close them: the spline is evaluated at the input
# parameter values, so it moves waypoints without ever changing their spacing.
# A hole wider than the reference projection can step over stalls the reference
# index silently (every solve optimal, no watchdog).
TRACK_RESAMPLE_SPACING="${TRACK_RESAMPLE_SPACING:-0.05}"

SESSION_DIR="${SSD_ROOT}/run/track"
MPC_PID_F="${SESSION_DIR}/mpc.pid"
MPC_LOG="${SESSION_DIR}/mpc.log"
SOLVER_CSV="${SESSION_DIR}/solver_$(date +%H%M%S).csv"

export ACADOS_SOURCE_DIR="${ACADOS_SOURCE_DIR:-/home/admin/sdks/acados}"
export LD_LIBRARY_PATH="${ACADOS_SOURCE_DIR}/lib:${LD_LIBRARY_PATH:-}"

alive()    { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }
read_pid() { [[ -s "$1" ]] && cat "$1" || true; }

# ---------------------------------------------------------------- checks ----
# The route CSV decides whether the run is meaningful, so it is validated
# before anything is launched rather than after the car has moved.
check_csv() {
  local csv="$1"
  [[ -s "$csv" ]] || die "no such waypoint CSV: $csv"

  local frame rows
  rows=$(( $(wc -l < "$csv") - 1 ))
  frame=$(awk -F, 'NR==2 {gsub(/ /,"",$1); print $1}' "$csv")
  info "route: $csv"
  printf '  %-22s %s\n' "waypoints" "$rows" "frame_id" "$frame"

  [[ "$rows" -ge 2 ]] || die "route has $rows waypoints; need at least 2"
  if [[ "$frame" != "$GLOBAL_FRAME" ]]; then
    die "CSV frame_id '$frame' != global_frame '$GLOBAL_FRAME'.
         The controller does NOT fail on a missing transform -- it warns once and
         silently tracks the raw odom pose as if it were '$frame'. See the
         GLOBAL_FRAME note above.
         Re-extract the route in '$GLOBAL_FRAME', or run: GLOBAL_FRAME=$frame $0 ..."
  fi

  # A recorded drive that ends in a reverse manoeuvre makes a bad route: the
  # reference speed goes negative at the tail and the car backs up at the goal.
  local rev
  rev=$(awk -F, 'NR>1 && $12 < -0.05 {n++} END {print n+0}' "$csv")
  if [[ "$rev" -gt 0 ]]; then
    warn "$rev of $rows waypoints have vx < -0.05 m/s (reverse segments).
          allow_reversing must be true, or re-extract with the reverse trimmed."
  fi
}

check_stack() {
  local missing=0 t
  for t in "odometry/local" "drive"; do
    if ! ros2 topic list 2>/dev/null | grep -qx "/${NS}/${t}"; then
      err "topic /${NS}/${t} is not present"; missing=1
    fi
  done
  [[ "$missing" -eq 0 ]] || die "vehicle stack is not up. Start it with 25_drive_session.sh launch"

  # The controller warm-starts from an echo of its own drive topic. Anything
  # else publishing there fights it for the actuator.
  local pubs
  pubs=$(ros2 topic info "/${NS}/drive" 2>/dev/null | awk '/Publisher count/ {print $3}')
  if [[ "${pubs:-0}" -gt 0 ]]; then
    warn "/${NS}/drive already has ${pubs} publisher(s) — a stale controller would
          fight this one for the actuator. Check with: ros2 node list"
  fi
  info "vehicle stack present (namespace /${NS}, domain ${ROS_DOMAIN_ID})"
}

# A map-frame route is only meaningful if something is publishing the correction.
# Probed explicitly because the controller cannot report its absence: a failed
# lookup degrades to the raw odom pose with a single warning, so a missing
# localizer looks exactly like a healthy run that tracks 0.8 m off the route.
check_localizer() {
  local target="$1"
  [[ "$target" == "odom" ]] && return 0

  info "probing for ${target}->odom on /${NS}/tf (the controller cannot detect its absence)"
  # tf2_echo never exits on success, so timeout's status is meaningless here --
  # it returns 124 either way. The output is what distinguishes the two.
  local probe
  probe=$(timeout 10 ros2 run tf2_ros tf2_echo "$target" odom \
            --ros-args -r /tf:="/${NS}/tf" -r /tf_static:="/${NS}/tf_static" 2>&1 | head -20)
  if grep -q 'Translation' <<<"$probe"; then
    info "localizer is publishing ${target}->odom"
  else
    die "no ${target}->odom transform on /${NS}/tf after 10 s.
         A '${target}'-frame route needs a live localizer. Without one the
         controller does not fail -- it tracks the raw odom pose as though it
         were already '${target}', i.e. off by the whole correction.
         Start the localizer, or drive an odom-frame route (data/*_odomframe.csv)."
  fi
}

# ---------------------------------------------------- post-launch asserts ----
# Read back what the node ACTUALLY resolved. A parameter can be shadowed between
# the launch line and the node -- the overlays in mpc.launch.py are applied after
# the individual launch args, and a node-level `parameters=` entry outranks both
# -- so the launch command is not evidence of anything. Every value here has a
# silent failure mode: the wrong steering cap plans turns the servo refuses, an
# unapplied resample_spacing leaves a waypoint hole that freezes the reference
# index with every solve still optimal, and allow_reversing=false quietly makes
# the reverse tail of these routes untrackable.
assert_live_params() {
  local node="/${NS}/$(case "$BACKEND" in
        acados) echo acados_mpc_node ;;
        casadi) echo casadi_mpc_node ;;
        do_mpc) echo do_mpc_node ;;
        *) echo "${BACKEND}_mpc_node" ;; esac)"
  local loader="/${NS}/waypoint_loader"

  # The node needs a moment past process start before its parameter services answer.
  local i
  for i in $(seq 1 10); do
    ros2 node list 2>/dev/null | grep -qx "$node" && break
    sleep 1
  done

  banner "effective parameters (read from the running node, not the launch line)"
  local bad=0
  _p() {  # _p <node> <param> <expected|-> ; prints and flags mismatches
    local got; got=$(timeout 10 ros2 param get "$1" "$2" 2>/dev/null \
                     | sed 's/.*value is: //')
    if [[ -z "$got" ]]; then
      err "  $2: could not read from $1"; bad=1; return
    fi
    if [[ "$3" != "-" ]] && ! python3 -c "
import sys
try: sys.exit(0 if abs(float('$got')-float('$3'))<1e-6 else 1)
except ValueError: sys.exit(0 if '$got'.strip().lower()=='$3'.strip().lower() else 1)"; then
      err "  $2 = $got   (EXPECTED $3)"; bad=1
    else
      printf '  %-24s %s\n' "$2" "$got"
    fi
  }

  _p "$node"   max_steer               "$TRACK_MAX_STEER_DEG"
  _p "$node"   min_steer               "-${TRACK_MAX_STEER_DEG}"
  _p "$node"   max_speed               "$TRACK_MAX_SPEED"
  _p "$node"   allow_reversing         true
  _p "$node"   arclength_index_advance true
  _p "$node"   control_rate            "$CONTROL_RATE"
  _p "$node"   global_frame            "$GLOBAL_FRAME"
  _p "$loader" resample_spacing        "$TRACK_RESAMPLE_SPACING"

  if [[ "$bad" -ne 0 ]]; then
    err "at least one parameter did not resolve as requested — do NOT drive.
         The overlays in mpc.launch.py (weights > platform > args > base) are the
         usual cause; the platform file pins max_steer 27, so only a weights file
         can change it. Current WEIGHTS=$WEIGHTS"
  else
    info "all checked parameters resolved as requested"
  fi

  # Resampling is only observable in the loader's own spacing report.
  grep -iE "spacing|resampl" "$MPC_LOG" | tail -4 || true
}

# ---------------------------------------------------------------- launch ----
do_launch() {
  local csv; csv="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  mkdir -p "$SESSION_DIR"

  local pid; pid="$(read_pid "$MPC_PID_F")"
  alive "$pid" && die "controller already running (pid $pid). Use: $0 stop"

  check_csv "$csv"
  check_stack
  check_localizer "$GLOBAL_FRAME"

  banner "launching $BACKEND MPC on /${NS}"
  # waypoint_target_frame is pinned to global_frame so the loader and the
  # controller cannot disagree about the path frame.
  nohup ros2 launch trajectory_following_ros2 mpc.launch.py \
      mpc_toolbox:="$BACKEND" \
      control_type:=mpc \
      platform:=f1tenth \
      weights:="$WEIGHTS" \
      use_namespace:=True \
      namespace:="$NS" \
      load_waypoints:=True \
      waypoints_csv:="$csv" \
      resample_spacing:="$TRACK_RESAMPLE_SPACING" \
      waypoint_target_frame:="$GLOBAL_FRAME" \
      global_frame:="$GLOBAL_FRAME" \
      robot_frame:="$ROBOT_FRAME" \
      odom_topic:=odometry/local \
      frequency:="$CONTROL_RATE" \
      max_speed:="$TRACK_MAX_SPEED" \
      max_steer:="$TRACK_MAX_STEER_DEG" \
      min_steer:="-${TRACK_MAX_STEER_DEG}" \
      solver_log_file:="$SOLVER_CSV" \
      load_visualizer:="${LOAD_VIZ:-false}" \
      viz_serve_web:="${VIZ_SERVE_WEB:-true}" \
      viz_spawn_viewer:=false \
      > "$MPC_LOG" 2>&1 &
  echo $! > "$MPC_PID_F"

  sleep 8
  pid="$(read_pid "$MPC_PID_F")"
  if ! alive "$pid"; then
    err "controller died within 8 s. Last log lines:"; tail -25 "$MPC_LOG" >&2
    rm -f "$MPC_PID_F"; return 1
  fi
  info "controller running (pid $pid)"
  printf '  %-22s %s\n' "log" "$MPC_LOG" "solver csv" "$SOLVER_CSV"
  assert_live_params
  warn "the car will move as soon as it receives the path. Deadman ready."
}

# ---------------------------------------------------------------- status ----
do_status() {
  local pid; pid="$(read_pid "$MPC_PID_F")"
  if alive "$pid"; then info "controller up (pid $pid)"; else warn "controller down"; fi

  # Cadence is judged from the solver CSV's own tick interval, never from
  # `ros2 topic hz` -- the CLI subscriber is itself a load on a busy box.
  if [[ -s "$SOLVER_CSV" ]]; then
    python3 - "$SOLVER_CSV" <<'PY'
import csv, sys, statistics
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print('  solver csv is empty'); raise SystemExit
def col(name):
    return [float(r[name]) for r in rows if r.get(name) not in (None, '', 'nan')]
opt = [r for r in rows if r.get('is_optimal') in ('1', 'True', 'true')]
st = col('solve_time_ms')
print('  ticks            %d' % len(rows))
print('  optimal          %.1f%%' % (100.0 * len(opt) / len(rows)))
if st:
    st.sort()
    print('  solve ms p50/p95 %.2f / %.2f' % (st[len(st)//2], st[int(len(st)*0.95)]))
stops = [r for r in rows if r.get('safety_reason')]
if stops:
    print('  safety stops     %d (%s)' % (len(stops),
          ', '.join(sorted({r['safety_reason'] for r in stops}))))
PY
  fi
  grep -c 'Final goal reached' "$MPC_LOG" 2>/dev/null \
    | awk '{printf "  goal reached     %s\n", $1}'
}

# ------------------------------------------------------------------ stop ----
do_stop() {
  local pid; pid="$(read_pid "$MPC_PID_F")"
  if alive "$pid"; then kill -INT "$pid" 2>/dev/null; sleep 3; fi

  # A SIGINT to `ros2 launch` does not always take the nodes with it, and an
  # orphaned controller keeps publishing to /drive. Kill by executable path.
  pkill -f "[t]rajectory_following_ros2/lib" 2>/dev/null; sleep 2
  pkill -9 -f "[t]rajectory_following_ros2/lib" 2>/dev/null; sleep 1
  rm -f "$MPC_PID_F"

  local orphans; orphans=$(ps -eo args | grep -cE '[t]rajectory_following_ros2/lib')
  if [[ "$orphans" -gt 0 ]]; then
    err "$orphans controller process(es) still alive — do NOT start another run"
  else
    info "controller stopped, no orphans"
  fi
}

# ------------------------------------------------------------------ tmux ----
# Four panes: controller log, solver stats, the drive command going to the car,
# and a free shell. Everything the operator needs to abort is visible at once.
do_tmux() {
  local csv="$1" s="track"
  command -v tmux >/dev/null || die "tmux not installed in this container"
  tmux has-session -t "$s" 2>/dev/null && die "tmux session '$s' exists: tmux attach -t $s"

  tmux new-session  -d -s "$s" -n run "cd $PWD && ./track_route_gosling1.sh launch '$csv'; exec bash"
  tmux split-window -t "$s:run" -h "cd $PWD && sleep 12 && watch -n2 ./track_route_gosling1.sh status; exec bash"
  tmux split-window -t "$s:run".0 -v "sleep 12 && tail -f '$MPC_LOG'; exec bash"
  tmux split-window -t "$s:run".2 -v "sleep 12 && ros2 topic echo /${NS}/drive --once --full-length; exec bash"
  tmux select-layout -t "$s:run" tiled
  info "tmux session '$s' started:  tmux attach -t $s"
  warn "pane 0 launches the controller — the car moves. Ctrl-C there to abort."
}

# ------------------------------------------------------------------ main ----
cmd="${1:-}"; shift || true
case "$cmd" in
  check)  print_env; check_csv "${1:?usage: $0 check <waypoints.csv>}"; check_stack
          check_localizer "$GLOBAL_FRAME" ;;
  launch) do_launch "${1:?usage: $0 launch <waypoints.csv>}" ;;
  status) do_status ;;
  stop)   do_stop ;;
  tmux)   do_tmux "${1:?usage: $0 tmux <waypoints.csv>}" ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac
