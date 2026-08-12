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
# Visualization: LOAD_VIZ=true adds rerun. Setting VIZ_RECORDING_PATH as well
# gives the live web viewer AND a .rrd (via tools/rerun_tee.sh); `stop` is what
# finalizes and compacts that file. VIZ_TEE=false keeps the old recording-only
# behaviour.
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
#   3. 'map' is the default because an odom-frame route is only valid inside the
#      session that recorded it. odom has no external datum -- its origin AND its
#      heading are fixed at EKF init, wherever the car sat and pointed when the
#      stack came up -- so replaying one after a restart drives the recorded path
#      ROTATED by the difference between the two sessions' init headings, and
#      nothing detects it: TF resolves, the controller tracks the wrong route to
#      centimetres, every diagnostic reads healthy. Measured 2026-08-06 with the
#      car parked on the route start: position matched to 7 mm, but waypoint 0
#      heads -31.0 deg against odom's -0.2 deg, and the straight 2.0 m reverse
#      tail came out 45.7 deg off the X axis. The 31 deg heading error then
#      pinned steering at the cap for 172 s -- over the 0.2 m distance_tolerance
#      reference anchor it demands an impossible radius, with no yaw authority at
#      v=0 to work it off.
#      Set GLOBAL_FRAME=odom explicitly to drive a *_odomframe.csv route in the
#      same session that recorded it.
GLOBAL_FRAME="${GLOBAL_FRAME:-map}"
ROBOT_FRAME="${ROBOT_FRAME:-base_link}"

# Speed cap for the run. Independent of the joystick MAX_SPEED in 00_env.sh --
# this one bounds what the solver may command.
#
# Like the steering cap below, this CANNOT be set from the launch line:
# config/platforms/f1tenth.yaml pins max_speed 1.5 and the platform overlay is
# applied after the launch args, so `max_speed:=0.8` reads back as 1.5 on the
# running node (measured on the car 2026-08-06). The value lives in the weights
# file that WEIGHTS points at; this variable only tells assert_live_params what
# to expect, so change the two together.
TRACK_MAX_SPEED="${TRACK_MAX_SPEED:-1.0}"

# Steering cap, DEGREES at the road wheel.
#
# Deliberately NOT the platform YAML's 27 deg, which its own comment flags as a
# guess. It is what the servo can DELIVER, from the recalibration measured on this
# car 2026-08-07 (gain -1.1448, offset 0.56, servo clamped to [0.08, 0.92]):
#
#   right lock (servo 0.92): (0.56 - 0.92) / 1.1448 = -0.3145 rad = -18.02 deg
#   left  lock (servo 0.08): (0.56 - 0.08) / 1.1448 = +0.4193 rad = +24.02 deg
#
# The cap is the smaller extreme applied symmetrically. Anything past it on the
# right clips inside the VESC driver with no error reported, and the solver then
# tracks a predicted yaw rate the car never achieves. On the 2026-08-08 ground
# legs the controller commanded past the right-hand bound on ~45% of
# physically-moving ticks (0% past the left), so this was live, not theoretical.
#
# The superseded numbers, so they are not resurrected: 14.72 deg came from an
# older calibration (gain -1.4) produced on a DIFFERENT F1/10; 23 deg was the
# conservative pre-recalibration value chosen from the recordings, which sit at or
# near full lock for 23-52% of each route.
#
# PROVISIONAL: gain and offset are measured, but servo_min/servo_max are inherited
# and servo_min was never reached in any archived bag -- the LEFT extreme has no
# empirical backing. That is why this is symmetric at the evidence-backed right
# bound. Recompute from a servo bench sweep once one is published.
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
# `max_steer/min_steer: +/-18` (diffed -- the cost matrices are identical).
#
# So: change the cap by pointing WEIGHTS at a file that pins it, and verify on
# the running node (assert_effective_steer_cap below does this automatically).
# This variable and config/weights/gosling1_acados_recal.yaml must agree or the
# assert fails; MAX_STEERING in 00_env.sh is f1tenth's side of the same number.
TRACK_MAX_STEER_DEG="${TRACK_MAX_STEER_DEG:-18.0}"

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

# Rerun tee (web viewer + .rrd at once). Ports match what the web viewer URL in
# the skill/handoff quotes, so a tee run and a plain serve_web run are reached
# the same way.
VIZ_TEE_TCP_PORT="${VIZ_TEE_TCP_PORT:-9876}"
VIZ_TEE_WS_PORT="${VIZ_TEE_WS_PORT:-9877}"
VIZ_TEE_WEB_PORT="${VIZ_TEE_WEB_PORT:-9090}"
VIZ_TEE_RUN_DIR="${VIZ_TEE_RUN_DIR:-${SESSION_DIR}/rerun_tee}"

# Existence is NOT enough to call a pid "our controller". The pidfile lives in
# SESSION_DIR on the shared SSD, so it outlives the container that wrote it, and
# after a restart the recorded number is routinely recycled by something else --
# observed 2026-08-10 as the vehicle stack's own base_link->right_front_wheel
# static_transform_publisher. A bare `kill -0` then both blocks a fresh launch
# and, through `stop`, SIGINTs a node the vehicle stack needs. So confirm the
# process is actually the launch we started; anything else is a stale pidfile.
alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'mpc\.launch\.py'
}
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
  # Discovery on this box is intermittently slow, and one cold `ros2 topic list`
  # can return an incomplete graph -- `check` has passed and `launch` failed this
  # same assert seconds later while odometry was publishing normally. Retry
  # before believing a silence, exactly as the parameter reads below do.
  local missing=0 t i topics=''
  for i in 1 2 3 4; do
    topics=$(timeout 20 ros2 topic list 2>/dev/null)
    grep -qx "/${NS}/odometry/local" <<<"$topics" && grep -qx "/${NS}/drive" <<<"$topics" && break
    sleep 2
  done
  for t in "odometry/local" "drive"; do
    if ! grep -qx "/${NS}/${t}" <<<"$topics"; then
      err "topic /${NS}/${t} is not present (4 attempts)"; missing=1
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
  #
  # 10 s was too tight: the listener has to see the frame published before it
  # will answer, and a live AMCL took ~5 s to resolve here while the probe was
  # already reporting 'frame does not exist'. A too-short window is worse than
  # no probe -- it sends the operator to restart a localizer that is running.
  # Do NOT merge stderr and truncate with head: the CycloneDDS config in
  # 00_env.sh emits an unbounded 'ddsi_udp_conn_write ... retcode -3' stream for
  # every unreachable static peer, which filled the whole window and made a
  # working localizer read as absent. Match the transform out of stdout instead
  # and let grep -m1 close the pipe.
  local probe i
  for i in 1 2; do
    probe=$(timeout 20 ros2 run tf2_ros tf2_echo "$target" odom \
              --ros-args -r /tf:="/${NS}/tf" -r /tf_static:="/${NS}/tf_static" \
              2>/dev/null | grep -m1 -A4 'Translation')
    grep -q 'Translation' <<<"$probe" && break
  done
  if grep -q 'Translation' <<<"$probe"; then
    info "localizer is publishing ${target}->odom"
    # Print the correction itself. A localizer that came up but never got an
    # initial pose sits at identity, which is indistinguishable from a healthy
    # one in a pass/fail probe and reproduces the exact failure this guard
    # exists to prevent.
    info "  correction $(grep -m1 '^- Translation' <<<"$probe"), $(grep -m1 'RPY (degree)' <<<"$probe" | sed 's/^- //')"
  else
    die "no ${target}->odom transform on /${NS}/tf after 2 x 20 s.
         A '${target}'-frame route needs a live localizer. Without one the
         controller does not fail -- it tracks the raw odom pose as though it
         were already '${target}', i.e. off by the whole correction.
         Start the localizer, or drive an odom-frame route (data/*_odomframe.csv)."
  fi
}

# A transform existing proves nothing about the car being where the route
# starts, and that gap has now ended two runs. 2026-08-05: an odom-frame route
# replayed after an EKF re-datum put waypoint 0 at -31 deg against the car's
# -0.2 deg. 2026-08-06: a localizer that came up without an initial pose sat at
# identity, leaving the car 0.74 m and 83 deg from waypoint 0 with a map->odom
# transform publishing normally the whole time. Both read as healthy everywhere
# else, and both end the same way: the reference is anchored distance_tolerance
# (0.2 m) ahead, a large heading error there demands a radius the car cannot
# turn, and at v=0 there is no yaw authority to work it off, so steering pins at
# the cap and the pose never changes.
check_start_pose() {
  local csv="$1" target="$2"

  info "comparing the car's ${target} pose against waypoint 0"
  local probe
  probe=$(timeout 20 ros2 run tf2_ros tf2_echo "$target" "$ROBOT_FRAME" \
            --ros-args -r /tf:="/${NS}/tf" -r /tf_static:="/${NS}/tf_static" \
            2>/dev/null | grep -m1 -A4 'Translation')
  if ! grep -q 'Translation' <<<"$probe"; then
    err "could not read ${target}->${ROBOT_FRAME}; skipping the start-pose comparison"
    return 0
  fi

  local car_xy car_yaw
  car_xy=$(grep -m1 '^- Translation' <<<"$probe" | tr -d '[],' | awk '{print $3, $4}')
  # '- Rotation: in RPY (radian) 0.000 0.000 -1.492' once the brackets are gone,
  # so yaw is field 8. Field 6 is the roll term and parses as a clean 0.0 -- a
  # wrong-but-plausible heading, which is the worst kind for a safety gate.
  car_yaw=$(grep -m1 'RPY (radian)' <<<"$probe" | tr -d '[],' | awk '{print $8}')

  python3 - "$csv" $car_xy "$car_yaw" <<'PY' || die "the car is not at the route start."
import csv, math, sys

path, cx, cy, cyaw = sys.argv[1], *map(float, sys.argv[2:5])
with open(path) as fh:
    row = next(iter(csv.DictReader(fh)))
wx, wy, wyaw = float(row['x']), float(row['y']), float(row['yaw'])

dpos = math.hypot(cx - wx, cy - wy)
dyaw = abs(math.degrees(math.atan2(math.sin(cyaw - wyaw), math.cos(cyaw - wyaw))))
print(f"  car        {cx: .3f} {cy: .3f}  {math.degrees(cyaw): .1f} deg")
print(f"  waypoint 0 {wx: .3f} {wy: .3f}  {math.degrees(wyaw): .1f} deg")
print(f"  offset     {dpos:.3f} m, {dyaw:.1f} deg")

# The heading term is the one that wedges the car; position error the reference
# anchor can absorb, heading error at standstill it cannot.
if dpos > 0.75 or dyaw > 30.0:
    print('  FAIL: park the car on the route start, or re-seed the localizer.')
    sys.exit(1)
if dpos > 0.30 or dyaw > 15.0:
    print('  WARN: larger than a clean start; expect a slow, wide first metre.')
PY
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
  # The loader's node name is waypoint_loader_node, not the executable name.
  local loader="/${NS}/waypoint_loader_node"

  # The node needs a moment past process start before its parameter services answer.
  local i
  for i in $(seq 1 10); do
    ros2 node list 2>/dev/null | grep -qx "$node" && break
    sleep 1
  done

  banner "effective parameters (read from the running node, not the launch line)"
  local bad=0
  _p() {  # _p <node> <param> <expected|-> ; prints and flags mismatches
    # The parameter service answers intermittently on a loaded box -- a single
    # timeout is indistinguishable from a genuinely missing parameter, and
    # reporting "could not read" for a value that is in fact correct is how a
    # real mismatch gets lost in the noise. Retry before believing a silence.
    local got i
    for i in 1 2 3 4; do
      got=$(timeout 20 ros2 param get "$1" "$2" 2>/dev/null | sed 's/.*value is: //')
      [[ -n "$got" ]] && break
      sleep 2
    done
    if [[ -z "$got" ]]; then
      err "  $2: could not read from $1 after 4 attempts"; bad=1; return
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

# -------------------------------------------------------------- rerun tee ----
# The tee lives in the package (tools/rerun_tee.sh) while this runner is
# deployed next to 00_env.sh, so look beside the runner first for a deployed
# copy and fall back to the repo checkout inside the container.
find_tee_script() {
  local c
  for c in "${RERUN_TEE_SCRIPT:-}" \
           "./rerun_tee.sh" \
           "/workspaces/f1tenth/src/trajectory_following_ros2/tools/rerun_tee.sh"; do
    [[ -n "$c" && -r "$c" ]] && { echo "$c"; return 0; }
  done
  return 1
}

start_rerun_tee() {
  local out="$1" tee_sh
  if ! tee_sh="$(find_tee_script)"; then
    warn "rerun tee script not found (set RERUN_TEE_SCRIPT) — falling back"
    return 1
  fi
  # A previous run's proxy still holding the ports would serve stale data under
  # a fresh recording, so clear it before starting rather than after failing.
  TCP_PORT="$VIZ_TEE_TCP_PORT" WS_PORT="$VIZ_TEE_WS_PORT" \
    WEB_PORT="$VIZ_TEE_WEB_PORT" RUN_DIR="$VIZ_TEE_RUN_DIR" \
    bash "$tee_sh" stop >/dev/null 2>&1 || true
  if ! TCP_PORT="$VIZ_TEE_TCP_PORT" WS_PORT="$VIZ_TEE_WS_PORT" \
       WEB_PORT="$VIZ_TEE_WEB_PORT" RUN_DIR="$VIZ_TEE_RUN_DIR" \
       bash "$tee_sh" start "$out" > "${SESSION_DIR}/rerun_tee.log" 2>&1; then
    warn "rerun tee failed to start — falling back. Last lines:"
    tail -5 "${SESSION_DIR}/rerun_tee.log" >&2
    return 1
  fi
  grep -aE 'http://[0-9]' "${SESSION_DIR}/rerun_tee.log" | sed 's/^/  /' || true
  return 0
}

# Always safe to call: a no-op when no tee is running.
stop_rerun_tee() {
  local tee_sh
  find_tee_script >/dev/null || return 0
  tee_sh="$(find_tee_script)"
  # This is where the .rrd is flushed and compacted (the CLI saver writes one
  # chunk per message, ~8x the size the SDK would write), so do not skip it and
  # do not SIGKILL the tee instead.
  TCP_PORT="$VIZ_TEE_TCP_PORT" WS_PORT="$VIZ_TEE_WS_PORT" \
    WEB_PORT="$VIZ_TEE_WEB_PORT" RUN_DIR="$VIZ_TEE_RUN_DIR" \
    bash "$tee_sh" stop 2>/dev/null || true
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
  check_start_pose "$csv" "$GLOBAL_FRAME"

  banner "launching $BACKEND MPC on /${NS}"

  # Rerun on the car is 0.22.x, whose SDK is SINGLE-SINK: one process cannot
  # serve the web viewer and write a .rrd, so asking for both used to get the
  # viewer and a silently dropped recording. tools/rerun_tee.sh lifts that by
  # chaining two rerun CLI processes (proxy -> saver), which is why a recording
  # path no longer costs the live view. The tee is preferred whenever a
  # recording is requested; VIZ_TEE=false forces the old exclusive behaviour,
  # and a tee that fails to start falls back to it rather than dropping the
  # recording the operator asked for.
  # Every viz argument is passed only when non-empty -- rcl rejects a bare
  # `-p name:=` with "Couldn't parse parameter override rule" and takes the
  # whole launch down with it.
  local viz_args=()
  if [[ -n "${VIZ_RECORDING_PATH:-}" ]]; then
    mkdir -p "$(dirname "$VIZ_RECORDING_PATH")"
    if [[ "${VIZ_TEE:-true}" == true ]] && start_rerun_tee "$VIZ_RECORDING_PATH"; then
      viz_args+=(viz_connect_addr:="127.0.0.1:${VIZ_TEE_TCP_PORT}" viz_serve_web:=false)
      info "rerun tee: web viewer + recording -> $VIZ_RECORDING_PATH"
    else
      viz_args+=(viz_recording_path:="$VIZ_RECORDING_PATH" viz_serve_web:=false)
      info "rerun recording -> $VIZ_RECORDING_PATH (web viewer off; no tee)"
    fi
  else
    viz_args+=(viz_serve_web:="${VIZ_SERVE_WEB:-true}")
  fi

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
      viz_backend:="${VIZ_BACKEND:-rerun}" \
      "${viz_args[@]}" \
      viz_spawn_viewer:=false \
      > "$MPC_LOG" 2>&1 &
  echo $! > "$MPC_PID_F"

  sleep 8
  pid="$(read_pid "$MPC_PID_F")"
  if ! alive "$pid"; then
    err "controller died within 8 s. Last log lines:"; tail -25 "$MPC_LOG" >&2
    stop_rerun_tee
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

  # SOLVER_CSV is stamped with the invocation time, so the name computed by a
  # `status` call is never the one `launch` gave the node -- status silently
  # printed nothing at all. Read the newest file in the session dir instead.
  local newest; newest="$(ls -t "$SESSION_DIR"/solver_*.csv 2>/dev/null | head -1)"
  [[ -n "$newest" ]] && SOLVER_CSV="$newest"

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

  # After the controller, so the tee captures everything it logged on the way down.
  stop_rerun_tee

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
          check_localizer "$GLOBAL_FRAME"
          check_start_pose "$1" "$GLOBAL_FRAME" ;;
  launch) do_launch "${1:?usage: $0 launch <waypoints.csv>}" ;;
  status) do_status ;;
  stop)   do_stop ;;
  tmux)   do_tmux "${1:?usage: $0 tmux <waypoints.csv>}" ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac
