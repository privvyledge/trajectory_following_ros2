#!/usr/bin/env python3
"""Phase 2 offline replay harness: batch-run the CARLA obstacle scene, one manifest per run.

Topology, matching the vehicle exactly::

    obstacle_replay_publisher  (dynamic -> source 0, static -> source 1)
        -> obstacle_aggregator (ego gate, id namespacing)
            -> closed_loop_sim.launch.py  (waypoint_loader + simulator + acados MPC)

The dynamic capture is source 0 so its ids pass through the aggregator unchanged --
the F2 obstacle stays id 208, as run G and run H logged it. Dynamic object 197 is the
**ego vehicle** (the capture was taken after it wedged, so it sits on the stall pose);
it is excluded, exactly as the live bridge excluded it.

Runs start at route index 0 with the untracked route CSV, which is what runs G and H
did -- the plan's idx-63 start would put the vehicle 30 m along a path whose monotonic
projection anchor is still at index 0, inside a 5 m ``projection_window``, so the
reference could not acquire. The 55 m run-up costs ~7 s of wall clock per run.

Scenarios:
  ``--scenario route``        existing route-from-index-0 replication batch.
  ``--scenario restart-208``  start at the measured run-I wedge pose beside obstacle
                              208, stationary, on a route slice whose local index 0 is
                              close enough for the projection anchor to acquire.

Configs:
  ``--config h``  historical run-H values (max_avoidance_offset 3.5,
                  min_obstacle_radius 0.1).
  ``--config g``  safe/reverted run-G values (2.0 / 0.3).

Both sets are applied and read back via ``ros2 param set``. This keeps manifests true
after the working-tree YAML was reverted to 2.0 / 0.3 and touches no YAML.

Usage::

    python3 phase2_batch_runner.py --generate      # one codegen warm-up run first
    python3 phase2_batch_runner.py --config h
    python3 phase2_batch_runner.py --config g
    python3 phase2_batch_runner.py --scenario restart-208 --config g --duration 30
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
ROUTE_CSV = os.path.join(REPO, 'data', 'carla_town01_moving.csv')
STATIC_JSON = os.path.join(REPO, 'data', 'carla_static_obstacles_town01.json')
DYNAMIC_JSON = os.path.join(REPO, 'data', 'carla_dynamic_objects_town01.json')
EGO_OBJECT_ID = 197
# Node names differ between mpc.launch.py and closed_loop_sim.launch.py for the same
# executable; this is the closed_loop_sim one.
CONTROLLER_NODE = '/kinematic_coupled_acados_controller'

# Ego gate: 80 m reproduces the live gated feed object-for-object at the stall pose
# (295 objects, verified by phase2_topology_check.py against the merged capture).
EGO_GATE_RADIUS = 80.0

# Route index whose waypoint is nearest obstacle 208 at (-2.600, -55.000): the F2
# scene's obstacle. The segment target for "clearly past and rejoined" is this + 60.
OBSTACLE_208_INDEX = 136

# Task-B restart gate. The pose is the byte-stable run-I tail after the controller
# stopped beside obstacle 208. Starting the full route here cannot acquire the
# monotonic projection anchor (its 5 m window begins at route index 0), so the replay
# uses an otherwise-verbatim route slice beginning 12 waypoints before the nearest
# route point. This is a harness fixture, not a controller re-anchoring change.
RESTART_208_STATE = {
    'initial_x': 0.052675,
    'initial_y': -53.159458,
    'initial_yaw': -1.334817,
    'initial_speed': 0.0,
}
RESTART_ROUTE_START_INDEX = 120
RESTART_WINDOW_TICKS = 240
# The controller is intentionally lockstepped, but solver throughput is a wall-time
# property. A prior 30 s run produced only 212 rows, so give the 12 s simulated
# window a measured wall-time margin. If a loaded host still falls short, continue
# while rows are arriving and fail only when the solver log itself stops advancing.
RESTART_MIN_WALL_DURATION_S = 45.0
RESTART_ROW_STALL_TIMEOUT_S = 60.0

RUN_G_PARAMS = {'max_avoidance_offset': 2.0, 'min_obstacle_radius': 0.3}
RUN_H_PARAMS = {'max_avoidance_offset': 3.5, 'min_obstacle_radius': 0.1}

# Fixed replication protocol: five start-station perturbations plus one identical
# repeat of the baseline, so residual nondeterminism is measured rather than assumed.
# Candidate and baseline always run this same set.
CONDITIONS = [
    ('s0', 0), ('s1', 1), ('s2', 2), ('s3', 3), ('s4', 4), ('s0_repeat', 0),
]


def route_state(index):
    """Return the route pose/speed at ``index`` as launch-argument floats."""
    import csv as _csv
    with open(ROUTE_CSV) as handle:
        rows = list(_csv.DictReader(handle, skipinitialspace=True))
    row = rows[index]
    return {'initial_x': float(row['x']), 'initial_y': float(row['y']),
            'initial_yaw': float(row['yaw']), 'initial_speed': float(row['vx'])}


def write_route_slice(out_dir, start_index):
    """Write a route suffix whose local index zero can acquire a mid-route pose."""
    import csv as _csv
    with open(ROUTE_CSV, newline='') as handle:
        reader = _csv.DictReader(handle, skipinitialspace=True)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not 0 <= start_index < len(rows):
        raise ValueError(f'route slice start {start_index} outside 0..{len(rows) - 1}')
    path = os.path.join(out_dir, f'route_from_{start_index}.csv')
    with open(path, 'w', newline='') as handle:
        writer = _csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[start_index:])
    return path


def git_head():
    def _run(cmd):
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {'head': _run(['git', 'rev-parse', 'HEAD']),
            'branch': _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD']),
            'dirty': bool(_run(['git', 'status', '--porcelain']))}


def _env(domain):
    env = dict(os.environ)
    env['ROS_DOMAIN_ID'] = str(domain)
    return env


class Proc:
    """A process group we can always tear down, even if the launch hangs."""

    def __init__(self, name, cmd, env, log_path):
        self.name = name
        self.handle = open(log_path, 'wb')
        self.proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=self.handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)

    def stop(self, grace=4.0):
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            if self.proc.poll() is not None:
                break
            try:
                os.killpg(os.getpgid(self.proc.pid), sig)
            except ProcessLookupError:
                break
            deadline = time.time() + grace
            while time.time() < deadline and self.proc.poll() is None:
                time.sleep(0.2)
        self.handle.close()


def assert_graph_clean(env, discovery_grace=10.0):
    """Refuse a live graph, allowing DDS discovery to forget the prior clean exit."""
    deadline = time.time() + discovery_grace
    while True:
        out = subprocess.run(['ros2', 'node', 'list'], env=env, capture_output=True,
                             text=True, timeout=30).stdout.split()
        stray = [n for n in out if n not in ('/_ros2cli_daemon', '')]
        if not stray:
            return
        if time.time() >= deadline:
            raise RuntimeError(
                f'ROS graph on domain {env["ROS_DOMAIN_ID"]} is not empty: {stray}. '
                'Tear the previous run down before starting a batch.')
        time.sleep(0.5)


def run_once(args, label, start_index, out_dir):
    env = _env(args.domain)
    assert_graph_clean(env)

    if args.scenario == 'restart-208':
        state = dict(RESTART_208_STATE)
        route_csv = write_route_slice(out_dir, RESTART_ROUTE_START_INDEX)
        route_index_offset = RESTART_ROUTE_START_INDEX
    else:
        state = route_state(start_index)
        route_csv = ROUTE_CSV
        route_index_offset = 0
    csv_path = os.path.join(out_dir, f'{label}.csv')
    manifest_path = os.path.join(out_dir, f'{label}.manifest.json')

    launch_args = {
        'mpc_toolbox': 'acados',
        'simulator': 'acados',
        'platform': 'carla',
        'weights': 'carla_acados_obstacle',
        'stage_cost_type': 'EXTERNAL',
        'terminal_cost_type': 'EXTERNAL',
        'num_obstacles': '10',
        'generate_mpc_model': 'true' if args.generate else 'false',
        'waypoints_csv': route_csv,
        'obstacle_topic': 'obstacles/object_array',
        'solver_log_file': csv_path,
        'launch_visualizer': 'false',
        # Lockstep. Live runs G/H ticked at ~4.5 Hz of WALL clock under CARLA's
        # RTF 0.2 -- i.e. one command per 50 ms of *simulated* time, the nominal
        # 20 Hz. Free-running offline at wall RTF 1, the same controller reaches
        # only ~10 Hz wall (obstacle deserialization plus five nodes on one box),
        # so the plant would advance ~100 ms per command: half the effective
        # control rate the vehicle had, which diverges an 8 m/s car on its own and
        # would be mistaken for the failure under study. step_on_command restores
        # the 1-command-per-dt relationship and makes the run cadence-independent
        # and deterministic, which the replication protocol assumes.
        'step_on_command': 'true',
        'forward_escape_enabled': ('true' if args.forward_escape == 'on' else 'false'),
        # Per-batch codegen directory. The default is shared, so a second sim that
        # regenerates the OCP (with a different num_obstacles, say) silently replaces
        # the compiled solver this batch keeps starting runs against, and every later
        # run solves a different problem than its manifest claims.
        'code_gen_directory': os.path.join(out_dir, 'mpc'),
        **{k: repr(v) for k, v in state.items()},
    }
    if getattr(args, 'failure_dump', False):
        # First hard failure only (the adapter latches _failure_dump_attempted), which
        # is exactly the tick that matters: the transition into the latch.
        launch_args['acados_failure_dump_file'] = os.path.join(
            out_dir, f'{label}.failure.npz')

    procs = []
    try:
        procs.append(Proc('publisher', [
            'ros2', 'run', 'trajectory_following_ros2', 'obstacle_replay_publisher',
            '--ros-args',
            '-p', f"snapshot_files:=['{DYNAMIC_JSON}','{STATIC_JSON}']",
            '-p', "output_topics:=['replay/dynamic_objects','replay/static_obstacles']",
            '-p', "publish_qos:=['volatile','volatile']",
            '-p', f'exclude_ids:=[{EGO_OBJECT_ID}]',
            '-p', 'publish_rate:=10.0',
        ], env, os.path.join(out_dir, f'{label}.publisher.log')))

        procs.append(Proc('aggregator', [
            'ros2', 'run', 'trajectory_following_ros2', 'obstacle_aggregator',
            '--ros-args',
            '-p', "input_topics:=['replay/dynamic_objects','replay/static_obstacles']",
            '-p', "input_qos:=['volatile','volatile']",
            '-p', 'output_topic:=obstacles/object_array',
            '-p', f'ego_gate_radius:={EGO_GATE_RADIUS}',
            '-p', 'odom_topic:=odometry/local',
            '-p', 'publish_rate:=10.0',
            # Lockstep advances simulation time independently of executor wall time;
            # retain replay sources across transient host-load stalls.
            '-p', 'source_timeout:=10.0',
        ], env, os.path.join(out_dir, f'{label}.aggregator.log')))

        procs.append(Proc('sim', [
            'ros2', 'launch', 'trajectory_following_ros2', 'closed_loop_sim.launch.py',
        ] + [f'{k}:={v}' for k, v in launch_args.items()],
            env, os.path.join(out_dir, f'{label}.sim.log')))

        overrides = dict(RUN_G_PARAMS if args.config == 'g' else RUN_H_PARAMS)
        overrides['forward_escape_enabled'] = args.forward_escape == 'on'
        _apply_overrides(env, overrides,
                         os.path.join(out_dir, f'{label}.params.log'),
                         args.param_wait)

        run_duration = (max(args.duration, RESTART_MIN_WALL_DURATION_S)
                        if args.scenario == 'restart-208' else args.duration)
        run_started = time.time()
        minimum_deadline = run_started + run_duration
        last_rows = -1
        row_stall_deadline = minimum_deadline + RESTART_ROW_STALL_TIMEOUT_S
        while True:
            time.sleep(2.0)
            if procs[-1].proc.poll() is not None:
                break
            now = time.time()
            rows = (sum(1 for _ in open(csv_path)) - 1
                    if os.path.exists(csv_path) else -1)
            if rows > last_rows:
                last_rows = rows
                row_stall_deadline = now + RESTART_ROW_STALL_TIMEOUT_S
            window_complete = (args.scenario != 'restart-208'
                               or rows >= RESTART_WINDOW_TICKS)
            if now >= minimum_deadline and window_complete:
                break
            if (args.scenario != 'restart-208' and now >= minimum_deadline
                    or args.scenario == 'restart-208' and now >= row_stall_deadline):
                break
        elapsed_wall_s = time.time() - run_started

        resolved = _dump_params(env)
    finally:
        for proc in reversed(procs):
            proc.stop()
        time.sleep(2.0)

    manifest = {
        'label': label,
        'scenario': args.scenario,
        'config': args.config,
        'condition_start_index': start_index,
        'route_index_offset': route_index_offset,
        'start_state': state,
        'git': git_head(),
        'launch_args': launch_args,
        'runtime_param_overrides': overrides,
        'runtime_overrides_applied_via': 'ros2 param set',
        'resolved_params': resolved,
        'obstacle_208_route_index': OBSTACLE_208_INDEX,
        'ego_gate_radius': EGO_GATE_RADIUS,
        'excluded_object_ids': [EGO_OBJECT_ID],
        'requested_duration_s': run_duration,
        'elapsed_wall_s': elapsed_wall_s,
        'domain_id': args.domain,
        'csv': csv_path,
    }
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2)
    if args.scenario == 'restart-208':
        rows = sum(1 for _ in open(csv_path)) - 1 if os.path.exists(csv_path) else -1
        if rows < RESTART_WINDOW_TICKS:
            raise RuntimeError(
                f'{label} produced {rows} solver rows; restart-208 requires at least '
                f'{RESTART_WINDOW_TICKS}. The run is incomplete and must not be gated.')
    return csv_path, manifest_path


def _apply_overrides(env, overrides, log_path, wait):
    """Push hot-reloadable values onto the running controller, and verify each landed.

    A ``ros2 param set`` that silently fails would run the batch under the *other*
    config and every conclusion drawn from it would be about the wrong thing, so each
    set is read back rather than assumed.
    """
    time.sleep(wait)
    with open(log_path, 'w') as handle:
        for name, value in overrides.items():
            for cmd in (['ros2', 'param', 'set', CONTROLLER_NODE, name, str(value)],
                        ['ros2', 'param', 'get', CONTROLLER_NODE, name]):
                result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                        timeout=60)
                handle.write(f'$ {" ".join(cmd)}\n{result.stdout}{result.stderr}\n')
                handle.flush()
            if str(value) not in result.stdout:
                raise RuntimeError(
                    f'{name} did not take the override {value}: {result.stdout!r}. '
                    'The batch would have run under the wrong config.')


def _dump_params(env):
    result = subprocess.run(['ros2', 'param', 'dump', CONTROLLER_NODE],
                            env=env, capture_output=True, text=True, timeout=60)
    return result.stdout if result.returncode == 0 else f'<dump failed: {result.stderr}>'


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', choices=('h', 'g'), default='h')
    parser.add_argument('--scenario', choices=('route', 'restart-208'), default='route',
                        help='route batch or the stopped obstacle-208 restart gate')
    parser.add_argument('--forward-escape', choices=('off', 'on'), default='off',
                        help='select the forward-escape A/B arm (default: off baseline)')
    parser.add_argument('--domain', type=int, default=44)
    parser.add_argument('--duration', type=float, default=120.0,
                        help='wall seconds per run (a wedge window needs >= 20 s)')
    parser.add_argument('--param-wait', type=float, default=25.0,
                        help='seconds to wait for the controller before ros2 param set')
    parser.add_argument('--generate', action='store_true',
                        help='rebuild the acados OCP (needed once per structural change)')
    parser.add_argument('--only', default='', help='run a single condition label')
    parser.add_argument('--repeat', type=int, default=1,
                        help='run each condition N times, labelled <label>_rN')
    parser.add_argument('--failure-dump', action='store_true',
                        help='write the first acados hard-failure replay dump per run')
    parser.add_argument('--out', default='')
    args = parser.parse_args()

    default_dir = ('phase2_restart208' if args.scenario == 'restart-208'
                   else f'phase2_run{args.config.upper()}')
    out_dir = args.out or os.path.join(REPO, 'data', default_dir)
    os.makedirs(out_dir, exist_ok=True)

    base_conditions = ([('restart_208', RESTART_ROUTE_START_INDEX)]
                       if args.scenario == 'restart-208' else CONDITIONS)
    conditions = [c for c in base_conditions if not args.only or c[0] == args.only]
    if args.repeat > 1:
        conditions = [(f'{label}_r{n}', idx)
                      for label, idx in conditions
                      for n in range(1, args.repeat + 1)]
    results = []
    for label, start_index in conditions:
        print(f'--- {args.config.upper()}/{label} (start idx {start_index}) ---',
              flush=True)
        csv_path, manifest = run_once(args, label, start_index, out_dir)
        args.generate = False  # codegen only on the first run of a batch
        results.append((label, csv_path, manifest))
        print(f'    csv={csv_path}', flush=True)

    print('\nBatch complete:')
    lines = []
    for label, csv_path, _m in results:
        rows = sum(1 for _ in open(csv_path)) - 1 if os.path.exists(csv_path) else -1
        lines.append(f'  {label:10s} {rows:6d} rows  {csv_path}')
        print(lines[-1])
    # One immutable completion record per invocation. BATCH_DONE remains a convenient
    # latest-run pointer, but says explicitly that it is not a directory aggregate.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    summary = (f'Invocation completed {stamp}; this summary covers only the files '
               f'listed below, not prior CSVs in {out_dir}.\n'
               + '\n'.join(lines) + '\n')
    with open(os.path.join(out_dir, f'BATCH_DONE_{stamp}_{os.getpid()}'), 'w') as handle:
        handle.write(summary)
    with open(os.path.join(out_dir, 'BATCH_DONE'), 'w') as handle:
        handle.write(summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
