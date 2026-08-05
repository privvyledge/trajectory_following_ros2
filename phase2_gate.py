#!/usr/bin/env python3
"""Phase 2 qualification gates for the captured CARLA obstacle-208 scene.

RE-SCOPED 2026-08-04. The gate previously demanded the harness reproduce F2, the
terminal wedge abeam obstacle 208. It cannot, by construction, and the target was
misread from the start:

  * `run_H.csv` contains **zero status-4 rows in the entire run** -- the wedge was
    never a solver failure, so it was never bug-365 either.
  * Its last 3220 ticks (~161 s) are byte-identical: every solve optimal, a full-lock
    reverse escape commanded at -0.15 m/s, `safety_reason` empty, and the pose frozen
    to the last decimal at 0.166 m clearance. `run_G.csv` ends the same way over 2234
    ticks. The controller was executing a correct escape; the **plant** was not
    responding.
  * That is a contact event. The replay harness's kinematic bicycle plant has no
    contact model -- a commanded -0.15 m/s always moves it -- so no amount of harness
    work reproduces the wedge.

The original ``approach`` target measures how close each config brings the ego to
obstacle 208. The Task-B ``restart-208`` target starts from the later run-I stop,
where clearance remained positive and the detour-bound latch was engaged. It asks
for physical *forward* progress: reverse does not count because the live plant moved
7 mm under -1.0 m/s while forward +1.0 m/s moved 1.267 m from this exact state.

The restart target deliberately does not model contact. Its scene checks are
controller/trace state: obstacle 208 selected, required detour above the bound, and
positive physical clearance. Latch activity is reported as baseline evidence but is
not a validity requirement, because a candidate that prevents the latch must still
be allowed to pass. Success requires at least 0.5 m of forward motion within the
requested window.

Reported per run so the worst case is visible, never only the aggregate.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

OBSTACLE_208 = (-2.59971, -54.99959)
MAX_STEER = 1.22173
WEDGE_MIN_TICKS = 400        # 20 s at the nominal 20 Hz sim rate
WEDGE_MAX_DISPLACEMENT = 0.5
APPROACH_WINDOW_M = 15.0     # ego-to-208 range that counts as "approaching 208"
LIVE_H_CLEARANCE_M = 0.166   # what run_H.csv pinned at when the plant stopped it
RESTART_FORWARD_YAW = -1.57117
RESTART_MIN_FORWARD_M = 0.5
RESTART_POSE = (0.052675, -53.159458)


def column(rows, name, cast=float):
    out = []
    for r in rows:
        try:
            out.append(cast(r[name]))
        except (TypeError, ValueError, KeyError):
            out.append(float('nan'))
    return np.array(out)


def classify(path, restart_window_s=12.0, control_rate=20.0,
             restart_min_forward_m=RESTART_MIN_FORWARD_M):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return {'run': os.path.basename(path), 'rows': 0, 'mode': 'EMPTY'}
    idx = column(rows, 'ref_idx')
    x, y = column(rows, 'ego_x'), column(rows, 'ego_y')
    steer = column(rows, 'applied_steering')
    speed = column(rows, 'applied_speed')
    avoid = column(rows, 'avoidance_stop')
    required = column(rows, 'avoidance_required_offset')
    bound = column(rows, 'avoidance_bound')
    selected_id = column(rows, 'sel_id')
    physical_clearance = column(rows, 'physical_clearance')
    status = [r['status'] for r in rows]

    # Longest window with a frozen reference index.
    best = (0, 0, 0)
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or idx[i] != idx[start]:
            if i - start > best[0]:
                best = (i - start, start, i)
            start = i
    length, lo, hi = best
    disp = (float(np.nanmax(np.hypot(x[lo:hi] - x[lo], y[lo:hi] - y[lo])))
            if length else float('nan'))
    dist_208 = (math.hypot(float(np.nanmedian(x[lo:hi])) - OBSTACLE_208[0],
                           float(np.nanmedian(y[lo:hi])) - OBSTACLE_208[1])
                if length else float('nan'))
    window = slice(lo, hi)
    commanded = float(np.nanmean(np.abs(speed[window]) > 0.01)) if length else 0.0
    pinned = (float(np.nanmean(np.abs(steer[window]) >= 0.95 * MAX_STEER))
              if length else 0.0)
    hard_fail = sum(1 for s in status if s == '4') / len(rows)

    # Approach criterion (the gate). How close did this config actually come to
    # obstacle 208? `physical_clearance` is body-to-body and excludes safe_distance,
    # so it is directly comparable with the 0.166 m the live run-H trace pinned at.
    # Restricted to the approach window -- ticks with the ego near 208 -- so a
    # clearance logged against some other obstacle elsewhere cannot leak in.
    range_208 = np.hypot(x - OBSTACLE_208[0], y - OBSTACLE_208[1])
    near = range_208 < APPROACH_WINDOW_M
    clearance = physical_clearance
    approach_clear = (float(np.nanmin(clearance[near]))
                      if near.any() and not np.all(np.isnan(clearance[near]))
                      else float('nan'))
    closest_208 = float(np.nanmin(range_208)) if len(range_208) else float('nan')
    reached_208 = bool(near.any())

    stalled = length >= WEDGE_MIN_TICKS and disp < WEDGE_MAX_DISPLACEMENT
    is_f2 = bool(stalled and dist_208 < 6.0 and commanded > 0.5
                 and float(np.nanmax(avoid[window])) == 0 and pinned > 0.5)
    if is_f2:
        mode = 'F2_WEDGE'
    elif stalled and hard_fail > 0.5:
        mode = 'SOLVER_FAILURE_LATCH'
    elif stalled:
        mode = 'OTHER_STALL'
    else:
        mode = 'NO_STALL'
    expected_restart_ticks = max(1, int(round(restart_window_s * control_rate)))
    restart_ticks = min(len(rows), expected_restart_ticks)
    restart_slice = slice(0, restart_ticks)
    forward = np.array([math.cos(RESTART_FORWARD_YAW), math.sin(RESTART_FORWARD_YAW)])
    displacement = np.column_stack((x - x[0], y - y[0]))
    along_track = displacement @ forward
    forward_progress = float(np.nanmax(along_track[restart_slice]))
    reverse_progress = max(0.0, -float(np.nanmin(along_track[restart_slice])))
    reached_forward = np.flatnonzero(
        along_track[restart_slice] >= float(restart_min_forward_m))
    restart_tick = int(reached_forward[0]) if len(reached_forward) else None
    latch_share = float(np.nanmean(avoid[restart_slice] > 0.5))
    bound_block_share = float(np.nanmean(
        required[restart_slice] > bound[restart_slice] + 1e-6))
    selected_208_share = float(np.nanmean(selected_id[restart_slice] == 208))
    min_restart_clearance = (float(np.nanmin(physical_clearance[restart_slice]))
                             if not np.all(np.isnan(physical_clearance[restart_slice]))
                             else float('nan'))
    start_pose_error = math.hypot(x[0] - RESTART_POSE[0], y[0] - RESTART_POSE[1])
    restart_window_complete = len(rows) >= expected_restart_ticks
    restart_scene_valid = bool(
        start_pose_error <= 0.5
        and bound_block_share > 0.0
        and selected_208_share > 0.0
        and not math.isnan(min_restart_clearance)
        and min_restart_clearance >= 0.0)
    restart_setup_valid = bool(restart_scene_valid and restart_window_complete)
    restart_pass = bool(restart_setup_valid and restart_tick is not None)

    return {'run': os.path.basename(path).replace('.csv', ''), 'rows': len(rows),
            'idx_max': int(np.nanmax(idx)), 'stall_ticks': length,
            'stall_idx': int(idx[lo]) if length else -1,
            'displacement_m': round(disp, 3), 'dist_to_208_m': round(dist_208, 2),
            'closest_208_m': round(closest_208, 3), 'reached_208': reached_208,
            'approach_clearance_m': round(approach_clear, 3),
            'restart_window_ticks': restart_ticks,
            'restart_window_complete': restart_window_complete,
            'restart_forward_m': round(forward_progress, 3),
            'restart_reverse_m': round(reverse_progress, 3),
            'restart_tick': restart_tick,
            'restart_latch_share': round(latch_share, 3),
            'restart_bound_block_share': round(bound_block_share, 3),
            'restart_selected_208_share': round(selected_208_share, 3),
            'restart_min_clearance_m': round(min_restart_clearance, 3),
            'restart_start_pose_error_m': round(start_pose_error, 3),
            'restart_scene_valid': restart_scene_valid,
            'restart_setup_valid': restart_setup_valid,
            'restart_pass': restart_pass,
            'commanded_share': round(commanded, 3),
            'steer_pinned_share': round(pinned, 3),
            'status4_share': round(hard_fail, 3), 'mode': mode}


def _print_restart_gate(results, window_s, min_forward_m):
    header = ('run', 'rows', 'restart_forward_m', 'restart_reverse_m', 'restart_tick',
              'restart_latch_share', 'restart_bound_block_share',
              'restart_selected_208_share', 'restart_min_clearance_m',
              'restart_scene_valid', 'restart_window_complete',
              'restart_setup_valid', 'restart_pass')
    print(' '.join(f'{h:>26}' for h in header))
    for result in results:
        print(' '.join(f'{str(result.get(h, "")):>26}' for h in header))

    invalid = [r for r in results if not r['restart_setup_valid']]
    failed = [r for r in results if r['restart_setup_valid'] and not r['restart_pass']]
    print(f'\nrestart-208 gate: require {min_forward_m:.2f} m forward progress '
          f'within {window_s:.1f} s, with bound/selection evidence and no overlap '
          '(latch activity is diagnostic)')
    if invalid:
        print(f'INVALID: {len(invalid)}/{len(results)} run(s) did not establish the gate scene.')
        return 2
    if failed:
        print(f'FAIL: {len(failed)}/{len(results)} run(s) did not restart forward.')
        return 1
    print(f'PASS: {len(results)}/{len(results)} run(s) restarted forward.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='?', default='data/phase2_runH/*.csv')
    parser.add_argument('--target', choices=('approach', 'restart-208'),
                        default='approach')
    parser.add_argument('--restart-window', type=float, default=12.0,
                        help='nominal simulated seconds allowed for forward restart')
    parser.add_argument('--control-rate', type=float, default=20.0)
    parser.add_argument('--restart-min-forward', type=float,
                        default=RESTART_MIN_FORWARD_M)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(args.paths))
    results = [classify(p, args.restart_window, args.control_rate,
                        args.restart_min_forward) for p in paths]
    if not results:
        print(f'No CSV files matched {args.paths!r}.', file=sys.stderr)
        return 2
    if args.target == 'restart-208':
        rc = _print_restart_gate(results, args.restart_window,
                                 args.restart_min_forward)
        print(json.dumps(results, indent=1))
        return rc

    header = ('run', 'rows', 'idx_max', 'stall_ticks', 'stall_idx', 'displacement_m',
              'dist_to_208_m', 'closest_208_m', 'approach_clearance_m',
              'commanded_share', 'steer_pinned_share', 'status4_share', 'mode')
    print(' '.join(f'{h:>20}' for h in header))
    for r in results:
        print(' '.join(f'{str(r.get(h, "")):>20}' for h in header))

    # Diagnostic only -- the wedge is a contact event and is not gated on.
    f2 = sum(1 for r in results if r['mode'] == 'F2_WEDGE')
    latched = sum(1 for r in results if r['mode'] == 'SOLVER_FAILURE_LATCH')
    print(f'\ndiagnostic: F2-shaped wedge in {f2}/{len(results)} runs, '
          f'solver-failure latch in {latched}/{len(results)} '
          f'(neither decides the gate)')

    # The gate: the approach criterion.
    reached = [r for r in results if r['reached_208']]
    clears = [r['approach_clearance_m'] for r in reached
              if not math.isnan(r['approach_clearance_m'])]
    print(f'\napproach to obstacle 208 within {APPROACH_WINDOW_M:.0f} m: '
          f'{len(reached)}/{len(results)} runs')
    if clears:
        print(f'  min body-to-body clearance  min {min(clears):.3f} m  '
              f'median {sorted(clears)[len(clears) // 2]:.3f} m  '
              f'max {max(clears):.3f} m')
        print(f'  live run_H pinned at {LIVE_H_CLEARANCE_M:.3f} m')
    else:
        print('  no run reached the approach window -- nothing to compare.')
    print('\nThis number is the comparison quantity. The gate is decided by '
          'run-G config vs run-H config, so run both batches and diff them; '
          'a single batch cannot pass or fail it on its own.')
    print(json.dumps(results, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
