"""Summarize a solver-stats CSV against the obstacle-safety acceptance criteria.

Usage:
    python3 analyze_safety_run.py <csv> [--safe-distance M]

``safe_distance`` MUST match the run's config, because the CSV logs
``sel_ego_clearance`` (body-to-body minus the comfort margin) and physical clearance is
``sel_ego_clearance + safe_distance``. Passing the wrong value silently shifts every
clearance number and the physical-overlap count by the difference: the F1/10 configs use
0.15 m, the CARLA ones 0.4 m, so reading a CARLA run with the default understates
clearance by the difference.
"""
import argparse
import csv
import sys

import numpy as np

# Per-config values, for reference: f1tenth_{acados,casadi,casadi_qp}_obstacle.yaml use
# 0.15; carla_{acados,casadi}_obstacle.yaml use 0.4.
DEFAULT_SAFE_DISTANCE = 0.15

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('csv', nargs='?', default='/tmp/run_acceptance_native.csv',
                    help='solver-stats CSV written by the solver_log_file parameter')
parser.add_argument('--safe-distance', '-s', type=float, default=DEFAULT_SAFE_DISTANCE,
                    help=f'safe_distance (m) used for the run (default '
                         f'{DEFAULT_SAFE_DISTANCE}, the F1/10 value; CARLA configs use 0.4)')
args = parser.parse_args()

path = args.csv
safe_distance = args.safe_distance
rows = list(csv.DictReader(open(path)))
print(f'{path}: {len(rows)} rows  (safe_distance = {safe_distance} m)')
if not rows:
    sys.exit(0)


def col(name, cast=float):
    out = []
    for r in rows:
        try:
            out.append(cast(r[name]))
        except (TypeError, ValueError):
            out.append(float('nan') if cast is float else None)
    return np.array(out) if cast is float else out


idx = col('ref_idx')
nsel = col('n_selected')
ego = col('sel_ego_clearance')
phys = ego + safe_distance
tick = col('tick_interval_ms')
stop = col('safety_stop')
reasons = col('safety_reason', str)
applied_v = col('applied_speed')
proposed_v = col('velocity_cmd')
avoidance_stop = (col('avoidance_stop') if 'avoidance_stop' in rows[0]
                  else np.zeros(len(rows)))

print(f'ref_idx: max {np.nanmax(idx):.0f}')
print(f'n_selected: {np.unique(nsel[~np.isnan(nsel)])}  '
      f'(rows with 2: {int(np.nansum(nsel == 2))})')
obs_rows = nsel > 0
if obs_rows.any():
    print(f'min sel_ego_clearance: {np.nanmin(ego[obs_rows]):+.6f}')
    print(f'min physical clearance: {np.nanmin(phys[obs_rows]):+.6f}')
    print(f'physical-overlap rows: {int(np.nansum(phys[obs_rows] < 0))}')
print(f'tick ms: p50 {np.nanpercentile(tick, 50):.3f}  '
      f'p95 {np.nanpercentile(tick, 95):.3f}  max {np.nanmax(tick):.3f}')

# Solve time is what the controller can actually control; tick interval also absorbs OS
# scheduling. A solve much longer than the tick budget stalls the loop while the plant
# keeps integrating the last command, which is how a CARLA-scale run loses the path.
solve = col('solve_time_ms')
if not np.all(np.isnan(solve)):
    print(f'solve ms: p50 {np.nanpercentile(solve, 50):.3f}  '
          f'p95 {np.nanpercentile(solve, 95):.3f}  '
          f'p99 {np.nanpercentile(solve, 99):.3f}  max {np.nanmax(solve):.3f}')

# Per-phase wall vs this-thread CPU. The wall-only numbers could not distinguish "the
# phase computed for that long" from "the thread was descheduled mid-phase". A ratio
# near 1 means real compute; a large ratio means the process lost the CPU and no amount
# of optimising that phase will speed the tick up.
phase_pairs = [('reference', 'reference_ms', 'reference_cpu_ms'),
               ('obstacle', 'obstacle_ms', 'obstacle_cpu_ms'),
               ('solver', 'solver_wall_ms', 'solver_cpu_ms'),
               ('tick(pre-log)', 'pre_log_ms', 'pre_log_cpu_ms')]
if any(cpu in rows[0] for _, _, cpu in phase_pairs):
    print('phase wall vs thread-cpu (p50 ms):')
    for label, wall_key, cpu_key in phase_pairs:
        if cpu_key not in rows[0]:
            continue
        w, c = col(wall_key), col(cpu_key)
        if np.all(np.isnan(w)) or np.all(np.isnan(c)):
            continue
        wp, cp = np.nanpercentile(w, 50), np.nanpercentile(c, 50)
        ratio = wp / cp if cp > 1e-9 else float('inf')
        verdict = 'compute-bound' if ratio < 2 else 'DESCHEDULED'
        print(f'  {label:<14} wall {wp:8.3f}  cpu {cp:8.3f}  '
              f'x{ratio:6.1f}  {verdict}')

n_stop = int(np.nansum(stop))
print(f'safety stops: {n_stop} / {len(rows)} rows')
by_reason = {}
for r in reasons:
    if r:
        by_reason[r] = by_reason.get(r, 0) + 1
print(f'  by reason: {by_reason}')
longest = cur = 0
for s in stop:
    cur = cur + 1 if s else 0
    longest = max(longest, cur)
print(f'  longest consecutive stop run: {longest}')
print(f'avoidance-stop reference rows: {int(np.nansum(avoidance_stop))}')
divergent = int(np.nansum(np.abs(applied_v - proposed_v) > 1e-9))
print(f'rows where applied != proposed speed: {divergent}')
status = {}
for r in rows:
    status[r['status']] = status.get(r['status'], 0) + 1
print(f'status counts: {status}')
