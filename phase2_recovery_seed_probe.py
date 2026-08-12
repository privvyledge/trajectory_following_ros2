"""Probe: is the persistent status-4 latch a property of the STATE or of solver memory?

Takes a first-failure dump as the geometry/reference fixture and re-solves it from a
freshly created solver under a sweep of ego speeds, including the v=0 standstill the
latched runs sit at. A cold solver has zero dual/QP memory, so a failure here is a
property of the problem; a success means the live latch came from the solver's own
accumulated iterate and a reset should have cleared it.
"""
import argparse
import os

import numpy as np

from trajectory_following_ros2.coupled_kinematic_acados import AcadosSolverAdapter


def _scalar(data, name, default=None):
    if name not in data or data[name].size == 0:
        return default
    return data[name].reshape(-1)[0].item()


def build_adapter(data, config_file):
    from acados_template import AcadosOcpSolver
    controller = AcadosOcpSolver.create_cython_solver(config_file)
    horizon = int(_scalar(data, 'horizon'))
    has_weights = bool(_scalar(data, 'has_weight_params', 0))
    rate_max = data.get('rate_max', np.array([]))
    weight_args = {}
    if has_weights:
        weight_args = {
            'Q': np.diag(data['Q_diag']), 'R': np.diag(data['R_diag']),
            'Qe': np.diag(data['Qe_diag']), 'Rd': np.diag(data['Rd_diag']),
        }
    adapter = AcadosSolverAdapter(
        controller, horizon=horizon,
        wheelbase=float(_scalar(data, 'wheelbase')),
        stage_cost_type=str(_scalar(data, 'stage_cost_type', 'EXTERNAL')),
        terminal_cost_type=str(_scalar(data, 'terminal_cost_type', 'EXTERNAL')),
        num_obstacles=int(_scalar(data, 'num_obstacles', 0)),
        ego_radius=float(_scalar(data, 'ego_radius', 1.0)),
        has_weight_params=has_weights, dt=float(_scalar(data, 'dt', 0.05)),
        u_min=data['u_min'], u_max=data['u_max'],
        rate_max=(rate_max if rate_max.size else None),
        solver_config_file=config_file, **weight_args)
    obstacle_states = data.get('obstacle_states', np.array([]))
    if obstacle_states.size:
        adapter.update_obstacles(obstacle_states)
    return adapter, horizon


def _rollout(x0, horizon, wheelbase, dt):
    from trajectory_following_ros2.utils.trajectory_utils import predict_state_rk4
    xs = [np.asarray(x0, dtype=float)]
    for _ in range(horizon):
        xs.append(predict_state_rk4(xs[-1], np.zeros(2), dt, wheelbase))
    return xs


def patch_recovery(adapter, mode, wheelbase, dt):
    """Swap the adapter's post-failure re-seed for a candidate strategy.

    'ref' is the shipped behaviour (stage 0 at x0, stages 1..N on the reference).
    The alternatives keep reset() -- which is what clears a NaN iterate --
    and only change what is seeded afterwards.
    """
    if mode == 'ref':
        return

    horizon = adapter._horizon

    def recover(x0, xref, _mode=mode):
        try:
            adapter._controller.reset()
        except Exception:
            pass
        x0 = np.asarray(x0, dtype=float)
        if _mode == 'tile':
            xs = [x0] * (horizon + 1)
        elif _mode == 'rollout':
            xs = _rollout(x0, horizon, wheelbase, dt)
        else:
            raise ValueError(_mode)
        for i in range(horizon + 1):
            adapter._controller.set(i, 'x', xs[i])
        for i in range(horizon):
            adapter._controller.set(i, 'u', np.zeros(2))

    adapter._recover_from_failure = recover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump')
    parser.add_argument('--config', default='')
    parser.add_argument('--repeats', type=int, default=8,
                        help='consecutive solves per condition on ONE solver instance')
    parser.add_argument('--recovery', default='ref', choices=('ref', 'tile', 'rollout'),
                        help='post-failure re-seed strategy to exercise')
    parser.add_argument('--seed', default='',
                        help='force one initial seed (default: sweep all)')
    parsed = parser.parse_args()

    with np.load(os.path.expanduser(parsed.dump), allow_pickle=False) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}
    config_file = parsed.config or str(_scalar(data, 'solver_config_file', ''))
    config_file = os.path.abspath(os.path.expanduser(config_file))

    x0 = data['x0']
    xref = data['xref']
    wheelbase = float(_scalar(data, 'wheelbase'))
    dt = float(_scalar(data, 'dt', 0.05))
    print(f'fixture x0={np.array2string(x0, precision=3)} '
          f'ref v0={xref[2, 0]:.2f}')

    # (label, ego speed, u_prev) -- the standstill case is what the latched runs sit at.
    cases = [
        ('as-captured', float(x0[2]), data['u_prev']),
        ('v=3.0', 3.0, np.array([0.0, float(data['u_prev'][1])])),
        ('v=1.0', 1.0, np.array([0.0, float(data['u_prev'][1])])),
        ('v=0.0 standstill', 0.0, np.zeros(2)),
        ('v=0.0 standstill, u_prev=captured', 0.0, data['u_prev']),
    ]

    seeds = {
        # What _recover_from_failure() does today: stage 0 at the true state, stages
        # 1..N snapped onto the reference. The reference is anchored distance_tolerance
        # ahead and laterally offset, so stage 0 -> stage 1 is a metre-scale jump the
        # dynamics cannot make in one dt.
        'recovery(x0+xref)': lambda xi, h: (
            [xi] + [xref[:, i] for i in range(1, h + 1)], [np.zeros(2)] * h),
        # Dynamically consistent alternative: hold the current state over the horizon.
        'x0-tiled': lambda xi, h: ([xi] * (h + 1), [np.zeros(2)] * h),
        # Strictly dynamically consistent: roll the kinematic model forward from x0
        # under u=0, so the equality residual at the seed is zero by construction.
        'rollout(u=0)': lambda xi, h: (_rollout(xi, h, wheelbase, dt), [np.zeros(2)] * h),
        # What the solver was actually warm-started from when it failed live.
        'captured': lambda xi, h: (
            [data['seed_x'][:, i] for i in range(h + 1)],
            [data['seed_u'][:, i] for i in range(h)]),
    }

    for label, speed, u_prev in cases:
        for seed_name, seed_fn in seeds.items():
            if seed_name == 'captured' and data.get('seed_x', np.array([])).size == 0:
                continue
            if parsed.seed and seed_name != parsed.seed:
                continue
            adapter, horizon = build_adapter(data, config_file)   # fresh solver each case
            patch_recovery(adapter, parsed.recovery, wheelbase, dt)
            xi = x0.copy()
            xi[2] = speed
            xs, us = seed_fn(xi, horizon)
            for i in range(horizon + 1):
                adapter._controller.set(i, 'x', np.asarray(xs[i], dtype=float))
            for i in range(horizon):
                adapter._controller.set(i, 'u', np.asarray(us[i], dtype=float))
            line = []
            for _ in range(parsed.repeats):
                r = adapter.solve(xi, xref, u_prev)
                line.append(f'{r.status}{"" if r.is_optimal else "*"}')
            res = adapter._solver_stat('residuals').reshape(-1)
            print(f'  {label:24s} seed={seed_name:20s} statuses={" ".join(line)}  '
                  f'final residuals={np.array2string(res, precision=2)}')


if __name__ == '__main__':
    main()
