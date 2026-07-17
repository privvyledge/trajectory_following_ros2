"""Replay an opt-in AcadosSolverAdapter first-failure dump against its built solver."""
import argparse
import os

import numpy as np

from trajectory_following_ros2.coupled_kinematic_acados import AcadosSolverAdapter


def _scalar(data, name, default=None):
    if name not in data or data[name].size == 0:
        return default
    return data[name].reshape(-1)[0].item()


def _seed_controller(controller, seed_x, seed_u, x0, xref, horizon):
    """Restore the captured primal seed, or a documented reference fallback."""
    captured = (seed_x.shape == (x0.size, horizon + 1)
                and seed_u.shape == (2, horizon))
    if not captured:
        seed_x = xref.copy()
        seed_x[:, 0] = x0
        seed_u = np.zeros((2, horizon))

    for stage in range(horizon + 1):
        controller.set(stage, 'x', seed_x[:, stage])
    for stage in range(horizon):
        controller.set(stage, 'u', seed_u[:, stage])
    return captured


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Replay an acados first-hard-failure .npz fixture.')
    parser.add_argument('dump', help='Path written by acados_failure_dump_file.')
    parser.add_argument(
        '--config', default='',
        help='Override the generated acados OCP JSON recorded in the dump.')
    parser.add_argument(
        '--attempts', type=int, default=2,
        help='Solve attempts to run; later attempts exercise adapter recovery (default: 2).')
    parsed = parser.parse_args(args)
    if parsed.attempts < 1:
        parser.error('--attempts must be >= 1')

    with np.load(os.path.expanduser(parsed.dump), allow_pickle=False) as loaded:
        data = {name: loaded[name].copy() for name in loaded.files}

    config_file = parsed.config or str(_scalar(data, 'solver_config_file', ''))
    config_file = os.path.abspath(os.path.expanduser(config_file)) if config_file else ''
    if not config_file or not os.path.isfile(config_file):
        parser.error(
            'generated solver config JSON is missing; pass its current path with --config')

    try:
        from acados_template import AcadosOcpSolver
    except ImportError as exc:
        parser.error(f'acados_template is unavailable in this Python environment: {exc}')

    controller = AcadosOcpSolver.create_cython_solver(config_file)
    horizon = int(_scalar(data, 'horizon'))
    has_weights = bool(_scalar(data, 'has_weight_params', 0))
    rate_max = data.get('rate_max', np.array([]))
    rate_max = rate_max if rate_max.size else None

    weight_args = {}
    if has_weights:
        weight_args = {
            'Q': np.diag(data['Q_diag']),
            'R': np.diag(data['R_diag']),
            'Qe': np.diag(data['Qe_diag']),
            'Rd': np.diag(data['Rd_diag']),
        }

    adapter = AcadosSolverAdapter(
        controller,
        horizon=horizon,
        wheelbase=float(_scalar(data, 'wheelbase')),
        stage_cost_type=str(_scalar(data, 'stage_cost_type', 'EXTERNAL')),
        terminal_cost_type=str(_scalar(data, 'terminal_cost_type', 'EXTERNAL')),
        num_obstacles=int(_scalar(data, 'num_obstacles', 0)),
        ego_radius=float(_scalar(data, 'ego_radius', 1.0)),
        has_weight_params=has_weights,
        dt=float(_scalar(data, 'dt', 0.05)),
        u_min=data['u_min'],
        u_max=data['u_max'],
        rate_max=rate_max,
        solver_config_file=config_file,
        **weight_args,
    )
    obstacle_states = data.get('obstacle_states', np.array([]))
    if obstacle_states.size:
        adapter.update_obstacles(obstacle_states)

    x0 = data['x0']
    xref = data['xref']
    u_prev = data['u_prev']
    captured_seed = _seed_controller(
        controller,
        data.get('seed_x', np.array([])),
        data.get('seed_u', np.array([])),
        x0, xref, horizon)
    print(f'primal_seed={"captured" if captured_seed else "reference fallback"}')

    for attempt in range(1, parsed.attempts + 1):
        result = adapter.solve(x0, xref, u_prev)
        qp_stat = adapter._solver_stat('qp_stat').reshape(-1).tolist()
        residuals = adapter._solver_stat('residuals').reshape(-1).tolist()
        print(
            f'attempt={attempt} status={result.status} optimal={result.is_optimal} '
            f'finite={np.isfinite(result.x_sequence).all()} '
            f'qp_stat={qp_stat} residuals={residuals}')


if __name__ == '__main__':
    main()
