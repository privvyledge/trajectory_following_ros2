import types
import numpy as np
import casadi
from abc import ABC, abstractmethod


class KinematicMPCBase(ABC):
    """Shared constructor logic and utility methods for CasADi MPC solvers.

    Subclasses must implement:
        _build_vehicle_model  -- construct self.vehicle, self.model, self.ode, self.ode_function
        constraints_setup     -- return (constraints, lbg, ubg, lbx, ubx)
        setup_solver          -- return (solver, opt_variables, opt_params, opt_constraints)
        solve                 -- return sol_dict
    """

    def __init__(self, vehicle=None, horizon=15, sample_time=0.02, wheelbase=0.256,
                 nx=4, nu=2, x0=None, u0=None,
                 Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                 Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                 vel_bound=(-5.0, 5.0),
                 delta_bound=(-np.radians(23.0), np.radians(23.0)),
                 acc_bound=(-3.0, 3.0),
                 jerk_bound=(-1.5, 1.5),
                 delta_rate_bound=(-np.radians(352.9411764706), np.radians(352.9411764706)),
                 symbol_type='MX', warmstart=True,
                 solver_options=None, solver_type='nlp', solver='ipopt',
                 suppress_ipopt_output=True, normalize_yaw_error=True,
                 slack_weights_u_rate=(1e-6, 1e-6),
                 slack_scale_u_rate=(1.0, 1.0),
                 slack_upper_bound_u_rate=None,
                 slack_objective_is_quadratic=False,
                 code_gen_mode='jit',
                 num_obstacles=0, collision_avoidance_scheme='euclidean',
                 ego_radius=None,
                 slack_weights_obstacle_avoidance=None,
                 slack_upper_bound_obstacle_avoidance=None):

        # --- Scalar / matrix attributes ---
        self.horizon = horizon
        self.nx = nx
        self.nu = nu
        self.Q = casadi.diag(Q)
        self.R = casadi.diag(R)
        self.Qf = casadi.diag(Qf)
        self.Rd = casadi.diag(Rd)
        self.P_u_rate = (casadi.diag(slack_weights_u_rate)
                         if slack_weights_u_rate is not None
                         else casadi.DM.zeros(self.nu, self.nu))
        self.V_scale_u_rate = (casadi.diag(slack_scale_u_rate)
                               if slack_scale_u_rate is not None
                               else casadi.DM.eye(self.nu))
        if slack_upper_bound_u_rate is not None and isinstance(
                slack_upper_bound_u_rate, (list, tuple, np.ndarray)):
            if isinstance(slack_upper_bound_u_rate, np.ndarray):
                slack_upper_bound_u_rate = slack_upper_bound_u_rate.flatten().tolist()
            elif isinstance(slack_upper_bound_u_rate, tuple):
                slack_upper_bound_u_rate = list(slack_upper_bound_u_rate)
            for i in range(len(slack_upper_bound_u_rate)):
                if slack_upper_bound_u_rate[i] < 0:
                    slack_upper_bound_u_rate[i] = np.inf
        self.slack_upper_bound_u_rate = (casadi.DM(slack_upper_bound_u_rate)
                                         if slack_upper_bound_u_rate is not None
                                         else casadi.DM([np.inf, np.inf]))
        self.slack_objective_is_quadratic = slack_objective_is_quadratic
        self.warmstart = warmstart
        self.solver_options = solver_options
        self.solver_type = solver_type
        self.solver_ = solver
        self.code_gen_mode = code_gen_mode
        self.normalize_yaw_error = normalize_yaw_error

        self.Ts = sample_time
        self.T_predict = self.Ts * self.horizon
        self.wheelbase = wheelbase
        self.lf = wheelbase / 2
        self.lr = self.wheelbase - self.lf
        self.width = 0.192

        self.vehicle = vehicle
        self.symbol_type = symbol_type
        if self.symbol_type.lower() == 'sx':
            self.symbol_type = casadi.SX
        else:
            self.symbol_type = casadi.MX

        # --- Subclass builds vehicle model, sets self.vehicle / self.model / self.ode / self.ode_function ---
        self._build_vehicle_model(vehicle, nx, nu, x0,
                                  vel_bound, delta_bound, acc_bound,
                                  jerk_bound, delta_rate_bound)

        # --- Value variables (all solvers share the same layout) ---
        self.u_prev = None
        self.z_k = x0
        self.u_prev_value = u0
        self.z_k_value = x0

        self.x_ref = None
        self.y_ref = None
        self.v_ref = None
        self.psi_ref = None
        self.z_ref = np.zeros((self.nx, self.horizon + 1))
        self.u_ref = np.zeros((self.nu, self.horizon))
        self.z_ref_value = np.zeros((self.nx, self.horizon + 1))
        self.u_ref_value = np.zeros((self.nu, self.horizon))

        self.x_dv = None
        self.y_dv = None
        self.v_dv = None
        self.psi_dv = None
        self.z_dv = np.zeros((self.nx, self.horizon + 1))
        self.z_dv_value = np.zeros((self.nx, self.horizon + 1))
        self.z_pred_dv = None
        self.z_pred_value = None

        self.acc_dv = None
        self.delta_dv = None
        self.u_dv = np.zeros((self.nu, self.horizon))
        self.u_dv_value = np.zeros((self.nu, self.horizon))

        self.jerk_dv = None
        self.delta_rate_dv = None
        self.u_rate_dv = None
        self.u_rate_dv_value = np.zeros((self.nu, self.horizon))

        self.sl_acc_dv = None
        self.sl_delta_dv = None
        self.sl_dv = np.zeros((self.nu, self.horizon))
        self.sl_dv_value = np.zeros((self.nu, self.horizon))

        self.lam_x_value = None
        self.lam_g_value = None
        self.lam_p_value = None

        # --- Obstacle avoidance attributes (no-ops when num_obstacles == 0) ---
        self.n_obstacles = num_obstacles
        self.P_obstacle_avoidance = (casadi.diag(slack_weights_obstacle_avoidance)
                                     if slack_weights_obstacle_avoidance is not None
                                     else casadi.DM.zeros(self.n_obstacles, self.n_obstacles))
        self.ego_radius = None
        self.ego_radius_value = ego_radius
        self.obstacles = None
        self.obstacles_value = np.ones((3 * self.n_obstacles, horizon + 1)) * 1000.0
        self.obstacle_distances = None
        self.obstacle_distances_value = np.ones((self.n_obstacles, horizon + 1)) * np.inf
        self.safe_distance = 0.5
        self.collision_avoidance_scheme = collision_avoidance_scheme

        if self.n_obstacles > 0:
            self.obstacles_value[2, :] = 2.0  # radius of first obstacle placeholder

            if slack_weights_obstacle_avoidance is not None and collision_avoidance_scheme == 'cbf':
                raise NotImplementedError("The slack formulation with CBF isn't correct.")

            if slack_upper_bound_obstacle_avoidance is not None and isinstance(
                    slack_upper_bound_obstacle_avoidance, (list, tuple, np.ndarray)):
                if isinstance(slack_upper_bound_obstacle_avoidance, np.ndarray):
                    slack_upper_bound_obstacle_avoidance = slack_upper_bound_obstacle_avoidance.flatten().tolist()
                elif isinstance(slack_upper_bound_obstacle_avoidance, tuple):
                    slack_upper_bound_obstacle_avoidance = list(slack_upper_bound_obstacle_avoidance)
                for i in range(len(slack_upper_bound_obstacle_avoidance)):
                    if slack_upper_bound_obstacle_avoidance[i] < 0:
                        slack_upper_bound_obstacle_avoidance[i] = np.inf
            self.slack_upper_bound_obstacle_avoidance = (
                casadi.DM(slack_upper_bound_obstacle_avoidance)
                if slack_upper_bound_obstacle_avoidance is not None
                else casadi.DM([np.inf] * self.n_obstacles))
            self.sl_obs_dv = np.zeros((self.n_obstacles, self.horizon + 1))
            self.sl_obs_dv_value = np.zeros((self.n_obstacles, self.horizon + 1))

            if collision_avoidance_scheme == 'cbf':
                self.cbf = None
                self.h = None
                self.gamma = np.clip(0.7, 0.0, 1.0)

        # --- MPC problem formulation ---
        self.mpc, self.objective, self.constraints = self.initialize_mpc(
            horizon=self.horizon, nx=self.nx, nu=self.nu)

        self.constraints, self.lbg, self.ubg, self.lbx, self.ubx = self.constraints_setup(
            vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound,
            jerk_bound=jerk_bound, delta_rate_bound=delta_rate_bound, reset=False)

        self.u_rate_func = casadi.Function(
            "u_rate",
            [self.u_prev, self.u_dv], [self.u_rate_dv],
            ['u_prev_k', 'u_k'], ['u_rate_k'])

        if self.n_obstacles > 0:
            if not casadi.is_equal(self.P_obstacle_avoidance,
                                   casadi.DM.zeros(self.n_obstacles, self.n_obstacles)):
                self.obstacle_distances_func = casadi.Function(
                    "obstacle_distances",
                    [self.z_k, self.z_dv, self.obstacles, self.ego_radius, self.sl_obs_dv],
                    [self.obstacle_distances],
                    ['z_k', 'z_dv', 'obstacles', 'ego_radius', 'slack_obs'],
                    ['obstacle_distances'])
            else:
                self.obstacle_distances_func = casadi.Function(
                    "obstacle_distances",
                    [self.z_k, self.z_dv, self.obstacles, self.ego_radius],
                    [self.obstacle_distances],
                    ['z_k', 'z_dv', 'obstacles', 'ego_radius'],
                    ['obstacle_distances'])
            self.ego_radius_func = casadi.Function(
                "ego_radius", [self.ego_radius], [self.ego_radius],
                ['ego_radius_value'], ['ego_radius'])
            self.update_ego_radius(ego_radius)

        self.objective = self.objective_function_setup()

        if x0 is not None:
            if isinstance(x0, np.ndarray):
                x0 = x0.flatten().tolist()
            self.update_initial_condition(x0[0], x0[1], x0[2], x0[3])

        if u0 is None:
            self.update_previous_input(0., 0.)
        else:
            if isinstance(u0, np.ndarray):
                u0 = u0.flatten().tolist()
            self.update_previous_input(u0[0], u0[1])

        open_loop_trajectory = np.zeros((self.nx, self.horizon + 1))
        open_loop_trajectory[:, 0] = x0
        for k in range(self.horizon):
            delta, acc = self.u_prev_value.flatten().tolist()
            x0_ = self.ode_function(open_loop_trajectory[:, k], [delta, acc])
            open_loop_trajectory[:, k + 1] = x0_.full()[:, 0]
        self.update_reference(*open_loop_trajectory.tolist())

        self.solver, self.opt_variables, \
            self.opt_params, self.opt_constraints = self.setup_solver(
                cost=self.objective, constraints=self.constraints,
                suppress_output=suppress_ipopt_output,
                solver_options=self.solver_options,
                solver=solver, solver_type=solver_type)

        self.solution = None
        self.solution_dict = None

        for _ in range(5):
            self.solution = self.solve()

    @abstractmethod
    def _build_vehicle_model(self, vehicle, nx, nu, x0,
                             vel_bound, delta_bound, acc_bound,
                             jerk_bound, delta_rate_bound):
        """Set self.vehicle, self.model, self.ode, self.ode_function."""

    def initialize_mpc(self, horizon, nx, nu):
        mpc = types.SimpleNamespace()
        self.u_prev = self.symbol_type.sym('u_prev', nu)
        self.z_k = self.symbol_type.sym('z_k', nx)
        self.z_ref = self.symbol_type.sym('z_ref', nx, horizon + 1)
        self.u_ref = self.symbol_type.sym('u_ref', nu, horizon)
        self.z_dv = self.symbol_type.sym('z_dv', nx, horizon + 1)
        self.u_dv = self.symbol_type.sym('u_dv', nu, horizon)
        self.u_rate_dv = None
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            self.sl_dv = self.symbol_type.sym('sl_dv', nu, horizon)
        if self.n_obstacles > 0:
            if self.collision_avoidance_scheme in ['euclidean', 'cbf']:
                self.ego_radius = self.symbol_type.sym('ego_radius', 1)
                self.obstacles = self.symbol_type.sym(
                    'obstacles', 3 * self.n_obstacles, horizon + 1)
            if not casadi.is_equal(self.P_obstacle_avoidance,
                                   casadi.DM.zeros(self.n_obstacles, self.n_obstacles)):
                self.sl_obs_dv = self.symbol_type.sym(
                    'sl_obs_dv', self.n_obstacles, horizon + 1)
        objective = 0.0
        constraints = []
        return mpc, objective, constraints

    @abstractmethod
    def constraints_setup(self, vel_bound, delta_bound, acc_bound,
                          jerk_bound, delta_rate_bound, reset):
        pass

    def objective_function_setup(self):
        cost = 0
        u_prev = self.u_prev
        for k in range(self.horizon):
            if self.normalize_yaw_error:
                cost += self._quad_form(self.z_dv[0:3, k] - self.z_ref[0:3, k],
                                        self.Q[0:3, 0:3])
                cost += self._quad_form(
                    casadi.fmod(self.z_dv[3, k] - self.z_ref[3, k] + np.pi, 2 * np.pi) - np.pi,
                    self.Q[3, 3])
            else:
                cost += self._quad_form(self.z_dv[:, k] - self.z_ref[:, k], self.Q)

            if k < (self.horizon - 1):
                cost += self._quad_form(self.u_dv[:, k], self.R)
                cost += self._quad_form(self.u_dv[:, k] - u_prev, self.Rd)
                u_prev = self.u_dv[:, k]

                if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                    if self.slack_objective_is_quadratic:
                        cost += self._quad_form(self.sl_dv[:, k], self.P_u_rate)
                    else:
                        cost += casadi.mtimes(casadi.diag(self.P_u_rate).T, self.sl_dv[:, k])

                if (self.n_obstacles > 0
                        and not casadi.is_equal(
                            self.P_obstacle_avoidance,
                            casadi.DM.zeros(self.n_obstacles, self.n_obstacles))):
                    if self.slack_objective_is_quadratic:
                        cost += self._quad_form(self.sl_obs_dv[:, k], self.P_obstacle_avoidance)
                    else:
                        cost += casadi.mtimes(
                            casadi.diag(self.P_obstacle_avoidance).T, self.sl_obs_dv[:, k])

        cost += self._quad_form(
            self.z_dv[:, self.horizon] - self.z_ref[:, self.horizon], self.Qf)
        cost *= 0.5
        return cost

    @abstractmethod
    def setup_solver(self, cost, constraints, solver_type, solver_options,
                     solver, suppress_output):
        pass

    @abstractmethod
    def solve(self):
        pass

    # --- Shared utility methods ---

    def _quad_form(self, z, Q):
        return casadi.mtimes(z.T, casadi.mtimes(Q, z))

    def update_initial_condition(self, x0, y0, vel0, psi0):
        self.z_k_value[:, 0] = [x0, y0, vel0, psi0]

    def update_reference(self, x_ref, y_ref, v_ref, psi_ref, acc_ref=0.0, delta_ref=0.0):
        self.z_ref_value[0, :] = x_ref
        self.z_ref_value[1, :] = y_ref
        self.z_ref_value[2, :] = v_ref
        self.z_ref_value[3, :] = psi_ref
        self.u_ref_value[0, :] = acc_ref
        self.u_ref_value[1, :] = delta_ref

    def update_previous_input(self, acc_prev, delta_prev):
        self.u_prev_value[:, 0] = [acc_prev, delta_prev]

    def update_u_rate(self, u_prev_value, u_dv_value):
        self.u_rate_dv_value = self.u_rate_func(u_prev_value, u_dv_value).full()

    def update_obstacles_state(self, obstacles_state):
        self.obstacles_value = obstacles_state

    def update_obstacle_distances(self, z_k, z_dv, obstacles_state, ego_radius,
                                  slack_obs=None):
        if slack_obs is None:
            self.obstacle_distances_value = self.obstacle_distances_func(
                z_k, z_dv, obstacles_state, ego_radius).full()
        else:
            self.obstacle_distances_value = self.obstacle_distances_func(
                z_k, z_dv, obstacles_state, ego_radius, slack_obs).full()

    def update_ego_radius(self, ego_radius):
        self.ego_radius_value = self.ego_radius_func(ego_radius).full().item()

    def update(self, state=None, ref_traj=None, previous_input=None,
               warmstart_variables=None):
        self.update_initial_condition(*state)
        self.update_reference(*ref_traj)
        self.update_previous_input(*previous_input)

        if self.warmstart and len(warmstart_variables) > 0:
            self.z_dv_value = warmstart_variables['z_ws']
            self.u_dv_value = warmstart_variables['u_ws']
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                if warmstart_variables.get('sl_ws') is not None:
                    self.sl_dv_value = warmstart_variables['sl_ws']
            if (self.n_obstacles > 0
                    and not casadi.is_equal(
                        self.P_obstacle_avoidance,
                        casadi.DM.zeros(self.n_obstacles, self.n_obstacles))):
                if warmstart_variables.get('sl_obs_ws') is not None:
                    self.sl_obs_dv_value = warmstart_variables['sl_obs_ws']
            self.lam_x_value = warmstart_variables['lam_x']
            self.lam_g_value = warmstart_variables['lam_g']
            self.lam_p_value = warmstart_variables['lam_p']

    def set_weights(self, Q, R, Qf=None, Rd=None):
        self.Q = casadi.diag(Q)
        self.R = casadi.diag(R)
        if Qf is not None:
            self.Qf = casadi.diag(Qf)
        if Rd is not None:
            self.Rd = casadi.diag(Rd)

    def reset(self, z0=None):
        if z0 is None:
            self.z_dv_value = np.zeros((self.nx, self.horizon + 1))
        self.u_dv_value = np.zeros((self.nu, self.horizon))
        self.z_k_value = self.symbol_type.sym('z_k', self.nx)
        self.u_prev_value = np.zeros((self.nu, 1))
        self.u_rate_dv_value = np.zeros((self.nu, self.horizon))
        self.sl_dv_value = np.zeros((self.nu, self.horizon))
        self.solve()

    def run(self, z0, u0, zref, zk, u_prev):
        pass