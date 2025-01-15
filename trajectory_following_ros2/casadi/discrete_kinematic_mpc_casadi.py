"""
See:
    To parallelize casadi:
        * https://web.casadi.org/docs/#for-loop-equivalents
        * https://github.com/nirajbasnet/Nonlinear_MPCC_for_autonomous_racing/blob/master/nonlinear_mpc_casadi/scripts/Nonlinear_MPC.py#L157

Notes:
    * QP solvers work when slack is added to rate inputs. Fails without slack or RD

Todo:
    * set solver [done]
    * simulate [done]
    * setup cost and constraints using vertcat [done]
    * set constraints as lbx, lbg, ubx, ubg instead [done]
    * add terminal state cost [done]
    * add rate change cost [done]
    * add method to set weights [done]
    * add method to reset, i.e set all back to zero, e.g u_prev, z_k, z_dv, z_ref, [Done]
    * add solver options [Done]
    * add model generation/loading, i.e code generation or compilation
        * Do-mpc example:
    * create run method to first update, (optionally) set weights, (optionally) reset, solve
    * setup rate constraints [done]
    * setup slack on rate constraints
    * rename variables
    * replace for loops with map
    * implement my discretizer

"""
import types
import time
import numpy as np
from scipy import linalg
import casadi

from trajectory_following_ros2.casadi.kinematic_bicycle_model import KinematicBicycleModel


class DiscreteKinematicMPCCasadi(object):
    """docstring for ClassName"""

    def __init__(self, vehicle=None, horizon=15, sample_time=0.02, wheelbase=0.256,
                 nx=4, nu=2, x0=None, u0=None,
                 Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                 Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                 vel_bound=(-5.0, 5.0), delta_bound=(-23.0, 23.0), acc_bound=(-3.0, 3.0),
                 jerk_bound=(-1.5, 1.5), delta_rate_bound=(-352.9411764706, 352.9411764706),
                 symbol_type='MX', warmstart=True,
                 solver_options=None, solver_type='nlp', solver='ipopt', suppress_ipopt_output=True,
                 normalize_yaw_error=True,
                 slack_weights_u_rate=(1e-6, 1e-6),  # (1e-6, 1e-6)
                 slack_scale_u_rate=(1.0, 1.0),
                 slack_upper_bound_u_rate=None, #(1., np.radians(30.0)), (np.inf, np.inf). Set to 0 to disable slack
                 slack_objective_is_quadratic=False,
                 code_gen_mode='jit',
                 num_obstacles=1, collision_avoidance_scheme='cbf',
                 ego_radius=None,
                 slack_weights_obstacle_avoidance=None, slack_upper_bound_obstacle_avoidance=None):
        """Constructor for KinematicMPCCasadi"""
        self.horizon = horizon
        self.nx = nx
        self.nu = nu
        self.Q = casadi.diag(Q)
        self.R = casadi.diag(R)
        self.Qf = casadi.diag(Qf)
        self.Rd = casadi.diag(Rd)
        self.P_u_rate = casadi.diag(slack_weights_u_rate) if slack_weights_u_rate is not None else casadi.DM.zeros(self.nu, self.nu)  # >= 0,should be small, for soft constraints and large for hard constraints. 0 to disable
        self.V_scale_u_rate = casadi.diag(slack_scale_u_rate) if slack_scale_u_rate is not None else casadi.DM.eye(self.nu)  # should be identity in most cases
        # set the positive slack upper bound
        if slack_upper_bound_u_rate is not None and isinstance(slack_upper_bound_u_rate, (list, tuple, np.ndarray)):
            if isinstance(slack_upper_bound_u_rate, np.ndarray):
                slack_upper_bound_u_rate = slack_upper_bound_u_rate.flatten().tolist()
            elif isinstance(slack_upper_bound_u_rate, tuple):
                slack_upper_bound_u_rate = list(slack_upper_bound_u_rate)
            for i in range(len(slack_upper_bound_u_rate)):
                if slack_upper_bound_u_rate[i] < 0:
                    slack_upper_bound_u_rate[i] = np.inf
        self.slack_upper_bound_u_rate = casadi.DM(slack_upper_bound_u_rate) if slack_upper_bound_u_rate is not None else casadi.DM([np.inf, np.inf])  # to put limits (upper bound) on the slack variables. Must be > 0. 0 implies None
        self.slack_objective_is_quadratic = slack_objective_is_quadratic  # if True, use quadratic slack objective else linear
        self.warmstart = warmstart
        self.solver_options = solver_options
        self.solver_type = solver_type
        self.solver_ = solver
        self.code_gen_mode = code_gen_mode  # '', 'jit', 'external'
        self.normalize_yaw_error = normalize_yaw_error  # normalizes the angle in the range [-pi, pi)

        self.Ts = sample_time
        self.T_predict = self.Ts * self.horizon  # prediction horizon, i.e T_predict / horizon = dt
        self.wheelbase = wheelbase
        self.lf = wheelbase / 2
        self.lr = self.wheelbase - self.lf
        self.width = 0.192

        self.vehicle = vehicle
        self.symbol_type = symbol_type
        if self.symbol_type.lower() == 'SX'.lower():
            self.symbol_type = casadi.SX
        else:
            self.symbol_type = casadi.MX

        if vehicle is not None:
            self.vehicle = vehicle
            self.model = self.vehicle.model
        else:
            self.vehicle = KinematicBicycleModel(nx=nx, nu=nu, x0=x0,
                                                 vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound,
                                                 jerk_bound=jerk_bound,
                                                 delta_rate_bound=delta_rate_bound,
                                                 vehicle_parameters={"wheelbase": self.wheelbase}, sample_time=self.Ts,
                                                 model_type='kinematic', model_name='vehicle_kinematic_model',
                                                 symbol_type=symbol_type,
                                                 discretization_method='cvodes', discrete=True)
            self.model = self.vehicle.model

        self.ode = self.model.f_expl_expr
        self.ode = casadi.substitute(self.ode,
                                     casadi.vertcat(self.model.params.wheelbase, self.model.params.dt),
                                     casadi.vertcat(self.wheelbase, self.Ts))
        self.ode_function = self.vehicle.create_ode_function(self.ode,
                                                             self.model.x, self.model.u,
                                                             function_name='discrete_ltv_ode')
        self.u_prev = None  # previous input: [u_{acc, -1}, u_{df, -1}]
        self.z_k = x0  # current state:  [x_0, y_0, psi_0, v_0]
        self.u_prev_value = u0  # previous input: [u_{acc, -1}, u_{df, -1}]
        self.z_k_value = x0  # current state:  [x_0, y_0, psi_0, v_0]

        self.x_ref = None
        self.y_ref = None
        self.v_ref = None
        self.psi_ref = None
        self.z_ref = np.zeros((self.nx, self.horizon + 1))  # reference trajectory ([x, y, vel, psi], horizon)
        self.u_ref = np.zeros((self.nu, self.horizon))  # reference input ([acc, delta], horizon)
        self.z_ref_value = np.zeros((self.nx, self.horizon + 1))  # reference trajectory ([x, y, vel, psi], horizon)
        self.u_ref_value = np.zeros((self.nu, self.horizon))  # reference input ([acc, delta], horizon)

        self.x_dv = None
        self.y_dv = None
        self.v_dv = None
        self.psi_dv = None
        self.z_dv = np.zeros((self.nx, self.horizon + 1))  # actual/predicted/openloop trajectory
        self.z_dv_value = np.zeros((self.nx, self.horizon + 1))
        self.z_pred_dv = None  # can be extracted from the constraints after solving
        self.z_pred_value = None  # can be extracted from the constraints after solving

        self.acc_dv = None
        self.delta_dv = None
        self.u_dv = np.zeros((self.nu, self.horizon))  # actual/predicted/openloop inputs
        self.u_dv_value = np.zeros((self.nu, self.horizon))

        self.jerk_dv = None
        self.delta_rate_dv = None
        self.u_rate_dv = None  # will be created in loop while defining constraints/objective
        self.u_rate_dv_value = np.zeros((self.nu, self.horizon))

        self.sl_acc_dv = None
        self.sl_delta_dv = None
        self.sl_dv = np.zeros((self.nu, self.horizon))  # slack variables for input rates
        self.sl_dv_value = np.zeros((self.nu, self.horizon))

        self.lam_x_value = None
        self.lam_g_value = None
        self.lam_p_value = None

        self.n_obstacles = num_obstacles
        self.P_obstacle_avoidance = casadi.diag(
            slack_weights_obstacle_avoidance) if slack_weights_obstacle_avoidance is not None else casadi.DM.zeros(
            self.n_obstacles,
            self.n_obstacles)  # >= 0,should be small, for soft constraints and large for hard constraints. 0 to disable
        self.ego_radius = None
        self.ego_radius_value = ego_radius
        self.obstacles = None
        self.obstacles_value = np.ones((3 * self.n_obstacles, horizon + 1)) * 1000.0  # setting to inf leads to infeasibilities
        if self.n_obstacles > 0:
            self.obstacles_value[2, :] = 2.0  # modify the radius

            if slack_weights_obstacle_avoidance is not None and collision_avoidance_scheme == 'cbf':
                raise NotImplementedError("The slack formulation with CBF isn't correct.")

            # set the positive slack upper bound
            if slack_upper_bound_obstacle_avoidance is not None and isinstance(slack_upper_bound_obstacle_avoidance, (list, tuple, np.ndarray)):
                if isinstance(slack_upper_bound_obstacle_avoidance, np.ndarray):
                    slack_upper_bound_obstacle_avoidance = slack_upper_bound_obstacle_avoidance.flatten().tolist()
                elif isinstance(slack_upper_bound_obstacle_avoidance, tuple):
                    slack_upper_bound_obstacle_avoidance = list(slack_upper_bound_obstacle_avoidance)
                for i in range(len(slack_upper_bound_obstacle_avoidance)):
                    if slack_upper_bound_obstacle_avoidance[i] < 0:
                        slack_upper_bound_obstacle_avoidance[i] = np.inf

            self.slack_upper_bound_obstacle_avoidance = casadi.DM(
                slack_upper_bound_obstacle_avoidance) if slack_upper_bound_obstacle_avoidance is not None else casadi.DM(
                [np.inf] * self.n_obstacles)  # to put limits (upper bound) on the slack variables. Must be > 0. 0 implies None
            self.sl_obs_dv = np.zeros((self.n_obstacles, self.horizon + 1))  # slack variables for input rates
            self.sl_obs_dv_value = np.zeros((self.n_obstacles, self.horizon + 1))

        self.obstacle_distances = None
        self.obstacle_distances_value = np.ones((self.n_obstacles, horizon + 1)) * np.inf
        self.safe_distance = 0.5  # (0.5) safe distance for obstacle avoidance.
        self.collision_avoidance_scheme = collision_avoidance_scheme  # euclidean, CBF
        if self.collision_avoidance_scheme == 'cbf':
            self.cbf = None
            self.h = None
            self.gamma = np.clip(0.7, 0.0, 1.0)

        self.mpc, self.objective, self.constraints = self.initialize_mpc(horizon=self.horizon, nx=self.nx, nu=self.nu)

        # define the objective function and constraints.
        self.constraints, self.lbg, self.ubg, self.lbx, self.ubx = self.constraints_setup(vel_bound=vel_bound,
                                                                                          delta_bound=delta_bound,
                                                                                          acc_bound=acc_bound,
                                                                                          jerk_bound=jerk_bound,
                                                                                          delta_rate_bound=delta_rate_bound,
                                                                                          reset=False)
        self.u_rate_func = casadi.Function("u_rate",
                                           [self.u_prev, self.u_dv], [self.u_rate_dv],
                                           # replace self.u_rate_dv with self.u_rate_dv_value to avoid allow_free
                                           ['u_prev_k', 'u_k'], [
                                               'u_rate_k'])  # , {'allow_free': True} allow_free argument required for MX functions

        if self.n_obstacles > 0:
            if not casadi.is_equal(self.P_obstacle_avoidance, casadi.DM.zeros((self.n_obstacles, self.n_obstacles))):
                self.obstacle_distances_func = casadi.Function("obstacle_distances",
                                                               [self.z_k, self.z_dv, self.obstacles, self.ego_radius, self.sl_obs_dv], [self.obstacle_distances],
                                                               ['z_k', 'z_dv', 'obstacles', 'ego_radius', 'slack_obs'], ['obstacle_distances'])
            else:
                self.obstacle_distances_func = casadi.Function("obstacle_distances",
                                                               [self.z_k, self.z_dv, self.obstacles, self.ego_radius],
                                                               [self.obstacle_distances],
                                                               ['z_k', 'z_dv', 'obstacles', 'ego_radius'],
                                                               ['obstacle_distances'])
            self.ego_radius_func = casadi.Function("ego_radius",
                                                   [self.ego_radius], [self.ego_radius],
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

        # update the open-loop reference trajectory based on the initial conditions
        # provide time-varing parameters: setpoints/reference.
        open_loop_trajectory = np.zeros((self.nx, self.horizon + 1))
        open_loop_trajectory[:, 0] = x0
        for k in range(self.horizon):
            # vel, psi = open_loop_trajectory[2, k], open_loop_trajectory[3, k]
            delta, acc = self.u_prev_value.flatten().tolist()
            x0_ = self.ode_function(open_loop_trajectory[:, k], [delta, acc])
            open_loop_trajectory[:, k + 1] = x0_.full()[:, 0]
        self.update_reference(*open_loop_trajectory.tolist())

        self.solver, self.opt_variables, \
            self.opt_params, self.opt_constraints = self.setup_solver(cost=self.objective, constraints=self.constraints,
                                                                      suppress_output=suppress_ipopt_output,
                                                                      solver_options=self.solver_options,
                                                                      solver_=self.solver_,
                                                                      solver_type=self.solver_type)

        num_tries = 5  # 1, 5, 10. todo: expose as a parameter and separate function
        for _ in range(num_tries):
            # a kind of hacky way to fix not finding a solution the first time the solver is called
            self.solution = self.solve()

    def initialize_mpc(self, horizon, nx, nu):
        mpc = types.SimpleNamespace()
        # Parameters
        self.u_prev = self.symbol_type.sym('u_prev', nu)
        self.z_k = self.symbol_type.sym('z_k', nx)

        # Reference trajectory
        self.z_ref = self.symbol_type.sym('z_ref', nx, horizon + 1)
        self.u_ref = self.symbol_type.sym('u_ref', nu, horizon)

        # Actual/Predicted/Open-Loop Trajectory
        self.z_dv = self.symbol_type.sym('z_dv', nx, horizon + 1)

        # Control inputs
        self.u_dv = self.symbol_type.sym('u_dv', nu, horizon)

        # Control input rate. Will be created in loop while defining constraints/objective
        self.u_rate_dv = None  # self.symbol_type.sym('u_rate_dv', nu, horizon)

        # Slack variables
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            self.sl_dv = self.symbol_type.sym('sl_dv', nu, horizon)

        # other non-linear constraints, e.g obstacle avoidance
        if self.n_obstacles > 0:
            if self.collision_avoidance_scheme in ['euclidean', 'cbf']:
                self.ego_radius = self.symbol_type.sym('ego_radius', 1)
                # self.obstacle_distances = self.symbol_type.sym('obstacle_distances', self.n_obstacles, horizon + 1)  # will be updated in constraint function
                self.obstacles = self.symbol_type.sym('obstacles', 3 * self.n_obstacles, horizon + 1)  # x, y, radius

            # Slack variables for obstacle avoidance
            if not casadi.is_equal(self.P_obstacle_avoidance, casadi.DM.zeros((self.n_obstacles, self.n_obstacles))):
                self.sl_obs_dv = self.symbol_type.sym('sl_obs_dv', self.n_obstacles, horizon + 1)

        objective = 0.0
        constraints = []
        # todo: remove objective and constraints since they're not being used
        return mpc, objective, constraints

    def constraints_setup(
            self, vel_bound=None, delta_bound=None, acc_bound=None,
            jerk_bound=None, delta_rate_bound=None, degrees=True, reset=False
    ):
        """
        G constraints (lbg, ubg): dynamic constraints in the form of an equation/function that change over time
            Examples -> zdot = f(x, u)
                        z_[k+1] = z_k + f(x, u) * dt
                        z_[k+1] = Az_k + Bu_k
        X (box) constraints (lbx, ubx): static constraints on [x, u]:
            Examples -> min <= variable <= max

        To add input rate constraints, augment the dynamics.
            See: https://forces.embotech.com/Documentation/examples/high_level_rate_constraints/index.html
        :param vel_bound:
        :param delta_bound:
        :param acc_bound:
        :param jerk_bound:
        :param delta_rate_bound:
        :param reset:
        :return:
        """
        if delta_bound is None:
            delta_bound = [-23.0, 23.0]

        if vel_bound is None:
            vel_bound = [-10.0, 10.0]

        if acc_bound is None:
            acc_bound = [-3.0, 3.0]

        if jerk_bound is None:
            jerk_bound = [-1.5, 1.5]

        if delta_rate_bound is None:
            delta_rate_bound = [-352.9411764706, 352.9411764706]

        if degrees:
            delta_bound = np.radians(delta_bound).tolist()
            delta_rate_bound = np.radians(delta_rate_bound).tolist()

        u_min = np.array([[acc_bound[0]], [delta_bound[0]]])
        u_max = np.array([[acc_bound[1]], [delta_bound[1]]])

        u_dot_min = np.array([[jerk_bound[0]], [delta_rate_bound[0]]])
        u_dot_max = np.array([[jerk_bound[1]], [delta_rate_bound[1]]])

        constraints = []  # todo: initialize the appropriate size
        '''
        Set lower bound of inequality constraint to zero to force: 
            1. n*N state dynamics
            2. n terminal contraints and
            3. CVX hull constraint

            [0] * (self.nx * (self.horizon + 1)) + [0] * self.nx 
        '''
        # lbg = [0] * (self.nx * (self.horizon + 1))  # + [0] * self.nx + [0]  # 0.0
        # ubg = [0] * (self.nx * (self.horizon + 1))  # + [0] * self.nx + [0]  # 0.0

        lbg = casadi.DM.zeros((self.nx * (self.horizon + 1)), 1)  # + [0] * self.nx + [0]  # 0.0
        ubg = casadi.DM.zeros((self.nx * (self.horizon + 1)), 1)  # + [0] * self.nx + [0]  # 0.0

        # initial state constraints
        constraints.append(self.z_dv[:, 0] - self.z_k)

        # # todo: add different integrators (https://github.com/casadi/casadi/blob/12fa60f676716ae09aa2abe34c1c9b9cf426e68a/docs/examples/python/direct_multiple_shooting.py#L42)
        # k1 = self.f(st, con)
        # if self.INTEGRATION_MODE == "Euler":
        #     st_next_euler = st + (self.dT * k1)
        # elif self.INTEGRATION_MODE == "RK3":
        #     k2 = self.f(st + self.dT / 2 * k1, con)
        #     k3 = self.f(st + self.dT * (2 * k2 - k1), con)
        #     st_next_euler = st + self.dT / 6 * (k1 + 4 * k2 + k3)
        # elif self.INTEGRATION_MODE == "RK4":
        #     k2 = self.f(st + self.dT / 2 * k1, con)
        #     k3 = self.f(st + self.dT / 2 * k2, con)
        #     k4 = self.f(st + self.dT * k3, con)
        #     st_next_euler = st + self.dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        # dt = T / N  # length of a control interval
        # for k in range(N):  # loop over control intervals
        #     # Runge-Kutta 4 integration
        #     k1 = f(X[:, k], U[:, k])
        #     k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
        #     k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
        #     k4 = f(X[:, k] + dt * k3, U[:, k])
        #     x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        z_pred_list = [self.z_k]
        for k in range(self.horizon):
            # todo: create a new symbolic state for x_next and update it at each timestep using my integrator function
            # state model/dynamics constraints.  a=b or a-b
            # ode_symbolic = self.ode_function(self.z_dv[:, k], self.u_dv[:, k])
            # z_dt = ode_symbolic * self.Ts

            z_next = self.ode_function(self.z_dv[:, k], self.u_dv[:, k])
            z_pred_list.append(z_next)

            # A = self.symbol_type.eye(self.nx) + self.Ts * self.symbol_type.jacobian(ode_symbolic, self.z_dv[:, k])
            # B = self.Ts * self.symbol_type.jacobian(ode_symbolic, self.u_dv[:, k])
            # z_next = A @ self.z_dv[:, k] + B @ self.u_dv[:, k]

            dynamics = self.z_dv[:, k + 1] - z_next  # (self.z_dv[:, k + 1] == z_next)
            constraints.append(dynamics)  # todo: use casadi map or fold

        constraints = casadi.vertcat(*constraints)
        z_pred_list = casadi.horzcat(*z_pred_list)
        self.z_pred_dv = z_pred_list

        # Input Rate Bound Constraints (lbx)
        u_prev = self.u_prev
        u_dot_list = []
        slack_flag = int(not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))))
        # u_dot = self.u_dv[:, 0] - self.u_prev
        # constraints.append(u_dot >= (u_dot_min * self.Ts - self.sl_dv[:, i + 1]))
        # constraints.append(u_dot <= (u_dot_max * self.Ts + self.sl_dv[:, i + 1]))
        for k in range(self.horizon):
            u_dot = self.u_dv[:, k] - u_prev  # u_dot = self.u_dv[:, k + 1] - self.u_dv[:, k]
            """
            The formulation below, i.e LBG - slack <= constraints <= UBG + slack is valid. 
            However, casadi requires LBG and UBG to only include numbers. 
            Therefore, we reformulate with an equivalent expression:
               i. LBG <= u_dot + slack <= np.inf (i.e, UBG), and
               ii. -UBG <= -u_dot + slack <= np.inf (or -np.inf (i.e, LBG) <= u_dot - slack <= UBG)
    
            constraints = casadi.vertcat(constraints, u_dot)
            lbg = casadi.vertcat(lbg, (u_dot_min * self.Ts) - (slack_flag * casadi.mtimes(self.V_scale_u_rate, self.sl_dv[:, k])))
            ubg = casadi.vertcat(ubg, (u_dot_max * self.Ts) + (slack_flag * casadi.mtimes(self.V_scale_u_rate, self.sl_dv[:, k])))
            """
            # i.
            constraints = casadi.vertcat(
                constraints,
                u_dot + (slack_flag * casadi.mtimes(self.V_scale_u_rate, self.sl_dv[:, k])))
            lbg = casadi.vertcat(lbg, u_dot_min * self.Ts)
            ubg = casadi.vertcat(ubg, casadi.inf * casadi.DM.ones((self.nu, 1)) * self.Ts)

            # ii.
            constraints = casadi.vertcat(
                constraints,
                -u_dot + (slack_flag * casadi.mtimes(self.V_scale_u_rate, self.sl_dv[:, k])))  # or u_dot - ...
            lbg = casadi.vertcat(lbg, -u_dot_max * self.Ts)  # or -casadi.inf * casadi.DM.ones((self.nu, 1))
            ubg = casadi.vertcat(ubg, casadi.inf * casadi.DM.ones((self.nu, 1)) * self.Ts)  # or  u_dot_max * self.Ts

            u_prev = self.u_dv[:, k]
            u_dot_list.append(u_dot)

        u_dot_list = casadi.horzcat(*u_dot_list)  # nu x horizon
        self.u_rate_dv = u_dot_list

        # Other Constraints.
        # e.g. things like collision avoidance or lateral acceleration bounds could go here.
        # add obstacle constraints
        if self.n_obstacles > 0:
            distance_expression_list = []

            if self.collision_avoidance_scheme == 'cbf':
                cbf_list = []  # same
                h_k_list = []  # same

            slack_obs_flag = int(not casadi.is_equal(
                self.P_obstacle_avoidance, casadi.DM.zeros((self.n_obstacles, self.n_obstacles))))

            for k in range(self.horizon):
                ego_pose = self.z_dv[0:3, k]  # get ego position (x, y). self.z_dv[0:3, k], z_pred_list[0:3, k]
                ego_pose_plus_1 = self.z_dv[0:3, k + 1]  # k + 1. self.z_dv[0:3, k + 1], z_pred_list[0:3, k + 1]
                for i in range(self.n_obstacles):
                    obstacle_state = self.obstacles[3 * i:3 * i + 3, k]  # get obstacle state (x, y, radius)
                    obstacle_state_plus_1 = self.obstacles[3 * i:3 * i + 3, k + 1]
                    distance_expression = casadi.sumsqr(
                        ego_pose[0:2] - obstacle_state[0:2]
                    )  # avoid norm_2 or sqrt since they have undefined derivatives at 0. casadi.sqrt(casadi.sumsqr(ego_pose[0:2] - obstacle_state[0:2])) is the same as casadi.norm_2(ego_pose[0:2] - obstacle_state[0:2])
                    distance_expression -= (self.ego_radius + obstacle_state[
                        2] + self.safe_distance - (slack_obs_flag * self.sl_obs_dv[:, k])) ** 2  # to avoid symbolic values in the LBG constraint. Square to avoid using sqrt in distance_expression above

                    if self.collision_avoidance_scheme == "euclidean":
                        distance_expression_list.append(distance_expression)
                        lbg = casadi.vertcat(lbg, np.zeros((self.n_obstacles, 1)))  # 0
                        ubg = casadi.vertcat(ubg, np.ones((self.n_obstacles, 1)) * np.inf)  # np.inf
                    elif self.collision_avoidance_scheme == "cbf":
                        h_k = distance_expression  # h(k)
                        h_k_plus_1 = casadi.sumsqr(ego_pose_plus_1[0:2] - obstacle_state_plus_1[0:2]) - (self.ego_radius + obstacle_state_plus_1[2] + self.safe_distance - (slack_obs_flag * self.sl_obs_dv[:, k + 1])) ** 2  # h(k+1)
                        delta_h_xk_uk = h_k_plus_1 - h_k
                        distance_expression_list.append(delta_h_xk_uk + (self.gamma * h_k))
                        lbg = casadi.vertcat(
                            lbg, np.zeros((self.n_obstacles, 1)))
                        ubg = casadi.vertcat(
                            ubg, np.ones((self.n_obstacles, 1)) * np.inf)

                        # h_k_list.append(h_k)
                    else:
                        distance_expression = casadi.DM(0)
                        # todo: add other collision avoidance schemes
            self.obstacle_distances = casadi.vertcat(*distance_expression_list)
            constraints = casadi.vertcat(
                constraints,
                self.obstacle_distances
            )

        # lbx = [-casadi.inf, -casadi.inf, vel_bound[0], -2 * casadi.pi] * (self.horizon + 1)  # state constraints
        # ubx = [casadi.inf, casadi.inf, vel_bound[1], 2 * casadi.pi] * (self.horizon + 1)  # state constraints
        #
        # lbx.extend([acc_bound[0], delta_bound[0]] * self.horizon)  # input constraints
        # ubx.extend([acc_bound[1], delta_bound[1]] * self.horizon)  # input constraints

        lbx = casadi.DM.zeros((self.nx * (self.horizon + 1) + self.nu * self.horizon, 1))
        ubx = casadi.DM.zeros((self.nx * (self.horizon + 1) + self.nu * self.horizon, 1))

        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            lbx = casadi.vertcat(lbx, casadi.DM.zeros((self.nu * self.horizon, 1)))
            ubx = casadi.vertcat(ubx, casadi.DM.zeros((self.nu * self.horizon, 1)))

        if not casadi.is_equal(self.P_obstacle_avoidance, casadi.DM.zeros((self.n_obstacles, self.n_obstacles))):
            lbx = casadi.vertcat(lbx, casadi.DM.zeros((self.n_obstacles * (self.horizon + 1), 1)))
            ubx = casadi.vertcat(ubx, casadi.reshape(casadi.mtimes(casadi.DM.ones((self.horizon + 1)), self.slack_upper_bound_obstacle_avoidance.T), self.n_obstacles * (self.horizon + 1), 1))

        # state constraints (self.v_dv, vel_bound[1]). todo: add variable constraint for each shooting node, e.g [-casadi.inf] * (self.horizon + 1)
        lbx[0: self.nx * (self.horizon + 1): self.nx] = -casadi.inf  # X lower bound
        lbx[1: self.nx * (self.horizon + 1): self.nx] = -casadi.inf  # Y lower bound
        lbx[2: self.nx * (self.horizon + 1): self.nx] = vel_bound[0]  # vel lower bound
        lbx[3: self.nx * (self.horizon + 1): self.nx] = -2 * casadi.pi  # psi lower bound

        ubx[0: self.nx * (self.horizon + 1): self.nx] = casadi.inf  # X upper bound
        ubx[1: self.nx * (self.horizon + 1): self.nx] = casadi.inf  # Y upper bound
        ubx[2: self.nx * (self.horizon + 1): self.nx] = vel_bound[1]  # vel upper bound
        ubx[3: self.nx * (self.horizon + 1): self.nx] = 2 * casadi.pi  # psi upper bound

        # controls
        lbx[self.nx * (self.horizon + 1):(self.nx * (self.horizon + 1) + (self.nu * self.horizon)):self.nu] = acc_bound[
            0]  # acc lower bound
        lbx[self.nx * (self.horizon + 1) + 1:(
                    self.nx * (self.horizon + 1) + (self.nu * self.horizon)):self.nu] = delta_bound[0]  # delta lower bound

        ubx[self.nx * (self.horizon + 1):(self.nx * (self.horizon + 1) + (self.nu * self.horizon)):self.nu] = acc_bound[
            1]  # acc lower bound
        ubx[self.nx * (self.horizon + 1) + 1:(
                    self.nx * (self.horizon + 1) + (self.nu * self.horizon)):self.nu] = delta_bound[1]  # delta lower bound

        # slack constraints (see https://forces.embotech.com/Documentation/examples/high_level_soft_constraints/index.html)
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            lbx[self.nx * (self.horizon + 1) + (self.nu * self.horizon):(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (
                            self.nu * self.horizon)):self.nu] = 0.0  # todo: make configurable
            lbx[self.nx * (self.horizon + 1) + (self.nu * self.horizon) + 1:(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (
                            self.nu * self.horizon)):self.nu] = 0.0  # todo: make configurable

            ubx[self.nx * (self.horizon + 1) + (self.nu * self.horizon):(
                    self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):self.nu] = \
            self.slack_upper_bound_u_rate[0]
            ubx[self.nx * (self.horizon + 1) + (self.nu * self.horizon) + 1:(
                    self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):self.nu] = \
            self.slack_upper_bound_u_rate[1]

        # # To verify for debugging purposes (should be commented out)
        # opt_to_state_input_dict = {
        #     'x_0': 'x', 'x_1': 'y', 'x_2': 'vel', 'x_3': 'psi',
        #     'u_0': 'acc', 'u_1': 'delta'
        # }
        # for index in range(constraints.size1()):
        #     print(f"{lbg[index]} <= {constraints[index]} <= {ubg[index]}")
        # print("\n")
        #
        # # print state (z) constraints. todo: fix
        # for index in range(0, self.nx * (self.horizon + 1), self.nx):
        #     expression = ""
        #     for state in range(self.nx):
        #         key = f'x_{state}'
        #         expression += f"{lbx[index + state]} <= {opt_to_state_input_dict[key]} <= {ubx[index + state]}\n"
        #     print(expression)
        #
        # # print input (u) constraints. todo: fix
        # for index in range(self.nx * (self.horizon + 1), lbx.size1(), self.nu):
        #     expression = ""
        #     for input_ in range(self.nu):
        #         key = f'u_{input_}'
        #         expression += f"{lbx[index + input_]} <= {opt_to_state_input_dict[key]} <= {ubx[index + input_]}\n"
        #     print(expression)
        return constraints, lbg, ubg, lbx, ubx

    def _quad_form(self, z, Q):
        return casadi.mtimes(z.T, casadi.mtimes(Q, z))  # z.T @ Q @ z

    def objective_function_setup(self):
        cost = 0
        ''' Lagrange/stage cost '''
        # tracking cost
        u_prev = self.u_prev
        for k in range(self.horizon):
            if self.normalize_yaw_error:
                cost += self._quad_form(self.z_dv[0:3, k] - self.z_ref[0:3, k], self.Q[0:3, 0:3])
                cost += self._quad_form(casadi.fmod(self.z_dv[3, k] - self.z_ref[3, k] + np.pi, 2 * np.pi) - np.pi,
                                        self.Q[3, 3])  # normalizes the angle in the range [-pi, pi)
            else:
                state_error = self.z_dv[:, k] - self.z_ref[:, k]  # self.z_dv[:, k + 1] - self.z_ref[:, k]
                cost += self._quad_form(state_error, self.Q)

            # input cost.
            if k < (self.horizon - 1):
                control_error = self.u_dv[:, k]  # - self.u_ref[:, k]
                cost += self._quad_form(control_error, self.R)

                # input derivative cost
                input_rate = (self.u_dv[:, k] - u_prev)  # todo: just use self.u_rate_dv indices
                cost += self._quad_form(input_rate, self.Rd)  # self.u_dv[:, k + 1] - self.u_dv[:, k]
                u_prev = self.u_dv[:, k]

                # slack cost on input derivatives
                if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                    if self.slack_objective_is_quadratic:
                        cost += self._quad_form(self.sl_dv[:, k], self.P_u_rate)  # slack cost
                    else:
                        slack_weights = casadi.diag(self.P_u_rate)  # to convert to a column vector
                        cost += casadi.mtimes(slack_weights.T, self.sl_dv[:, k])  # slack cost

                # slack cost for obstacle  avoidance
                if not casadi.is_equal(self.P_obstacle_avoidance, casadi.DM.zeros((self.n_obstacles, self.n_obstacles))):
                    if self.slack_objective_is_quadratic:
                        cost += self._quad_form(self.sl_obs_dv[:, k], self.P_obstacle_avoidance)  # slack cost
                    else:
                        slack_weights = casadi.diag(self.P_obstacle_avoidance)  # to convert to a column vector
                        cost += casadi.mtimes(slack_weights.T, self.sl_obs_dv[:, k])  # slack cost

        ''' Mayer/terminal cost '''
        cost += self._quad_form(self.z_dv[:, self.horizon] - self.z_ref[:, self.horizon], self.Qf)  # terminal state

        cost *= 0.5
        return cost

    def setup_solver(self, cost, constraints, solver_type='nlp',
                     solver_options=None, solver_='ipopt', suppress_output=True, use_nlp_interface_for_qp=True,
                     overwrite_c_code=True):
        # todo: setup overwriting c code by checking for 'nlp.so' and 'nlp.c' files. See my do_mpc mpc_setup script
        flat_z = casadi.reshape(self.z_dv, self.nx * (self.horizon + 1), 1)  # flatten (self.nx * (self.horizon + 1), 1)
        flat_u = casadi.reshape(self.u_dv, self.nu * self.horizon, 1)  # flatten (self.nu * self.horizon, 1)
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            flat_sl = casadi.reshape(self.sl_dv, self.nu * self.horizon, 1)

        if not casadi.is_equal(self.P_obstacle_avoidance, casadi.DM.zeros((self.n_obstacles, self.n_obstacles))) and self.n_obstacles > 0:
            flat_sl_obs = casadi.reshape(self.sl_obs_dv, self.n_obstacles * (self.horizon + 1), 1)

        flat_z_ref = casadi.reshape(self.z_ref, self.nx * (self.horizon + 1),
                                    1)  # flatten (self.nx * (self.horizon + 1), 1)
        flat_z_k = casadi.reshape(self.z_k, self.nx, 1)
        flat_u_ref = casadi.reshape(self.u_ref, -1, 1)  # flatten     (self.nu * self.horizon, 1)
        flat_u_prev = casadi.reshape(self.u_prev, self.nu, 1)

        # flatten and append inputs and states
        opt_variables = casadi.vertcat(
                flat_z,
                flat_u,
        )

        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            opt_variables = casadi.vertcat(opt_variables, flat_sl)

        if not casadi.is_equal(self.P_obstacle_avoidance, casadi.DM.zeros((self.n_obstacles, self.n_obstacles))) and self.n_obstacles > 0:
            opt_variables = casadi.vertcat(opt_variables, flat_sl_obs)

        opt_params = casadi.vertcat(
                flat_z_ref,
                flat_z_k,
                flat_u_prev
        )  # casadi.vertcat(flat_z_ref, flat_u_ref)

        if self.n_obstacles > 0:
            flat_obstacles = casadi.reshape(self.obstacles, 3 * self.n_obstacles * (self.horizon + 1), 1)
            opt_params = casadi.vertcat(
                opt_params,
                flat_obstacles,
                self.ego_radius
            )

        # lbg, ubg constraints (i.e dynamic constraints in the form of an equation)
        g = casadi.vertcat(constraints)

        optimization_prob = {'f': cost, 'x': opt_variables, 'p': opt_params, 'g': g}

        if solver_options is None:
            solver_options = {
                'record_time': True,
                'print_time': not suppress_output,
                'verbose': not suppress_output,
                'expand': True,
                # 'max_cpu_time': 0.1,
                # 'max_iter': 1000,
            }

            if solver_ == 'ipopt':
                if solver_type != 'quad':
                    ipopt_options = {
                        'ipopt.print_level': not suppress_output,
                        'ipopt.sb': 'yes',
                        'ipopt.max_iter': 100,  # 2000
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6,
                        'error_on_fail': False,  # to raise an exception if the solver fails to find a solution
                        # "ipopt.linear_solver": "ma27",  # Comment this line if you don't have MA27
                    }
                    solver_options.update(ipopt_options)

                    if self.warmstart:
                        # see (https://coin-or.github.io/Ipopt/OPTIONS.html#OPT_Warm_Start and
                        # https://www.gams.com/latest/docs/S_IPOPT.html#IPOPT_WARMSTART) for tips
                        warmstart_options = {
                            'ipopt.warm_start_init_point': 'yes',
                            'ipopt.warm_start_bound_push': 1e-8,  # 1e-9
                            'ipopt.warm_start_mult_bound_push': 1e-8,  # 1e-9
                            'ipopt.mu_init': 1e-5,
                            'ipopt.bound_relax_factor': 1e-9,
                            # # todo: test with the below values
                            # 'ipopt.warm_start_bound_frac': 1e-9,
                            # 'ipopt.warm_start_slack_bound_frac': 1e-9,
                            # 'ipopt.mu_strategy': 'monotone',
                            # 'mu_init': 0.0001,
                            # 'nlp_scaling_method': 'none',
                        }
                        solver_options.update(warmstart_options)

                else:
                    use_nlp_interface_for_qp = True  # ipopt requires the nlp interface below
                    if use_nlp_interface_for_qp:
                        # see https://github.com/casadi/casadi/wiki/FAQ:-how-to-use-IPOPT-as-QP-solver%3F
                        # must use alternate nlpsol formulation (or qpsol)
                        ipopt_quad_settings = {
                            'qpsol': 'nlpsol',
                            'qpsol_options': {}
                        }

                        ipopt_quad_settings['qpsol_options']['nlpsol'] = 'ipopt'
                        ipopt_quad_settings['qpsol_options']['nlpsol_options'] = {}
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt'] = {}
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.tol'] = 1e-12  # 1e-7
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.tiny_step_tol'] = 1e-20
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.fixed_variable_treatment'] = 'make_constraint'
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.hessian_constant'] = 'yes'
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.jac_c_constant'] = 'yes'
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.jac_d_constant'] = 'yes'
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.accept_every_trial_step'] = 'yes'
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.mu_init'] = 1e-5  # 1e-3
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.linear_solver'] = 'ma27'
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.sb'] = 'yes'
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.max_iter'] = 100  # 2000
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.acceptable_tol'] = 1e-8
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.acceptable_obj_change_tol'] = 1e-6

                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.print_level'] = 0
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['print_time'] = False
                        solver_options.update(ipopt_quad_settings)

                    else:
                        raise NotImplementedError(f"Solver {solver_}, with optimization type: {solver_type} "
                                                  f"requires setting {use_nlp_interface_for_qp}. "
                                                  f"Either set 'use_nlp_interface_for_qp' to "
                                                  f"True or optimization type to 'nlp'.")

        '''
        Setup code generation flags. See the following for details:
           * https://github.com/casadi/casadi/blob/main/docs/examples/python/nlp_codegen.py#L50
           * https://github.com/casadi/casadi/wiki/FAQ:-how-to-perform-jit-for-function-evaluations-of-my-optimization-problem%3F
           * https://github.com/casadi/casadi/wiki/FAQ:-how-to-make-jit-not-recompile-when-problem-is-unchanged
        '''
        # compiler and flags are common to jit and external code_generation
        compiler = "ccache gcc"  # Linux (gcc, clang, ccache gcc)  # todo: catch exception if ccache is not installed
        # compiler = "clang"  # OSX
        # compiler = "cl.exe" # Windows
        flags = ["-O3"]  # Linux/OSX. ['O3'] enables the most optimizations
        if compiler == "gcc":
            flags = [
                # "-O3"
                "-Ofast",  # enables even more optimization than -O3 but could be less precise/safe but is okay in most situations
                "-march=native",  # optimizes for the specific hardware (CPU) but isn't transferable
                "-fPIC",  # for shared_libraries
            ]  # for performance boost using gcc

        if self.code_gen_mode == 'jit':
            # JIT compilation automatically compiles helper functions as well as the solver.
            # Yields about a 2.5x increase in performance
            # By default, the compiler will be gcc or cl.exe
            jit_options = {
                "jit": True, "compiler": "shell", "jit_temp_suffix": False,
                "jit_options": {
                    "flags": flags, "compiler": compiler, "verbose": True,
                    # "compiler_flags": flags
                }, # use ("compiler": "ccache gcc") if ccache is installed for a performance boost
                'jit_cleanup': False  # True: delete files on shutdown
                       }
            solver_options.update(jit_options)

        if solver_type == 'quad':
            '''
            f must be convex and g must be linear
            solvers: qpoases, osqp
            '''
            quad_solver = solver_
            if quad_solver not in ["osqp", "qpoases", "qrqp", "ipopt"]:
                quad_solver = 'osqp'  # osqp, qpoases
                print(f"solver_: {solver_} not valid. Defaulting to {quad_solver}")

            if (quad_solver == "osqp") and not use_nlp_interface_for_qp:
                quad_settings = {
                    # 'verbose': not suppress_output,
                    # 'max_iter': 10,
                    # 'eps_pr': 1e-8,
                    # 'eps_r': 1e-8,
                    # 'warm_start_dual': True,
                    # 'warm_start_primal': True,
                    # 'sparse': True,  # for qpoases only
                }
                solver_options.update(quad_settings)
            if not use_nlp_interface_for_qp:
                solver = casadi.qpsol('solver', quad_solver, optimization_prob, solver_options)
            else:
                # see https://web.casadi.org/python-api/
                qp_options = {
                    "qpsol": "nlpsol" if solver_ == "ipopt" else solver_,  # else solver_
                    "print_time": not suppress_output,
                    "convexify_strategy": "regularize",  # NONE|regularize|eigen- reflect|eigen-clip
                    "convexify_margin": 1e-4,
                    "max_iter": 30,  # ipopt, 2000
                    "hessian_approximation": "exact",   # "limited-memory" (sqpmethod), "exact" (all plugins), "gauss-neuton" (scpgen). Feasiblesqpmethod only works with exact and get better performance with regularization
                    # "tol_du": 1e-10,  # ipopt, qrqp
                    # "tol_pr": 1e-10,   # ipopt
                    # "min_step_size": 1e-14,  # ipopt, qrqp
                    # "max_iter_ls": 0,  #  qrqp
                    "print_header": not suppress_output,
                    "print_iteration": not suppress_output,
                    # 'init_feasible': True,  # for sqpmethod and feasiblesqpmethod plugins
                    # "warmstart": True,  # for sqpmethod or blocksqp plugin (QRQP)
                    # "qp_init": True,  # for blocksqp plugin
                    "expand": True
                }
                solver_options.update(qp_options)

                common_qp_sol_options = {
                        "print_problem": not suppress_output,
                        "print_out": not suppress_output,
                        "print_iter": not suppress_output,  # disable for osqp/QPOases/ipopt
                        "print_time": not suppress_output,
                        # 'printLevel': 'none',  # For QPOases
                        "error_on_fail": False,
                        # 'warm_start_dual': True,  # OSQP
                        # 'warm_start_primal': True,  # OSQP
                        'verbose': not suppress_output,
                        # "sparse": True, # (for QPOases only)
                    }  # common to ipopt and the QP solvers

                if solver_options.get('qpsol_options', None) is None:
                    solver_options['qpsol_options'] = common_qp_sol_options
                else:
                    solver_options['qpsol_options'].update(common_qp_sol_options)

                # sqpmethod, feasiblesqpmethod, blocksqp
                plugin = 'sqpmethod'
                if plugin not in ["sqpmethod", "feasiblesqpmethod", "blocksqp", "scpgen"]:
                    plugin = "sqpmethod"

                if plugin in ["feasiblesqpmethod", "blocksqp", "scpgen"]:
                    print("Only 'sqpmethod' works at the moment. \n "
                          "Feasiblesqpmethod is not stable/robust and fails when the initial condition is not "
                          "sufficient or without warmstarting. I have not been able to get it to work yet. \n "
                          "Blocksqp requires HSL to be installed, specifically the ma27 solver. \n"
                          "For scpgen there is a bug in casadi at least as of 3.6.6 that leads to an Exception when creating. \n"
                          "Defaulting to sqpmethod plugin.")
                    plugin = "sqpmethod"

                # qrqp, osqp, qpoases
                solver = casadi.nlpsol('solver', plugin, optimization_prob, solver_options)

                if self.code_gen_mode == 'external':
                    # Generate C code for the NLP functions. todo: rename the files
                    solver.generate_dependencies("casadi_nlp.c")

                    if isinstance(compiler, str):
                        compiler = compiler.split()

                    import subprocess
                    # On Windows, use other flags
                    cmd_args = compiler + ["-shared"] + flags + ["casadi_nlp.c", "-o", "casadi_nlp.so"]
                    try:
                        result = subprocess.run(cmd_args, check=True, capture_output=True)
                        if result.returncode != 0:
                            print("Error:", result.returncode)
                            print("Output:", result.stdout)
                            print("Error output:", result.stderr)
                    except subprocess.CalledProcessError as e:
                        # print("Error:", e)
                        # print("Return code:", e.returncode)
                        # print("Error output:", e.stderr)
                        if "ccache" in str(e) or "No such file or directory" in e.stderr:
                            cmd_args.pop(0)
                            result = subprocess.run(cmd_args, check=True, capture_output=True)

                    # Create a new NLP solver instance from the compiled code
                    solver = casadi.nlpsol("solver", solver, "./casadi_nlp.so")

        else:
            nlp_solver = solver_
            # nlp_settings = {
            #
            # }
            # solver_options.update(nlp_settings)
            solver = casadi.nlpsol('solver', nlp_solver, optimization_prob, solver_options)
            if self.code_gen_mode == 'external':
                # Generate C code for the NLP functions. todo: rename the files
                solver.generate_dependencies("casadi_nlp.c")

                if isinstance(compiler, str):
                    compiler = compiler.split()

                import subprocess
                # On Windows, use other flags
                cmd_args = compiler + ["-shared"] + flags + ["casadi_nlp.c", "-o", "casadi_nlp.so"]
                try:
                    result = subprocess.run(cmd_args, check=True, capture_output=True)
                    if result.returncode != 0:
                        print("Error:", result.returncode)
                        print("Output:", result.stdout)
                        print("Error output:", result.stderr)
                except subprocess.CalledProcessError as e:
                    # print("Error:", e)
                    # print("Return code:", e.returncode)
                    # print("Error output:", e.stderr)
                    if "ccache" in str(e) or "No such file or directory" in e.stderr:
                        cmd_args.pop(0)
                        result = subprocess.run(cmd_args, check=True, capture_output=True)

                # Create a new NLP solver instance from the compiled code
                solver = casadi.nlpsol("solver", nlp_solver, "./casadi_nlp.so")
        return solver, opt_variables, opt_params, g

    def update_initial_condition(self, x0, y0, vel0, psi0):
        self.z_k_value[:, 0] = [x0, y0, vel0, psi0]

    def update_reference(self, x_ref, y_ref, v_ref, psi_ref, acc_ref=0.0, delta_ref=0.0):
        """
        Todo: pass z_ref instead
        :param x_ref:
        :param y_ref:
        :param v_ref:
        :param psi_ref:
        :return:
        """
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

    def update_obstacle_distances(self, z_k, z_dv, obstacles_state, ego_radius, slack_obs=None):
        if slack_obs is None:
            self.obstacle_distances_value = self.obstacle_distances_func(
                z_k, z_dv, obstacles_state, ego_radius).full()
        else:
            self.obstacle_distances_value = self.obstacle_distances_func(
                z_k, z_dv, obstacles_state, ego_radius, slack_obs).full()

    def update_ego_radius(self, ego_radius):
        self.ego_radius_value = self.ego_radius_func(ego_radius).full().item()

    def update(self, state=None, ref_traj=None, previous_input=None, warmstart_variables=None):
        self.update_initial_condition(*state)
        self.update_reference(*ref_traj)
        self.update_previous_input(*previous_input)

        if self.warmstart and len(warmstart_variables) > 0:
            # Warm Start used if provided.  Else I believe the problem is solved from scratch with initial values of 0.
            self.z_dv_value = warmstart_variables['z_ws']
            self.u_dv_value = warmstart_variables['u_ws']
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                if warmstart_variables.get('sl_ws', None) is not None:
                    self.sl_dv_value = warmstart_variables['sl_ws']

            if not casadi.is_equal(self.P_obstacle_avoidance,
                                   casadi.DM.zeros((self.n_obstacles, self.n_obstacles))) and self.n_obstacles > 0:
                if warmstart_variables.get('sl_obs_ws', None) is not None:
                    self.sl_obs_dv_value = warmstart_variables['sl_obs_ws']

            # lagrange multipliers
            self.lam_x_value = warmstart_variables['lam_x']
            self.lam_g_value = warmstart_variables['lam_g']
            self.lam_p_value = warmstart_variables['lam_p']

    def solve(self):
        """
        Call update method before this.
        :return:
        """
        st = time.process_time()

        sl_mpc = np.zeros((self.horizon, self.nu))  # or return None
        sl_obs_mpc = np.zeros((self.horizon, self.n_obstacles))
        try:
            """
            init_control, x0, opt_variables: [z_dv, u_dv], [(nx, horizon+1), (nu, horizon)]
            control_parameter, c_p, opt_parameters: [x_ref, u_ref]
            
            Can just substitue values using casadi.substitute, 
                e.g self.u_dv_value[:, 0] = casadi.substitute(self.u_dv[:, 0], self.u_dv[:, 0], [acc_prev, delta_prev])
            """
            opt_variables = casadi.vertcat(
                    casadi.reshape(self.z_dv_value, self.nx * (self.horizon + 1), 1),
                    casadi.reshape(self.u_dv_value, self.nu * self.horizon, 1)
            )
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                opt_variables = casadi.vertcat(opt_variables,
                                               casadi.reshape(self.sl_dv_value, self.nu * self.horizon, 1))

            if not casadi.is_equal(self.P_obstacle_avoidance,
                                   casadi.DM.zeros((self.n_obstacles, self.n_obstacles))) and self.n_obstacles > 0:
                opt_variables = casadi.vertcat(opt_variables,
                                               casadi.reshape(self.sl_obs_dv_value, self.n_obstacles * (self.horizon + 1), 1))

            opt_parameters = casadi.vertcat(
                    casadi.reshape(self.z_ref_value, self.nx * (self.horizon + 1), 1),
                    casadi.reshape(self.z_k_value, self.nx, 1),
                    casadi.reshape(self.u_prev_value, self.nu, 1)
            )

            if self.n_obstacles > 0:
                # todo: update obstacles state
                # self.update_obstacles_state(self.obstacles_value)  # todo: do this in the ROS node
                opt_parameters = casadi.vertcat(
                    opt_parameters,
                    casadi.reshape(self.obstacles_value, 3 * self.n_obstacles * (self.horizon + 1), 1),
                    self.ego_radius_value
                )

            dual_variables = {}
            if self.warmstart:
                if self.lam_x_value is not None:
                   dual_variables['lam_x0'] = self.lam_x_value

                if self.lam_g_value is not None:
                    dual_variables['lam_g0'] = self.lam_g_value

            result_dict = self.solver(x0=opt_variables, p=opt_parameters,
                                      lbg=self.lbg, lbx=self.lbx, ubg=self.ubg, ubx=self.ubx, **dual_variables)
            sol = result_dict['x']
            cost = result_dict['f']
            g = result_dict['g']
            lam_g = result_dict['lam_g']  # lagrange multipliers (constraints)
            lam_x = result_dict['lam_x']  # lagrange multipliers (optimization variables)
            lam_p = result_dict['lam_p']  # lagrange multipliers (parameters)

            # todo: add hessian calculation here for convexity check
            #  (https://www.mathworks.com/help/optim/ug/hessian.html#bsapedt | https://github.com/casadi/casadi/issues/3297#issuecomment-1662226255 | https://github.com/meco-group/rockit/blob/master/rockit/ocp.py | https://groups.google.com/g/casadi-users/c/h05oCge2vkk)

            # # to test constraints
            # for index in range(g.size1()):
            #     print(f"{self.lbg[index]} <= {g[index]} <= {self.ubg[index]}")

            # Optimal solution.
            z_mpc = casadi.reshape(sol[:self.nx * (self.horizon + 1)], self.nx,
                                   self.horizon + 1).full()  # sol.value(self.z_dv)
            u_mpc = casadi.reshape(
                sol[self.nx * (self.horizon + 1):self.nx * (self.horizon + 1) + self.nu * self.horizon],
                self.nu, self.horizon).full()
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                sl_mpc = casadi.reshape(sol[self.nx * (self.horizon + 1) + (self.nu * self.horizon):(
                            self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon))], self.nu,
                                        self.horizon).full()
                # sl_mpc = casadi.reshape(g[self.nx * (self.horizon + 1) + self.nu * self.horizon:], self.nu, -1).full()  # wrong.

            if not casadi.is_equal(self.P_obstacle_avoidance,
                                   casadi.DM.zeros((self.n_obstacles, self.n_obstacles))) and self.n_obstacles > 0:
                sl_obs_mpc = casadi.reshape(sol[(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)) + (self.n_obstacles * (self.horizon + 1))], self.n_obstacles,
                                        self.horizon + 1).full()

            # evaluate/calculate u_rate
            self.update_u_rate(self.u_prev_value, u_mpc)
            u_rate = self.u_rate_dv_value

            # unpack the parameters to ensure it matches the original after unpacking as a sanity check
            z_ref = casadi.reshape(opt_parameters[:self.nx * (self.horizon + 1)], self.nx, self.horizon + 1).full()
            z_k_value = casadi.reshape(
                opt_parameters[self.nx * (self.horizon + 1):self.nx * (self.horizon + 1) + self.nx],
                self.nx, 1).full()
            u_prev_value = casadi.reshape(
                    opt_parameters[self.nx * (self.horizon + 1) + self.nx:self.nx * (self.horizon + 1) + self.nx + self.nu],
                    self.nu, 1).full()
            if self.n_obstacles > 0:
                obstacles_value = casadi.reshape(
                    opt_parameters[self.nx * (self.horizon + 1) + self.nx + self.nu:self.nx * (self.horizon + 1) + self.nx + self.nu + (3 * self.n_obstacles * (self.horizon + 1))],
                    3 * self.n_obstacles, self.horizon + 1).full()
                ego_radius_value = casadi.reshape(
                        opt_parameters[self.nx * (self.horizon + 1) + self.nx + self.nu + (3 * self.n_obstacles * (self.horizon + 1))],
                        1, 1).full().item()

            try:
                iteration_count = self.solver.stats()['iter_count']  # sol.stats()["iter_count"]
            except KeyError:
                # blocksqp plugin does not return iteration count
                iteration_count = -1
            is_opt = True  # self.solver.stats()['success']
            success = self.solver.stats()['success']
            return_status = self.solver.stats()['return_status']
            unified_return_status = self.solver.stats()['unified_return_status']
            if not success:
                print(f"Return status: {return_status}, unified return status: {unified_return_status}")
            solve_time = self.solver.stats()['t_wall_total']

            # Useful fo debugging and checking solver stats
            initial_condition_constraint = casadi.reshape(g[:self.nx], self.nx, 1)
            model_dynamics = casadi.reshape(g[self.nx:self.nx + (self.nx * self.horizon)], self.nx, self.horizon)
            # rate constraints
            u_rate_i = casadi.reshape(
                g[self.nx + (self.nx * self.horizon):self.nx + (self.nx * self.horizon) + self.nu * self.horizon], self.nu,
                -1).full()
            u_rate_ii = casadi.reshape(g[self.nx + (self.nx * self.horizon) + self.nu * self.horizon:self.nx + (self.nx *
                    self.horizon) + (self.nu * self.horizon) + (self.nu * self.horizon)], self.nu, -1).full()
            ''' u_rate = u_rate_i - sl_mpc or -u_rate_ii + sl_mpc '''

            if self.n_obstacles > 0:
                # update obstacle distances
                if casadi.is_equal(self.P_obstacle_avoidance, casadi.DM.zeros((self.n_obstacles, self.n_obstacles))):
                    self.update_obstacle_distances(z_k_value, z_mpc, self.obstacles_value, self.ego_radius_value)
                else:
                    self.update_obstacle_distances(z_k_value, z_mpc, self.obstacles_value, self.ego_radius_value, sl_obs_mpc)

            # # get functions used by the solver.
            # # Could be useful when replacing default functions.
            # # See (https://github.com/casadi/casadi/wiki/FAQ:-How-to-specify-a-custom-Hessian-approximation%3F | https://groups.google.com/g/casadi-users/c/XnDBUWrPTlQ)
            # hess_l = self.solver.get_function('nlp_hess_l')

            # # To verify for debugging purposes (should be commented out)
            # opt_to_state_input_dict = {
            #     'x_0': 'x', 'x_1': 'y', 'x_2': 'vel', 'x_3': 'psi',
            #     'u_0': 'acc', 'u_1': 'delta'
            # }
            # for index in range(g.size1()):
            #     print(f"{self.lbg[index]} <= {g[index]} <= {self.ubg[index]}")
            # print("\n")
            #
            # # print state (z) constraints.
            # for index in range(0, self.nx * (self.horizon + 1), self.nx):
            #     expression = ""
            #     for state in range(self.nx):
            #         key = f'x_{state}'
            #         expression += f"{self.lbx[index + state]} <= {sol[index + state]} <= {self.ubx[index + state]}\n"
            #     print(expression)
            # print("\n")
            #
            # # print input (u) constraints.
            # for index in range(self.nx * (self.horizon + 1), self.nx * (self.horizon + 1) + self.nu * self.horizon,
            #                    self.nu):
            #     expression = ""
            #     for input_ in range(self.nu):
            #         expression += f"{self.lbx[index + input_]} <= {sol[index + input_]} <= {self.ubx[index + input_]}\n"
            #     print(expression)
            # print("\n")
            #
            # # print input (slack) constraints.
            # if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            #     for index in range(self.nx * (self.horizon + 1) + (self.nu * self.horizon), (
            #             self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)), self.nu):
            #         expression = ""
            #         for input_ in range(self.nu):
            #             # key = f'u_{input_}'
            #             expression += f"{self.lbx[index + input_]} <= {sol[index + input_]} <= {self.ubx[index + input_]}\n"
            #         print(expression)
            #     print("\n")
            #
            # # # print input rate constraints. todo
            # # for index in range(self.nx * (self.horizon + 1), self.nx * (self.horizon + 1) + self.nu * self.horizon,
            # #                    self.nu):
            # #     expression = ""
            # #     for input_ in range(self.nu):
            # #         expression += f"{self.lbx[index + input_]} <= {sol[index + input_]} <= {self.ubx[index + input_]}\n"
            # #     print(expression)
            # # print("\n")
        except Exception as e:
            print(f"Encountered: {e}")
            # Suboptimal solution (e.g. timed out). Todo: decide what to do
            u_mpc = np.zeros((self.nu, self.horizon))  # self.mpc.debug.value(self.u_dv)
            z_mpc = np.zeros((self.nx, self.horizon + 1))  # self.mpc.debug.value(self.z_dv)
            u_rate = np.zeros((self.nu, self.horizon))
            # if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            #     sl_mpc = None  # self.mpc.debug.value(self.sl_dv)
            z_ref = casadi.reshape(opt_parameters[:self.nx * (self.horizon + 1)], self.nx,
                                   self.horizon + 1).full()  # self.mpc.debug.value(self.z_ref)
            z_k_value = casadi.reshape(
                opt_parameters[self.nx * (self.horizon + 1):self.nx * (self.horizon + 1) + self.nx],
                self.nx, 1).full()
            u_prev_value = casadi.reshape(opt_parameters[self.nx * (self.horizon + 1) + self.nx:], self.nu, 1).full()
            iteration_count = None
            is_opt = False  # self.solver.stats()['success']
            success = False
            lam_g = np.zeros(self.lbg.shape)  # lagrange multipliers (constraints)
            lam_x = np.zeros_like(opt_parameters)  # lagrange multipliers (optimization variables)
            lam_p = np.zeros_like(opt_parameters)  # lagrange multipliers (parameters)

            solve_time = time.process_time() - st  # self.solver.stats()['t_wall_total']

        sol_dict = {'u_control': u_mpc[:, 0],
                    'u_mpc': u_mpc,
                    'z_mpc': z_mpc,
                    'u_rate': u_rate,
                    'u_prev': u_prev_value,
                    'sl_mpc': sl_mpc,
                    'sl_obs_mpc': sl_obs_mpc,
                    'z_ref': z_ref.T,
                    'optimal': is_opt,
                    'solve_time': solve_time,  # todo: get from solver time
                    'iter_count': iteration_count,
                    'solver_status': success,
                    'lam_x': lam_x,
                    'lam_g': lam_g,
                    'lam_p': lam_p,
                    'solver_stats': self.solver.stats()
                    }

        return sol_dict

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
        # update
        # solve
        pass


if __name__ == '__main__':
    dt = 0.0001
    x0 = np.array([[1., 0.3, 3, 0.1]]).T
    u0 = np.array([[1., 0.3]]).T
    kin_mpc = DiscreteKinematicMPCCasadi(vehicle=None, horizon=25, sample_time=dt, wheelbase=0.256,
                                 nx=4, nu=2, x0=x0, u0=u0,
                                 Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                                 Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                                 vel_bound=(-5.0, 5.0), delta_bound=(-23.0, 23.0), acc_bound=(-3.0, 3.0),
                                 jerk_bound=(-1.5, 1.5), delta_rate_bound=(-352.9411764706, 352.9411764706),
                                 symbol_type='SX', suppress_ipopt_output=True)
    solution_dictionary = kin_mpc.solve()
    print(solution_dictionary)
