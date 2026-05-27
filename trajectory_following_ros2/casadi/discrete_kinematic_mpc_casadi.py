"""
See:
    To parallelize casadi:
        * https://web.casadi.org/docs/#for-loop-equivalents
        * https://github.com/nirajbasnet/Nonlinear_MPCC_for_autonomous_racing/blob/master/nonlinear_mpc_casadi/scripts/Nonlinear_MPC.py#L157

Notes:
    * QP solvers work when slack is added to rate inputs. Fails without slack or RD
"""
import time
import numpy as np
import casadi

from trajectory_following_ros2.casadi.kinematic_bicycle_model import KinematicBicycleModel
from trajectory_following_ros2.casadi._casadi_base import KinematicMPCBase


class DiscreteKinematicMPCCasadi(KinematicMPCBase):
    """Discrete-time LTV kinematic MPC with optional obstacle avoidance."""

    def __init__(self, vehicle=None, horizon=15, sample_time=0.02, wheelbase=0.256,
                 nx=4, nu=2, x0=None, u0=None,
                 Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                 Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                 vel_bound=(-5.0, 5.0), delta_bound=(-np.radians(23.0), np.radians(23.0)), acc_bound=(-3.0, 3.0),
                 jerk_bound=(-1.5, 1.5), delta_rate_bound=(-np.radians(352.9411764706), np.radians(352.9411764706)),
                 symbol_type='MX', warmstart=True,
                 solver_options=None, solver_type='nlp', solver='ipopt', suppress_ipopt_output=True,
                 normalize_yaw_error=True,
                 slack_weights_u_rate=(1e-6, 1e-6),
                 slack_scale_u_rate=(1.0, 1.0),
                 slack_upper_bound_u_rate=None,
                 slack_objective_is_quadratic=False,
                 code_gen_mode='jit',
                 num_obstacles=1, collision_avoidance_scheme='cbf',
                 ego_radius=None,
                 slack_weights_obstacle_avoidance=None,
                 slack_upper_bound_obstacle_avoidance=None):
        super().__init__(
            vehicle=vehicle, horizon=horizon, sample_time=sample_time, wheelbase=wheelbase,
            nx=nx, nu=nu, x0=x0, u0=u0,
            Q=Q, R=R, Qf=Qf, Rd=Rd,
            vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound,
            jerk_bound=jerk_bound, delta_rate_bound=delta_rate_bound,
            symbol_type=symbol_type, warmstart=warmstart,
            solver_options=solver_options, solver_type=solver_type, solver=solver,
            suppress_ipopt_output=suppress_ipopt_output,
            normalize_yaw_error=normalize_yaw_error,
            slack_weights_u_rate=slack_weights_u_rate,
            slack_scale_u_rate=slack_scale_u_rate,
            slack_upper_bound_u_rate=slack_upper_bound_u_rate,
            slack_objective_is_quadratic=slack_objective_is_quadratic,
            code_gen_mode=code_gen_mode,
            num_obstacles=num_obstacles,
            collision_avoidance_scheme=collision_avoidance_scheme,
            ego_radius=ego_radius,
            slack_weights_obstacle_avoidance=slack_weights_obstacle_avoidance,
            slack_upper_bound_obstacle_avoidance=slack_upper_bound_obstacle_avoidance)

    def _build_vehicle_model(self, vehicle, nx, nu, x0,
                             vel_bound, delta_bound, acc_bound,
                             jerk_bound, delta_rate_bound):
        if vehicle is not None:
            self.vehicle = vehicle
            self.model = self.vehicle.model
        else:
            self.vehicle = KinematicBicycleModel(
                nx=nx, nu=nu, x0=x0,
                vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound,
                jerk_bound=jerk_bound, delta_rate_bound=delta_rate_bound,
                vehicle_parameters={"wheelbase": self.wheelbase}, sample_time=self.Ts,
                model_type='kinematic', model_name='vehicle_kinematic_model',
                symbol_type=self.symbol_type,
                discretization_method='cvodes', discrete=True)
            self.model = self.vehicle.model

        self.ode = self.model.f_expl_expr
        self.ode = casadi.substitute(
            self.ode,
            casadi.vertcat(self.model.params.wheelbase, self.model.params.dt),
            casadi.vertcat(self.wheelbase, self.Ts))
        self.ode_function = self.vehicle.create_ode_function(
            self.ode, self.model.x, self.model.u, function_name='discrete_ltv_ode')

    def constraints_setup(
            self, vel_bound=None, delta_bound=None, acc_bound=None,
            jerk_bound=None, delta_rate_bound=None, reset=False
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
        """
        if delta_bound is None:
            delta_bound = [-np.radians(23.0), np.radians(23.0)]

        if vel_bound is None:
            vel_bound = [-10.0, 10.0]

        if acc_bound is None:
            acc_bound = [-3.0, 3.0]

        if jerk_bound is None:
            jerk_bound = [-1.5, 1.5]

        if delta_rate_bound is None:
            delta_rate_bound = [-np.radians(352.9411764706), np.radians(352.9411764706)]

        u_min = np.array([[acc_bound[0]], [delta_bound[0]]])
        u_max = np.array([[acc_bound[1]], [delta_bound[1]]])

        u_dot_min = np.array([[jerk_bound[0]], [delta_rate_bound[0]]])
        u_dot_max = np.array([[jerk_bound[1]], [delta_rate_bound[1]]])

        constraints = []

        lbg = casadi.DM.zeros((self.nx * (self.horizon + 1)), 1)
        ubg = casadi.DM.zeros((self.nx * (self.horizon + 1)), 1)

        # initial state constraints
        constraints.append(self.z_dv[:, 0] - self.z_k)

        z_pred_list = [self.z_k]
        for k in range(self.horizon):
            z_next = self.ode_function(self.z_dv[:, k], self.u_dv[:, k])
            z_pred_list.append(z_next)
            dynamics = self.z_dv[:, k + 1] - z_next
            constraints.append(dynamics)

        constraints = casadi.vertcat(*constraints)
        self.z_pred_dv = casadi.horzcat(*z_pred_list)

        # Input Rate Bound Constraints
        u_prev = self.u_prev
        u_dot_list = []
        slack_flag = int(not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))))
        for k in range(self.horizon):
            u_dot = self.u_dv[:, k] - u_prev
            """
            The formulation below, i.e LBG - slack <= constraints <= UBG + slack is valid.
            However, casadi requires LBG and UBG to only include numbers.
            Therefore, we reformulate with an equivalent expression:
               i. LBG <= u_dot + slack <= np.inf (i.e, UBG), and
               ii. -UBG <= -u_dot + slack <= np.inf (or -np.inf (i.e, LBG) <= u_dot - slack <= UBG)
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
                -u_dot + (slack_flag * casadi.mtimes(self.V_scale_u_rate, self.sl_dv[:, k])))
            lbg = casadi.vertcat(lbg, -u_dot_max * self.Ts)
            ubg = casadi.vertcat(ubg, casadi.inf * casadi.DM.ones((self.nu, 1)) * self.Ts)

            u_prev = self.u_dv[:, k]
            u_dot_list.append(u_dot)

        u_dot_list = casadi.horzcat(*u_dot_list)
        self.u_rate_dv = u_dot_list

        # Obstacle avoidance constraints
        if self.n_obstacles > 0:
            distance_expression_list = []

            slack_obs_flag = int(not casadi.is_equal(
                self.P_obstacle_avoidance,
                casadi.DM.zeros(self.n_obstacles, self.n_obstacles)))

            # Euclidean: constrain stages 0..N (full horizon including terminal k=N).
            # CBF: constrain stages 0..N-1; the h(k+1) term at k=N-1 implicitly covers k=N.
            n_stages = self.horizon + 1 if self.collision_avoidance_scheme == 'euclidean' \
                else self.horizon
            for k in range(n_stages):
                for i in range(self.n_obstacles):
                    ego_xy = self.z_dv[0:2, k]
                    obs_state = self.obstacles[3 * i:3 * i + 3, k]
                    # Use per-obstacle slack sl_obs_dv[i, k] (scalar).
                    sl_i = self.sl_obs_dv[i, k] if slack_obs_flag else 0
                    dist_sq = casadi.sumsqr(ego_xy - obs_state[0:2])
                    h = dist_sq - (self.ego_radius + obs_state[2]
                                   + self.safe_distance - sl_i) ** 2

                    if self.collision_avoidance_scheme == 'euclidean':
                        distance_expression_list.append(h)
                        lbg = casadi.vertcat(lbg, casadi.DM([[0.]]))
                        ubg = casadi.vertcat(ubg, casadi.DM([[casadi.inf]]))
                    elif self.collision_avoidance_scheme == 'cbf':
                        ego_xy_next = self.z_dv[0:2, k + 1]
                        obs_state_next = self.obstacles[3 * i:3 * i + 3, k + 1]
                        sl_i_next = self.sl_obs_dv[i, k + 1] if slack_obs_flag else 0
                        dist_sq_next = casadi.sumsqr(ego_xy_next - obs_state_next[0:2])
                        h_next = dist_sq_next - (self.ego_radius + obs_state_next[2]
                                                  + self.safe_distance - sl_i_next) ** 2
                        distance_expression_list.append(h_next - h + self.gamma * h)
                        lbg = casadi.vertcat(lbg, casadi.DM([[0.]]))
                        ubg = casadi.vertcat(ubg, casadi.DM([[casadi.inf]]))

            self.obstacle_distances = casadi.vertcat(*distance_expression_list)
            constraints = casadi.vertcat(constraints, self.obstacle_distances)

        lbx = casadi.DM.zeros((self.nx * (self.horizon + 1) + self.nu * self.horizon, 1))
        ubx = casadi.DM.zeros((self.nx * (self.horizon + 1) + self.nu * self.horizon, 1))

        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            lbx = casadi.vertcat(lbx, casadi.DM.zeros((self.nu * self.horizon, 1)))
            ubx = casadi.vertcat(ubx, casadi.DM.zeros((self.nu * self.horizon, 1)))

        if not casadi.is_equal(self.P_obstacle_avoidance,
                               casadi.DM.zeros(self.n_obstacles, self.n_obstacles)):
            lbx = casadi.vertcat(lbx, casadi.DM.zeros((self.n_obstacles * (self.horizon + 1), 1)))
            ubx = casadi.vertcat(ubx, casadi.reshape(
                casadi.mtimes(casadi.DM.ones((self.horizon + 1)),
                              self.slack_upper_bound_obstacle_avoidance.T),
                self.n_obstacles * (self.horizon + 1), 1))

        # state constraints (self.v_dv, vel_bound[1]).
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

        # slack constraints
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            lbx[self.nx * (self.horizon + 1) + (self.nu * self.horizon):(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (
                            self.nu * self.horizon)):self.nu] = 0.0
            lbx[self.nx * (self.horizon + 1) + (self.nu * self.horizon) + 1:(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (
                            self.nu * self.horizon)):self.nu] = 0.0

            ubx[self.nx * (self.horizon + 1) + (self.nu * self.horizon):(
                    self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):self.nu] = \
                self.slack_upper_bound_u_rate[0]
            ubx[self.nx * (self.horizon + 1) + (self.nu * self.horizon) + 1:(
                    self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):self.nu] = \
                self.slack_upper_bound_u_rate[1]

        return constraints, lbg, ubg, lbx, ubx

    def setup_solver(self, cost, constraints, solver_type='nlp',
                     solver_options=None, solver='ipopt', suppress_output=True,
                     use_nlp_interface_for_qp=True, overwrite_c_code=True):
        flat_z = casadi.reshape(self.z_dv, self.nx * (self.horizon + 1), 1)
        flat_u = casadi.reshape(self.u_dv, self.nu * self.horizon, 1)
        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            flat_sl = casadi.reshape(self.sl_dv, self.nu * self.horizon, 1)

        if not casadi.is_equal(self.P_obstacle_avoidance,
                               casadi.DM.zeros(self.n_obstacles, self.n_obstacles)) \
                and self.n_obstacles > 0:
            flat_sl_obs = casadi.reshape(
                self.sl_obs_dv, self.n_obstacles * (self.horizon + 1), 1)

        flat_z_ref = casadi.reshape(self.z_ref, self.nx * (self.horizon + 1), 1)
        flat_z_k = casadi.reshape(self.z_k, self.nx, 1)
        flat_u_ref = casadi.reshape(self.u_ref, -1, 1)
        flat_u_prev = casadi.reshape(self.u_prev, self.nu, 1)

        opt_variables = casadi.vertcat(flat_z, flat_u)

        if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
            opt_variables = casadi.vertcat(opt_variables, flat_sl)

        if not casadi.is_equal(self.P_obstacle_avoidance,
                               casadi.DM.zeros(self.n_obstacles, self.n_obstacles)) \
                and self.n_obstacles > 0:
            opt_variables = casadi.vertcat(opt_variables, flat_sl_obs)

        opt_params = casadi.vertcat(flat_z_ref, flat_z_k, flat_u_prev)

        if self.n_obstacles > 0:
            flat_obstacles = casadi.reshape(
                self.obstacles, 3 * self.n_obstacles * (self.horizon + 1), 1)
            opt_params = casadi.vertcat(opt_params, flat_obstacles, self.ego_radius)

        if self._use_symbolic_weights:
            opt_params = casadi.vertcat(opt_params, self.Q_sym, self.R_sym, self.Qf_sym, self.Rd_sym)

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

            if solver == 'ipopt':
                if solver_type != 'quad':
                    ipopt_options = {
                        'ipopt.print_level': not suppress_output,
                        'ipopt.sb': 'yes',
                        'ipopt.max_iter': 100,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6,
                        'error_on_fail': False,  # to raise an exception if the solver fails to find a solution
                        # "ipopt.linear_solver": "ma27",  # Comment this line if you don't have MA27
                    }
                    solver_options.update(ipopt_options)

                    if self.warmstart:
                        warmstart_options = {
                            'ipopt.warm_start_init_point': 'yes',
                            'ipopt.warm_start_bound_push': 1e-8,
                            'ipopt.warm_start_mult_bound_push': 1e-8,
                            'ipopt.mu_init': 1e-5,
                            'ipopt.bound_relax_factor': 1e-9,
                        }
                        solver_options.update(warmstart_options)

                else:
                    use_nlp_interface_for_qp = True  # ipopt requires the nlp interface
                    if use_nlp_interface_for_qp:
                        # see https://github.com/casadi/casadi/wiki/FAQ:-how-to-use-IPOPT-as-QP-solver%3F
                        ipopt_quad_settings = {
                            'qpsol': 'nlpsol',
                            'qpsol_options': {}
                        }
                        ipopt_quad_settings['qpsol_options']['nlpsol'] = 'ipopt'
                        ipopt_quad_settings['qpsol_options']['nlpsol_options'] = {}
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt'] = {}
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.tol'] = 1e-12
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.tiny_step_tol'] = 1e-20
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.fixed_variable_treatment'] = 'make_constraint'
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.hessian_constant'] = 'yes'
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.jac_c_constant'] = 'yes'
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.jac_d_constant'] = 'yes'
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.accept_every_trial_step'] = 'yes'
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.mu_init'] = 1e-5  # 1e-3
                        # ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.linear_solver'] = 'ma27'
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.sb'] = 'yes'
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.max_iter'] = 100
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.acceptable_tol'] = 1e-8
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.acceptable_obj_change_tol'] = 1e-6
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['ipopt.print_level'] = 0
                        ipopt_quad_settings['qpsol_options']['nlpsol_options']['print_time'] = False
                        solver_options.update(ipopt_quad_settings)
                    else:
                        raise NotImplementedError(
                            f"Solver {solver}, with optimization type: {solver_type} "
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
                print(f"solver: {solver} not valid. Defaulting to {quad_solver}")

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
                    "qpsol": "nlpsol" if solver == "ipopt" else solver,
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
                        if "ccache" in str(e) or "No such file or directory" in e.stderr:
                            cmd_args.pop(0)
                            result = subprocess.run(cmd_args, check=True, capture_output=True)

                    # Create a new NLP solver instance from the compiled code
                    solver = casadi.nlpsol("solver", solver, "./casadi_nlp.so")

        else:
            nlp_solver = solver
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
                    if "ccache" in str(e) or "No such file or directory" in e.stderr:
                        cmd_args.pop(0)
                        result = subprocess.run(cmd_args, check=True, capture_output=True)

                # Create a new NLP solver instance from the compiled code
                solver = casadi.nlpsol("solver", nlp_solver, "./casadi_nlp.so")

        return solver, opt_variables, opt_params, g

    def solve(self):
        """Call update method before this."""
        st = time.process_time()

        sl_mpc = np.zeros((self.horizon, self.nu))
        sl_obs_mpc = np.zeros((self.horizon, self.n_obstacles))
        try:
            opt_variables = casadi.vertcat(
                casadi.reshape(self.z_dv_value, self.nx * (self.horizon + 1), 1),
                casadi.reshape(self.u_dv_value, self.nu * self.horizon, 1)
            )
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                opt_variables = casadi.vertcat(
                    opt_variables,
                    casadi.reshape(self.sl_dv_value, self.nu * self.horizon, 1))

            if not casadi.is_equal(self.P_obstacle_avoidance,
                                   casadi.DM.zeros(self.n_obstacles, self.n_obstacles)) \
                    and self.n_obstacles > 0:
                opt_variables = casadi.vertcat(
                    opt_variables,
                    casadi.reshape(self.sl_obs_dv_value,
                                   self.n_obstacles * (self.horizon + 1), 1))

            opt_parameters = casadi.vertcat(
                casadi.reshape(self.z_ref_value, self.nx * (self.horizon + 1), 1),
                casadi.reshape(self.z_k_value, self.nx, 1),
                casadi.reshape(self.u_prev_value, self.nu, 1)
            )

            if self.n_obstacles > 0:
                opt_parameters = casadi.vertcat(
                    opt_parameters,
                    casadi.reshape(self.obstacles_value,
                                   3 * self.n_obstacles * (self.horizon + 1), 1),
                    self.ego_radius_value
                )
            if self._use_symbolic_weights:
                opt_parameters = casadi.vertcat(
                    opt_parameters,
                    casadi.DM(self.Q_diag_value),
                    casadi.DM(self.R_diag_value),
                    casadi.DM(self.Qf_diag_value),
                    casadi.DM(self.Rd_diag_value),
                )

            dual_variables = {}
            if self.warmstart:
                if self.lam_x_value is not None:
                    dual_variables['lam_x0'] = self.lam_x_value
                if self.lam_g_value is not None:
                    dual_variables['lam_g0'] = self.lam_g_value

            result_dict = self.solver(x0=opt_variables, p=opt_parameters,
                                      lbg=self.lbg, lbx=self.lbx,
                                      ubg=self.ubg, ubx=self.ubx, **dual_variables)
            sol = result_dict['x']
            cost = result_dict['f']
            g = result_dict['g']
            lam_g = result_dict['lam_g']  # lagrange multipliers (constraints)
            lam_x = result_dict['lam_x']  # lagrange multipliers (optimization variables)
            lam_p = result_dict['lam_p']  # lagrange multipliers (parameters)

            # todo: add hessian calculation here for convexity check
            #  (https://www.mathworks.com/help/optim/ug/hessian.html#bsapedt | https://github.com/casadi/casadi/issues/3297#issuecomment-1662226255 | https://github.com/meco-group/rockit/blob/master/rockit/ocp.py | https://groups.google.com/g/casadi-users/c/h05oCge2vkk)

            z_mpc = casadi.reshape(sol[:self.nx * (self.horizon + 1)],
                                   self.nx, self.horizon + 1).full()
            u_mpc = casadi.reshape(
                sol[self.nx * (self.horizon + 1):self.nx * (self.horizon + 1) + self.nu * self.horizon],
                self.nu, self.horizon).full()
            if not casadi.is_equal(self.P_u_rate, casadi.DM.zeros((self.nu, self.nu))):
                sl_mpc = casadi.reshape(
                    sol[self.nx * (self.horizon + 1) + (self.nu * self.horizon):(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon))],
                    self.nu, self.horizon).full()

            if not casadi.is_equal(self.P_obstacle_avoidance,
                                   casadi.DM.zeros(self.n_obstacles, self.n_obstacles)) \
                    and self.n_obstacles > 0:
                sl_obs_mpc = casadi.reshape(
                    sol[(self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon)):(
                        self.nx * (self.horizon + 1) + (self.nu * self.horizon) + (self.nu * self.horizon))
                        + (self.n_obstacles * (self.horizon + 1))],
                    self.n_obstacles, self.horizon + 1).full()

            self.update_u_rate(self.u_prev_value, u_mpc)
            u_rate = self.u_rate_dv_value

            z_ref = casadi.reshape(
                opt_parameters[:self.nx * (self.horizon + 1)],
                self.nx, self.horizon + 1).full()
            z_k_value = casadi.reshape(
                opt_parameters[self.nx * (self.horizon + 1):self.nx * (self.horizon + 1) + self.nx],
                self.nx, 1).full()
            u_prev_value = casadi.reshape(
                opt_parameters[self.nx * (self.horizon + 1) + self.nx:self.nx * (self.horizon + 1) + self.nx + self.nu],
                self.nu, 1).full()
            if self.n_obstacles > 0:
                obstacles_value = casadi.reshape(
                    opt_parameters[self.nx * (self.horizon + 1) + self.nx + self.nu:
                                   self.nx * (self.horizon + 1) + self.nx + self.nu + (3 * self.n_obstacles * (self.horizon + 1))],
                    3 * self.n_obstacles, self.horizon + 1).full()
                ego_radius_value = casadi.reshape(
                    opt_parameters[self.nx * (self.horizon + 1) + self.nx + self.nu + (3 * self.n_obstacles * (self.horizon + 1))],
                    1, 1).full().item()

            try:
                iteration_count = self.solver.stats()['iter_count']
            except KeyError:
                # blocksqp plugin does not return iteration count
                iteration_count = -1
            is_opt = True
            success = self.solver.stats()['success']
            return_status = self.solver.stats()['return_status']
            unified_return_status = self.solver.stats()['unified_return_status']
            if not success:
                print(f"Return status: {return_status}, unified return status: {unified_return_status}")
            solve_time = self.solver.stats()['t_wall_total']

            # Useful for debugging; keep commented unless actively debugging.
            initial_condition_constraint = casadi.reshape(g[:self.nx], self.nx, 1)
            model_dynamics = casadi.reshape(g[self.nx:self.nx + (self.nx * self.horizon)], self.nx, self.horizon)
            u_rate_i = casadi.reshape(
                g[self.nx + (self.nx * self.horizon):self.nx + (self.nx * self.horizon) + self.nu * self.horizon],
                self.nu, -1).full()
            u_rate_ii = casadi.reshape(
                g[self.nx + (self.nx * self.horizon) + self.nu * self.horizon:self.nx + (self.nx *
                    self.horizon) + (self.nu * self.horizon) + (self.nu * self.horizon)],
                self.nu, -1).full()

            if self.n_obstacles > 0:
                if casadi.is_equal(self.P_obstacle_avoidance,
                                   casadi.DM.zeros(self.n_obstacles, self.n_obstacles)):
                    self.update_obstacle_distances(
                        z_k_value, z_mpc, self.obstacles_value, self.ego_radius_value)
                else:
                    self.update_obstacle_distances(
                        z_k_value, z_mpc, self.obstacles_value, self.ego_radius_value, sl_obs_mpc)

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
            u_mpc = np.zeros((self.nu, self.horizon))
            z_mpc = np.zeros((self.nx, self.horizon + 1))
            u_rate = np.zeros((self.nu, self.horizon))
            z_ref = casadi.reshape(
                opt_parameters[:self.nx * (self.horizon + 1)],
                self.nx, self.horizon + 1).full()
            z_k_value = casadi.reshape(
                opt_parameters[self.nx * (self.horizon + 1):self.nx * (self.horizon + 1) + self.nx],
                self.nx, 1).full()
            _up_start = self.nx * (self.horizon + 1) + self.nx
            u_prev_value = casadi.reshape(
                opt_parameters[_up_start:_up_start + self.nu],
                self.nu, 1).full()
            iteration_count = None
            is_opt = False
            success = False
            lam_g = np.zeros(self.lbg.shape)
            lam_x = np.zeros_like(opt_parameters)
            lam_p = np.zeros_like(opt_parameters)
            solve_time = time.process_time() - st

        sol_dict = {'u_control': u_mpc[:, 0],
                    'u_mpc': u_mpc,
                    'z_mpc': z_mpc,
                    'u_rate': u_rate,
                    'u_prev': u_prev_value,
                    'sl_mpc': sl_mpc,
                    'sl_obs_mpc': sl_obs_mpc,
                    'z_ref': z_ref.T,
                    'optimal': is_opt,
                    'solve_time': solve_time,
                    'iter_count': iteration_count,
                    'solver_status': success,
                    'lam_x': lam_x,
                    'lam_g': lam_g,
                    'lam_p': lam_p,
                    'solver_stats': self.solver.stats()
                    }

        self.solution_dict = sol_dict
        return sol_dict


if __name__ == '__main__':
    dt = 0.0001
    x0 = np.array([[1., 0.3, 3, 0.1]]).T
    u0 = np.array([[1., 0.3]]).T
    kin_mpc = DiscreteKinematicMPCCasadi(vehicle=None, horizon=25, sample_time=dt, wheelbase=0.256,
                                         nx=4, nu=2, x0=x0, u0=u0,
                                         Q=(1e-1, 1e-8, 1e-8, 1e-8), R=(1e-3, 5e-3),
                                         Qf=(0.0, 0.0, 0.0, 0.0), Rd=(0.0, 0.0),
                                         vel_bound=(-5.0, 5.0),
                                         delta_bound=(-np.radians(23.0), np.radians(23.0)),
                                         acc_bound=(-3.0, 3.0),
                                         jerk_bound=(-1.5, 1.5),
                                         delta_rate_bound=(-np.radians(352.9411764706), np.radians(352.9411764706)),
                                         symbol_type='SX', suppress_ipopt_output=True)
    solution_dictionary = kin_mpc.solve()
    print(solution_dictionary)