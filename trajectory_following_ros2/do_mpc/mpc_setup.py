"""

"""

import os
import numpy as np
# from casadi import *
import casadi
from casadi import mtimes, vertcat, vertsplit
# from casadi.tools import *
import do_mpc
from trajectory_following_ros2.do_mpc.model import BicycleKinematicModel


class MPC(object):
    """docstring for ClassName"""

    def __init__(self, vehicle, horizon=15, sample_time=0.02, wheelbase=0.256,
                 Q=np.diag([1e-1, 1e-8, 1e-8, 1e-8]), R=np.diag([1e-3, 5e-3]),
                 Qf=np.diag([0.0, 0.0, 0.0, 0.0]), Rd=np.diag([0.0, 0.0]),
                 vel_bound=(-5.0, 5.0), delta_bound=(-23.0, 23.0), acc_bound=(-3.0, 3.0),
                 max_iterations=20, tolerance=1e-6, suppress_ipopt_output=True,
                 warmstart=True, quad_prog_mode=True,
                 compile_mpc=True, jit_compilation=False):
        """Constructor for MPC"""
        self.model = vehicle.model

        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.Rd = Rd

        self.Ts = sample_time
        self.compile = compile_mpc
        self.reference_states = np.zeros((self.horizon + 1, self.Q.shape[0]))
        self.warmstart = warmstart
        self.quad_prog_mode = quad_prog_mode

        self.mpc = self.initialize_mpc(model=self.model, horizon=self.horizon,
                                       timestep=self.Ts, store_full_solution=True)

        nlpsol_opts = {
            'ipopt.max_iter': max_iterations,
            'record_time': True,
            'ipopt.acceptable_obj_change_tol': tolerance,
            # 'ipopt.linear_solver': 'MA27'
        }
        if suppress_ipopt_output:
            nlpsol_opts.update({"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0})

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
            nlpsol_opts.update(warmstart_options)

        if self.quad_prog_mode:
            # use ipopt with some QP optimization settings.
            # Convexification and hessian approximation do not work with Do-MPC as the OCP is non convex and not a QP
            qp_options = {
                # WARNING: DO NOT APPLY convexification to Do MPC problem formulation
                # WARNING. DO NOT USE limited memory or exact hessian approximation with Do MPC. Leave IPOPT default.
                'ipopt.tol': 1e-12,
                'ipopt.tiny_step_tol': 1e-20,
                'ipopt.fixed_variable_treatment': 'make_constraint',
                'ipopt.mu_init': 1e-5  # 1e-3
            }
            nlpsol_opts.update(qp_options)

        self.mpc.set_param(nlpsol_opts=nlpsol_opts)

        # define the objective function and constraints
        self.objective_function_setup()
        self.constraints_setup(vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound, reset=False)

        # provide time-varing parameters: setpoints/references
        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)

        # compilation parameters
        compiler = "ccache gcc"  # Linux (gcc, clang, ccache gcc)  # todo: catch exception if ccache is not installed
        # compiler = "clang"  # OSX
        # compiler = "cl.exe" # Windows
        flags = ["-O3"]  # Linux/OSX. ['O3'] enables the most optimizations
        if compiler == "gcc":
            flags = [
                "-Ofast",
                # enables even more optimization than -O3 but could be less precise/safe but is okay in most situations
                "-march=native",  # optimizes for the specific hardware (CPU) but isn't transferable
                "-fPIC",  # for shared_libraries
            ]  # for performance boost using gcc

        # JIT compilation automatically compiles helper functions as well as the solver.
        # Yields about a 2.5x increase in performance
        # By default, the compiler will be gcc or cl.exe
        jit_options = {
            "jit": True, "compiler": "shell", "jit_temp_suffix": False,
            "jit_options": {
                "flags": flags, "compiler": compiler, "verbose": True,
                "compiler_flags": flags
            },  # use ("compiler": "ccache gcc") if ccache is installed for a performance boost
            'jit_cleanup': False  # True: delete files on shutdown
        }

        if self.compile and jit_compilation:
            nlpsol_opts.update(jit_options)
            self.mpc.set_param(nlpsol_opts=nlpsol_opts)

        self.mpc.setup()

        if self.compile and not jit_compilation:
            # self.mpc.compile_nlp(overwrite=False, cname='nlp.c', libname='nlp.so', compiler_command=None)
            # todo: for now there is a bug in the do_mpc code. Compile using my method in casadi node and modify the solver self.mpc.S
            # this is faster than jit for do_mpc
            self.compile_model(overwrite=True, cname='nlp_do_mpc.c', libname='nlp_do_mpc.so', compiler=compiler, flags=flags)

    def initialize_mpc(self, model, horizon, timestep=0.01, store_full_solution=False):
        """

        :param model:
        :param horizon:
        :param timestep:
        :param store_full_solution:
        :return:
        """
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': horizon,
            'n_robust': 0,  # Robust horizon for robust scenario-tree MPC,
            'open_loop': 0,
            't_step': timestep,
            'state_discretization': 'collocation',  # no other option at the moment
            'collocation_type': 'radau',  # no other option at the moment
            'collocation_deg': 2,
            'collocation_ni': 2,
            'store_full_solution': store_full_solution,  # re
            'store_solver_stats': ['success',
                                   # 'return_status',
                                   't_proc_callback_fun', 't_proc_nlp_f', 't_proc_nlp_g', 't_proc_nlp_grad',
                                   't_proc_nlp_grad_f', 't_proc_nlp_hess_l',
                                   't_proc_nlp_jac_g', 't_proc_S', 't_wall_callback_fun',  # t_proc_{S or total}
                                   't_wall_nlp_f', 't_wall_nlp_g', 't_wall_nlp_grad',
                                   't_wall_nlp_grad_f', 't_wall_nlp_hess_l', 't_wall_nlp_jac_g', 't_wall_total'],
            # Use MA27 linear solver in ipopt for faster calculations:
            # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}  # highly recommended (MA27), 'mumps'
        }
        mpc.set_param(**setup_mpc)
        return mpc
        
    def tvp_fun(self, t_now):
        """
        provides data into time-varying parameters.
        """
        # self.tvp_template['_tvp', 0] = [xk, yk, vel_k, psi_k]
        self.tvp_template['_tvp', :] = np.vsplit(self.reference_states, self.horizon + 1)
        # or self.tvp_template['_tvp', 0:(self.horizon + 1)] = casadi.vertsplit(self.reference_states)

        return self.tvp_template

    def objective_function_setup(self):
        # path following
        lterm = (self.Q[0, 0] * (self.model.x['pos_x'] - self.model.tvp['x_ref']) ** 2
                 + self.Q[1, 1] * (self.model.x['pos_y'] - self.model.tvp['y_ref']) ** 2
                 + self.Q[2, 2] * (self.model.x['vel'] - self.model.tvp['vel_ref']) ** 2
                 + self.Q[3, 3] * (self.model.aux['psi_diff']) ** 2
                 # + self.Q[3, 3] * (self.model.x['psi'] - self.model.tvp['psi_ref']) ** 2
                 + self.R[0, 0] * (self.model.u['acc']) ** 2
                 + self.R[1, 1] * (self.model.u['delta']) ** 2
                 )  # stage/lagrange cost
        mterm = (self.Qf[0, 0] * (self.model.x['pos_x'] - self.model.tvp['x_ref']) ** 2
                 + self.Qf[1, 1] * (self.model.x['pos_y'] - self.model.tvp['y_ref']) ** 2
                 + self.Qf[2, 2] * (self.model.x['vel'] - self.model.tvp['vel_ref']) ** 2
                 # + self.Qf[3, 3] * (self.model.x['psi'] - self.model.tvp['psi_ref']) ** 2
                 )  # terminal/mayer cost

        # lterm = lterm + self.R[0, 0] * (self.model.u['acc'] ** 2) + \
        #         self.R[1, 1] * (self.model.u['delta'] ** 2)   # x'Qx + u'Ru
        # mterm = lterm
        # x_packed = vertcat(
        #         self.model.x['pos_x'] - self.model.tvp['x_ref'],  # self.model.x['pos_x'] | self.model.x['pos_x'] - self.model.tvp['x_ref']
        #         self.model.x['pos_y'] - self.model.tvp['y_ref'],  # self.model.x['pos_y'] | self.model.x['pos_y'] - self.model.tvp['y_ref']
        #         self.model.x['vel'] - self.model.tvp['vel_ref'],  # self.model.x['vel'] | self.model.x['vel'] - self.model.tvp['vel_ref']
        #         self.model.x['psi'] - self.model.tvp['psi_ref']  # self.model.x['psi'] | self.model.x['psi'] - self.model.tvp['psi_ref']
        # )
        # u_packed = vertcat(
        #         self.model.u['acc'],
        #         self.model.u['delta'],
        # )
        # lterm = mtimes([x_packed.T, self.Q, x_packed]) + \
        #         mtimes([u_packed.T, self.R, u_packed])  # x_packed.T @ self.Q @ x_packed + u_packed.T @ self.R @ u_packed
        # mterm = mtimes([x_packed.T, self.Q, x_packed])

        self.mpc.set_objective(lterm=lterm, mterm=mterm)
        self.mpc.set_rterm(acc=self.Rd[0, 0], delta=self.Rd[1, 1])  # input penalty

    def constraints_setup(
            self, vel_bound=None, delta_bound=None, acc_bound=None, reset=False
    ):

        # states constraints
        if delta_bound is None:
            delta_bound = [-23.0, 23.0]

        if vel_bound is None:
            vel_bound = [-10.0, 10.0]

        if acc_bound is None:
            acc_bound = [-3.0, 3.0]

        # self.mpc.bounds['lower', '_x', 'pos_x'] = -np.inf
        # self.mpc.bounds['upper', '_x', 'pos_x'] = np.inf
        # self.mpc.bounds['lower', '_x', 'pos_y'] = -np.inf
        # self.mpc.bounds['upper', '_x', 'pos_y'] = np.inf
        self.mpc.bounds['lower', '_x', 'vel'] = vel_bound[0]
        self.mpc.bounds['upper', '_x', 'vel'] = vel_bound[1]
        # self.mpc.bounds['lower', '_x', 'psi'] = -2 * np.pi
        # self.mpc.bounds['upper', '_x', 'psi'] = 2 * np.pi
        # self.mpc.bounds['lower', '_x', 'e_y'] = -2
        # self.mpc.bounds['upper', '_x', 'e_y'] = 2

        # input constraints
        self.mpc.bounds['lower', '_u', 'acc'] = acc_bound[0]
        self.mpc.bounds['upper', '_u', 'acc'] = acc_bound[1]
        self.mpc.bounds['lower', '_u', 'delta'] = np.radians(delta_bound[0])
        self.mpc.bounds['upper', '_u', 'delta'] = np.radians(delta_bound[1])

        if reset:
            self.mpc.setup()

    def get_control(self, x0):

        # # update current waypoint
        # self.vehicle.get_current_waypoint()

        # solve optization problem
        u0 = self.mpc.make_step(x0)

        return u0

    def update(self, reference):
        # todo: to update reference used by tvp instead of calculating
        self.reference_states = reference
        pass

    def distance_update(self, states, s):
        vel, psi = states[2], states[3]

        # Compute velocity along path
        s_dot = vel * np.cos(self.mpc.data['_aux', 'psi_diff'][0])

        # Update distance travelled along reference path
        s += s_dot * self.Ts

        return s

    def compile_model(self, overwrite=True, cname='nlp_do_mpc.c', libname='nlp_do_mpc.so', compiler='gcc', flags=None):
        # self.mpc.compile_nlp(overwrite=False, cname='nlp.c', libname='nlp.so', compiler_command=None)

        if isinstance(compiler, str):
            compiler = compiler.split()

        if flags is None:
            flags = [
                "-Ofast",
                # enables even more optimization than -O3 but could be less precise/safe but is okay in most situations
                "-march=native",  # optimizes for the specific hardware (CPU) but isn't transferable
                "-fPIC",  # for shared_libraries
            ]  # for performance boost using gcc

        # Only compile if not already compiled:
        if overwrite or not os.path.isfile(libname):
            self.mpc.S.generate_dependencies(cname)

            import subprocess
            # On Windows, use other flags
            cmd_args = compiler + ["-shared"] + flags + [cname, "-o", libname]

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
            # self.mpc.S = casadi.nlpsol("solver", self.mpc.S, libname)
            self.mpc.S = casadi.nlpsol('solver_compiled', 'ipopt', libname, self.mpc.settings.nlpsol_opts)


def initialize_mpc_problem(reference_path, horizon=15, sample_time=0.02,
                           Q=None, R=None, Qf=None, Rd=None, wheelbase=0.256,
                           delta_min=-23.0, delta_max=23.0, vel_min=-10.0, vel_max=10.0,
                           ay_max=4.0, acc_min=-3.0, acc_max=3.0,
                           max_iterations=100, tolerance=1e-6, suppress_ipopt_output=True, model_type='continuous',
                           warmstart=True, quad_prog_mode=True, compile_model=False, jit_compilation=True):
    """
    Get configured do-mpc modules:
    """
    # model setup
    Vehicle = BicycleKinematicModel(wheelbase=wheelbase, width=0.192, sample_time=sample_time, model_type=model_type)
    # Vehicle.model_setup()
    # Vehicle.reference_path = reference_path
    # Vehicle.current_waypoint = [Vehicle.reference_path[Vehicle.wp_id, 0], Vehicle.reference_path[Vehicle.wp_id, 1],
    #                             Vehicle.reference_path[Vehicle.wp_id, 2], Vehicle.reference_path[Vehicle.wp_id, 3],
    #                             0]

    Controller = MPC(Vehicle, horizon=horizon, sample_time=sample_time, Q=Q, R=R, Qf=Qf, Rd=Rd, wheelbase=wheelbase,
                     vel_bound=(vel_min, vel_max), delta_bound=(delta_min, delta_max), acc_bound=(acc_min, acc_max),
                     max_iterations=max_iterations, tolerance=tolerance, suppress_ipopt_output=suppress_ipopt_output,
                     warmstart=warmstart, quad_prog_mode=quad_prog_mode, compile_mpc=compile_model, jit_compilation=jit_compilation)

    # Sim = Simulator(Vehicle, sample_time=sample_time)
    Sim = None

    return Vehicle, Controller, Sim
