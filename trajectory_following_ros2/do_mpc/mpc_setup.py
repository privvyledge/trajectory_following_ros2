"""

"""

import numpy as np
# from casadi import *
from casadi import mtimes, vertcat
# from casadi.tools import *
import do_mpc
from trajectory_following_ros2.do_mpc.model import BicycleKinematicModel


class MPC(object):
    """docstring for ClassName"""

    def __init__(self, vehicle, horizon=15, sample_time=0.02, wheelbase=0.256,
                 Q=np.diag([1e-1, 1e-8, 1e-8, 1e-8]), R=np.diag([1e-3, 5e-3]),
                 Qf=np.diag([0.0, 0.0, 0.0, 0.0]), Rd=np.diag([0.0, 0.0]),
                 vel_bound=(-5.0, 5.0), delta_bound=(-23.0, 23.0), acc_bound=(-3.0, 3.0),
                 max_iterations=20, tolerance=1e-6, suppress_ipopt_output=True):
        """Constructor for MPC"""
        self.vehicle = vehicle
        self.model = vehicle.model

        self.horizon = horizon
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.Rd = Rd

        self.Ts = sample_time
        self.length = wheelbase  # todo: set as parameters
        self.width = 0.192  # todo: set as parameters

        self.current_prediction = None

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

        self.mpc.set_param(nlpsol_opts=nlpsol_opts)

        # define the objective function and constraints
        self.objective_function_setup()
        self.constraints_setup(vel_bound=vel_bound, delta_bound=delta_bound, acc_bound=acc_bound, reset=False)

        # provide time-varing parameters: setpoints/references
        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)

        self.mpc.setup()

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
            # Use MA27 linear solver in ipopt for faster calculations:
            # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}  # highly recommended (MA27), 'mumps'
        }
        mpc.set_param(**setup_mpc)
        return mpc
        
    def tvp_fun(self, t_now):
        """
        provides data into time-varying parameters.
        todo: change how reference trajectories are generated, e.g using velocity
        todo: pass trajectories or trajectory generator function to initializer and call here
        """
        # # Method 1
        # for k in range(self.horizon):
        #     # extract information from current waypoint
        #     current_waypoint = self.vehicle.reference_path[
        #             self.vehicle.wp_id + k
        #     ]
        #     self.tvp_template['_tvp', k, 'x_ref'] = current_waypoint[0]
        #     self.tvp_template['_tvp', k, 'y_ref'] = current_waypoint[1]
        #
        #     if current_waypoint[2] is not None:
        #         self.tvp_template['_tvp', k,
        #                           'vel_ref'] = current_waypoint[2]
        #     else:
        #         self.tvp_template['_tvp', k, 'vel_ref'] = 0
        #
        #     self.tvp_template['_tvp', k, 'psi_ref'] = current_waypoint[3]
        #     # self.tvp_template['_tvp', k, 'ey_lb'] = ey_lb[k]
        #     # self.tvp_template['_tvp', k, 'ey_ub'] = ey_ub[k]

        # method 2
        current_waypoint = self.vehicle.reference_path[self.vehicle.wp_id]
        xk = current_waypoint[0]
        yk = current_waypoint[1]
        vel_k = current_waypoint[2]  # sp[ind] or state.vel
        psi_k = current_waypoint[3]
        try:
            delta = int(self.mpc.data['_u', 'delta'][0])
        except IndexError:
            delta = 0.0

        for k in range(self.horizon + 1):
            self.tvp_template['_tvp', k, 'x_ref'] = xk
            self.tvp_template['_tvp', k, 'y_ref'] = yk
            self.tvp_template['_tvp', k, 'vel_ref'] = vel_k
            self.tvp_template['_tvp', k, 'psi_ref'] = psi_k

            xk += (vel_k * np.cos(psi_k)) * self.Ts
            yk += (vel_k * np.sin(psi_k)) * self.Ts
            vel_k += 0.0 * self.Ts
            psi_k += ((vel_k / self.vehicle.length) * np.tan(delta)) * self.Ts

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

    def distance_update(self, states, s):
        vel, psi = states[2], states[3]

        # Compute velocity along path
        s_dot = vel * np.cos(self.mpc.data['_aux', 'psi_diff'][0])

        # Update distance travelled along reference path
        s += s_dot * self.Ts

        return s


def initialize_mpc_problem(reference_path, horizon=15, sample_time=0.02,
                           Q=None, R=None, Qf=None, Rd=None, wheelbase=0.256,
                           delta_min=-23.0, delta_max=23.0, vel_min=-10.0, vel_max=10.0,
                           ay_max=4.0, acc_min=-3.0, acc_max=3.0, max_iterations=100, tolerance=1e-6, suppress_ipopt_output=True):
    """
    Get configured do-mpc modules:
    """
    # model setup
    Vehicle = BicycleKinematicModel(length=wheelbase, width=0.192, sample_time=sample_time)
    # Vehicle.model_setup()
    Vehicle.reference_path = reference_path
    Vehicle.current_waypoint = [Vehicle.reference_path[Vehicle.wp_id, 0], Vehicle.reference_path[Vehicle.wp_id, 1],
                                Vehicle.reference_path[Vehicle.wp_id, 2], Vehicle.reference_path[Vehicle.wp_id, 3],
                                0]

    Controller = MPC(Vehicle, horizon=horizon, sample_time=sample_time, Q=Q, R=R, Qf=Qf, Rd=Rd, wheelbase=wheelbase,
                     vel_bound=(vel_min, vel_max), delta_bound=(delta_min, delta_max), acc_bound=(acc_min, acc_max),
                     max_iterations=max_iterations, tolerance=tolerance, suppress_ipopt_output=suppress_ipopt_output)

    # Sim = Simulator(Vehicle, sample_time=sample_time)
    Sim = None

    return Vehicle, Controller, Sim
