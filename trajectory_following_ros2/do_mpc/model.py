"""
Sets up the vehicle model.
"""

import numpy as np
import casadi
from casadi import cos, sin, tan, fmod
import do_mpc


class BicycleKinematicModel(object):
    """docstring for ClassName"""
    
    def __init__(self, wheelbase, width, sample_time, model_type='continuous'):
        """Constructor for BicycleKinematicModel"""
        # car parameters
        self.wheelbase = wheelbase  # wheelbase
        self.width = width

        # # reference path
        # self.reference_path = reference_path

        # # waypoint
        # self.wp_id = 0

        # if self.reference_path is not None:
        #     self.current_waypoint = [self.reference_path[self.wp_id, 0], self.reference_path[self.wp_id, 1],
        #                              self.reference_path[self.wp_id, 2], self.reference_path[self.wp_id, 3], 0]

        # model
        self.Ts = sample_time
        if model_type == 'continuous':
            self.model = self.model_setup()
        else:
            self.model = self.discrete_model_setup()


    def model_setup(self):
        model_type = "continuous"  # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type)

        # States struct (optimization variables):
        pos_x = model.set_variable(var_type="_x", var_name="pos_x")  # Vehicle global X position
        pos_y = model.set_variable(var_type="_x", var_name="pos_y")  # Vehicle global Y position
        vel = model.set_variable(var_type="_x", var_name="vel")  # Vehicle longitudinal velocity
        psi = model.set_variable(var_type="_x", var_name="psi")  # Vehicle global yaw angle
        # e_y = model.set_variable(var_type="_x", var_name="e_y")  # Vehicle lateral error

        # Input struct (optimization variables), i.e control inputs:
        acc = model.set_variable(var_type="_u", var_name="acc")  # vehicle acceleration
        # u_throttle = model.set_variable('_u', 'car_v')
        delta = model.set_variable(var_type="_u", var_name="delta")  # steering angle in radians

        # reference data. todo: move away from model and into MPC setup
        # using time-varing parameters data type
        x_ref = model.set_variable(var_type="_tvp", var_name="x_ref")
        y_ref = model.set_variable(var_type="_tvp", var_name="y_ref")
        vel_ref = model.set_variable(var_type="_tvp", var_name="vel_ref")
        psi_ref = model.set_variable(var_type="_tvp", var_name="psi_ref")
        # ey_lb = model.set_variable(var_type="_tvp", var_name="ey_lb")
        # ey_ub = model.set_variable(var_type="_tvp", var_name="ey_ub")

        # tracking errors (optimization variables):
        psi_diff = (fmod(psi - psi_ref + np.pi, 2 * np.pi) - np.pi)  # [-pi, pi)
        model.set_expression('psi_diff', psi_diff)

        model.set_rhs("pos_x", vel * cos(psi))
        model.set_rhs("pos_y", vel * sin(psi))
        model.set_rhs("vel", acc)
        model.set_rhs("psi", vel * tan(delta) / self.wheelbase)
        # model.set_rhs("e_y", vel * sin(psi_diff))
        # model.set_rhs("e_psi", vel / self.wheelbase * tan(delta))

        model.setup()

        return model

    def discrete_model_setup(self):
        model_type = "discrete"  # either 'discrete' or 'continuous'
        model = do_mpc.model.Model(model_type)

        # _x = model.set_variable(var_type='_x', var_name='x', shape=(4, 1))
        # _u = model.set_variable(var_type='_u', var_name='u', shape=(2, 1))

        # States struct (optimization variables):
        pos_x = model.set_variable(var_type="_x", var_name="pos_x")  # Vehicle global X position
        pos_y = model.set_variable(var_type="_x", var_name="pos_y")  # Vehicle global Y position
        vel = model.set_variable(var_type="_x", var_name="vel")  # Vehicle longitudinal velocity
        psi = model.set_variable(var_type="_x", var_name="psi")  # Vehicle global yaw angle
        # e_y = model.set_variable(var_type="_x", var_name="e_y")  # Vehicle lateral error
        _x = casadi.vertcat(pos_x, pos_y, vel, psi)

        # Input struct (optimization variables), i.e control inputs:
        acc = model.set_variable(var_type="_u", var_name="acc")  # vehicle acceleration
        # u_throttle = model.set_variable('_u', 'car_v')
        delta = model.set_variable(var_type="_u", var_name="delta")  # steering angle in radians
        _u = casadi.vertcat(acc, delta)

        # reference data.
        # using time-varing parameters data type
        x_ref = model.set_variable(var_type="_tvp", var_name="x_ref")
        y_ref = model.set_variable(var_type="_tvp", var_name="y_ref")
        vel_ref = model.set_variable(var_type="_tvp", var_name="vel_ref")
        psi_ref = model.set_variable(var_type="_tvp", var_name="psi_ref")
        # ey_lb = model.set_variable(var_type="_tvp", var_name="ey_lb")
        # ey_ub = model.set_variable(var_type="_tvp", var_name="ey_ub")

        # tracking errors (optimization variables):
        psi_diff = (fmod(psi - psi_ref + np.pi, 2 * np.pi) - np.pi)
        model.set_expression('psi_diff', psi_diff)

        # A = casadi.blockcat([
        #     [1, 0, cos(_x[3]) * self.Ts, -_x[2] * sin(_x[3]) * self.Ts],
        #     [0, 1, sin(_x[3]) * self.Ts, _x[2] * cos(_x[3]) * self.Ts],
        #     [0, 0, 1, 0],
        #     [0, 0, self.Ts * tan(_u[1]) / self.wheelbase, 1],
        # ])
        # B = casadi.blockcat([
        #     [0, 0],
        #     [0, 0],
        #     [self.Ts, 0],  #
        #     [0, self.Ts * _x[2] / (self.wheelbase * cos(_u[1]) ** 2)],
        # ])
        A = casadi.blockcat([
            [1, 0, cos(psi) * self.Ts, -vel * sin(psi) * self.Ts],
            [0, 1, sin(psi) * self.Ts, vel * cos(psi) * self.Ts],
            [0, 0, 1, 0],
            [0, 0, self.Ts * tan(delta) / self.wheelbase, 1],
        ])
        B = casadi.blockcat([
            [0, 0],
            [0, 0],
            [self.Ts, 0],  #
            [0, self.Ts * vel / (self.wheelbase * cos(delta) ** 2)],
        ])
        G = casadi.vertcat(vel * sin(psi) * psi * self.Ts,
                           -vel * cos(psi) * psi * self.Ts,
                           0,
                           -(vel * delta) * self.Ts / (self.wheelbase * cos(delta) ** 2))
        x_next = A @ _x + B @ _u + G

        model.set_rhs("pos_x", x_next[0])
        model.set_rhs("pos_y", x_next[1])
        model.set_rhs("vel", x_next[2])
        model.set_rhs("psi", x_next[3])
        # model.set_rhs("e_y", vel * sin(psi_diff))
        # model.set_rhs("e_psi", vel / self.wheelbase * tan(delta))

        # model.set_rhs('x', x_next)

        model.setup()

        return model
