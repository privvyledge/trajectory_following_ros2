"""
Sets up the vehicle model.
"""

import numpy as np
from casadi import cos, sin, tan, fmod
import do_mpc


class BicycleKinematicModel(object):
    """docstring for ClassName"""
    
    def __init__(self, length, width, sample_time, reference_path=None):
        """Constructor for BicycleKinematicModel"""
        # car parameters
        self.length = length  # wheelbase
        self.width = width

        # reference path
        self.reference_path = reference_path

        # waypoint
        self.wp_id = 0

        if self.reference_path is not None:
            self.current_waypoint = [self.reference_path[self.wp_id, 0], self.reference_path[self.wp_id, 1],
                                     self.reference_path[self.wp_id, 2], self.reference_path[self.wp_id, 3], 0]

        # model
        self.Ts = sample_time
        self.model = self.model_setup()

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
        psi_diff = (fmod(psi - psi_ref + np.pi, 2 * np.pi) - np.pi)
        model.set_expression('psi_diff', psi_diff)

        model.set_rhs("pos_x", vel * cos(psi))
        model.set_rhs("pos_y", vel * sin(psi))
        model.set_rhs("vel", acc)
        model.set_rhs("psi", vel * tan(delta) / self.length)
        # model.set_rhs("e_y", vel * sin(psi_diff))
        # model.set_rhs("e_psi", vel / self.length * tan(delta))

        model.setup()

        return model
