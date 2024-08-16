"""
Configures the DAE/ODE/discrete simulator
Sources:
    https://github.com/yahsiuhsieh/model-predictive-control/blob/master/src/simulator.py
"""

import numpy as np
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches


import pdb
import sys

import do_mpc

# Colors
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'


class Simulator(object):
    """docstring for ClassName"""
    
    def __init__(self, vehicle, sample_time=0.02):
        """Constructor for Simulator"""
        self.vehicle = vehicle
        self.simulator = do_mpc.simulator.Simulator(vehicle.model)

        # provide time-varing parameters: setpoints/references
        self.tvp_template = self.simulator.get_tvp_template()
        self.simulator.set_tvp_fun(self.tvp_fun)

        # Setup the parameters for the simulator
        simulator_params = {
            'integration_tool': 'cvodes',
            'abstol': 1e-10,
            'reltol': 1e-10,
            't_step': sample_time
        }
        self.simulator.set_param(**simulator_params)

        self.simulator.setup()

    def tvp_fun(self, t_now):
        return self.tvp_template

    #################################################################
    #                                                               #
    #   This plotting function is cited from:                       #
    #   the file "spatial_bicycle_models.py" of matssteinweg, ZTH   #
    #   to use the simulator created by the author                  #
    #                                                               #
    #################################################################
    def show(self, states):
        '''
        Display car on current axis.
        '''
        x, y, psi = states[0], states[1], states[2]

        # Get car's center of gravity
        cog = (x, y)
        # Get current angle with respect to x-axis
        yaw = np.rad2deg(psi)
        # Draw rectangle
        car = plt_patches.Rectangle(
                cog,
                width=self.vehicle.length,
                height=self.vehicle.width,
                angle=yaw,
                facecolor=CAR,
                edgecolor=CAR_OUTLINE,
                zorder=20,
        )

        # Shift center rectangle to match center of the car
        car.set_x(
                car.get_x()
                - (
                        self.vehicle.length / 2 * np.cos(psi)
                        - self.vehicle.width / 2 * np.sin(psi)
                )
        )
        car.set_y(
                car.get_y()
                - (
                        self.vehicle.width / 2 * np.cos(psi)
                        + self.vehicle.length / 2 * np.sin(psi)
                )
        )

        # Add rectangle to current axis
        ax = plt.gca()
        ax.add_patch(car)
        # plt.plot(car)
