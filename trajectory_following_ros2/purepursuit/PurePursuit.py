import math
import numpy as np


class PurePursuit(object):
    """docstring for ClassName"""

    def __init__(self, desired_lookahead=0.2, use_adaptive_lookahead=False, lookahead_gain=0.2,
                 minimum_lookahead=0.1, maximum_lookahead=10.0,
                 wheelbase=0.256, desired_speed=10.0):
        """Constructor for PurePursuit"""
        self.desired_lookahead = desired_lookahead
        self.lookahead = desired_lookahead
        self.use_adaptive_lookahead = use_adaptive_lookahead
        self.lookahead_gain = lookahead_gain
        self.minimum_lookahead = minimum_lookahead
        self.maximum_lookahead = maximum_lookahead

        self.wheelbase = wheelbase
        self.longitudinal_velocity = desired_speed

        self.alpha = 0.0
        self.crosstrack_error = 0.0
        self.turn_radius = 0.0
        self.desired_curvature = 0.0
        self.desired_steering_angle = 0.0
        self.desired_yaw_rate = 0.0

    def get_desired_steering_angle(self, location, target_point, speed=None, in_place=True):
        """"""
        if speed is None:
            speed = self.longitudinal_velocity

        if self.use_adaptive_lookahead:
            self.update_lookahead(speed=speed)

        current_x, current_y, current_psi = location
        goal_x, goal_y, goal_psi = target_point
        '''Calculate the yaw error'''
        # v1 = target_point[0:2] - location[0:2]
        # v2 = [np.cos(current_psi), np.sin(current_psi)]
        # alpha1 = find_angle(v1, v2)
        alpha = math.atan2(goal_y - current_y, goal_x - current_x) - current_psi

        ''' (Optional) Get cross-track error: 
                the lateral distance between the heading vector and the goal point in body frame'''
        crosstrack_error = math.sin(alpha) * self.lookahead
        # crosstrack_error = goal_y - current_y

        '''(Optional) Calculate turn radius, r = 1 / curvature'''
        turn_radius = self.lookahead ** 2 / (2 * abs(crosstrack_error))

        '''Get the curvature, kappa '''
        desired_curvature = (2.0 * math.sin(alpha)) / self.lookahead
        # desired_curvature = (2.0 / (self.lookahead ** 2)) * crosstrack_error  # same as curvature above
        # desired_curvature = 1 / turn_radius  # curvature3 = 2 * abs(y) / Lf**2
        # desired_curvature = 2.0 * abs(crosstrack_error) / self.lookahead ** 2

        '''Get the steering angle, delta'''
        desired_steering_angle = math.atan2(2.0 * self.wheelbase * math.sin(alpha), self.lookahead)
        # desired_steering_angle = math.atan(desired_curvature * self.wheelbase)
        # desired_steering_angle = math.atan2(2.0 * abs(crosstrack_error) * self.wheelbase, self.lookahead ** 2)
        # # desired_steering_angle = math.atan2(2.0 * abs(crosstrack_error), self.lookahead ** 2)  # Might be wrong

        ''' Get the desired yaw rate '''
        desired_yaw_rate = (2.0 * speed * math.sin(alpha)) / self.lookahead

        if in_place:
            self.alpha = alpha
            self.crosstrack_error = crosstrack_error
            self.turn_radius = turn_radius
            self.desired_curvature = desired_curvature
            self.desired_steering_angle = desired_steering_angle
            self.desired_yaw_rate = desired_yaw_rate

        return desired_steering_angle

    def update_lookahead(self, speed=None):
        if speed is None:
            speed = self.longitudinal_velocity
        """Update the lookahead distance based on speed"""
        self.lookahead = self.lookahead_gain * speed + self.desired_lookahead
        self.lookahead = np.clip(self.lookahead, self.minimum_lookahead, self.maximum_lookahead)
