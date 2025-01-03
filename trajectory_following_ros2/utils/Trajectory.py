"""
Steps:
    (optional) order waypoints to make sure they're increasing. This might lead to bad behaviours, especially with loops. I don't think this is necessary
    (optional: run once) Get dt between waypoints (https://gitlab.com/autowarefoundation/autoware.ai/core_planning/-/blob/a056c3908676e3bf8f39daadb1699ab946dd55ca/mpc_follower/src/mpc_utils.cpp#L288)
    (optional) Keep track of indices traversed, e.g check purepursuit waypoint > lookahead. This might disable reversing
    Get current robot position and orientation (rear axle. Convert to rear axle if not). Note this should be done by the robot, not this trajectory class
    (optional) calculate trajectory yaw from xy and compare with actual yaw (https://gitlab.com/autowarefoundation/autoware.ai/core_planning/-/blob/a056c3908676e3bf8f39daadb1699ab946dd55ca/mpc_follower/src/mpc_utils.cpp#L150)
    (optional) calculate curvature
    (optional) setup conversion to and from the frenet frame
    Get current robot velocity and optionally acceleration
    Normalize yaw error, e.g from Pi 2 pi
    Add the following flags
        * traversed threshold: a distance to a waypoint to mark it as a success
        * car radius: a radius (or other metric) used to mark waypoints as traversed. Could be merged with the above. Similar to minimum lookahead distance. Can be used to avoid selecting points behind the car
        * ignore waypoints behind: if we missed waypoints due to bad tracking, do we try to go back to them or do we ignore/skip them and keep moving forward
        * max distance (or index search radius): to prevent searching for points outside of a radius. Similar to minimum lookahead distance.
    (optional) Get robot/car direction from transformation to know waypoints direction (could convert to frenet frame first). Do later as might be unnecessary
    Get closest waypoint to the current robot position and orientation considering a search radius to limit (i.e one around the car to exclude and one defining the maximum radius to search)
        the number of waypoints searched. (might need to do some transformation first).
        Note that the closest waypoint may be behind the robot or due to delay might also be the same as the robots
        position or fall behind.
        * using L2 norm
        * using kdtree
    Calculate reference trajectory with:
        velocity (current/desired),
        acceleration,
        vehicle state/kinematics (optional)
        (horizon) points,
        sample time and
        prediction horizon
    (optional but recommended) Interpolate/resample based on time, position, velocity, acceleration. E.g using moving average or spline. Might be done before xref generation
    Reject points which are too close or duplicates. Should be done in interpolation step.
    Send to MPC

See:
    To get nearest pose: https://gitlab.com/autowarefoundation/autoware.ai/core_planning/-/blob/a056c3908676e3bf8f39daadb1699ab946dd55ca/mpc_follower/src/mpc_utils.cpp#L311
                         https://gitlab.com/autowarefoundation/autoware.ai/core_planning/-/blob/a056c3908676e3bf8f39daadb1699ab946dd55ca/mpc_follower/src/mpc_utils.cpp#L352
                         https://github.com/autowarefoundation/autoware.universe/blob/main/control/mpc_lateral_controller/src/mpc_utils.cpp

                         https://github.com/berlala/genesis_path_follower/blob/master/scripts/gps_utils/ref_gps_traj.py#L120C20-L120C20
    For path smoothing:
        * https://github.com/mlab-upenn/mpcc/blob/master/scripts/acados/Bezier.py
Todo:
    (optional) smooth/interpolate x, y, yaw
    setup curvature from x, y, yaw
    (optional) smooth yaw
    (optional) create speed profile
    add option for different integration methods
"""

from copy import deepcopy
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal, fft, ndimage, spatial

import trajectory_following_ros2.utils.filters as filters
import trajectory_following_ros2.utils.trajectory_utils as trajectory_utils


class Trajectory(object):
    """docstring for ClassName"""

    def __init__(self, search_index_number=10, goal_tolerance=0.2, stop_speed=0.5 / 3.6):
        """Constructor for Trajectory"""
        self.state_keys = ['x', 'y', 'speed', 'yaw', 'omega',
                           'curvature', 'cum_dist', 'vy', 'omega', 'acc']
        self.state_key_to_column = {key: i for i, key in enumerate(self.state_keys)}

        self.trajectory_keys = ['x', 'y', 'speed', 'yaw', 'omega',
                                'curvature', 'cum_dist', 'dt', 'total_time_elapsed']
        self.trajectory_key_to_column = {key: i for i, key in enumerate(self.trajectory_keys)}

        # x, y, vel, yaw, curvature, cum_dist, vy, omega, acc
        self.state = np.zeros((1, len(self.state_keys)), dtype=float)
        self.state_error = np.zeros(())  # long_error, lat, psi.

        self.previous_index = 0
        self.current_index = 0
        self.goal_index = 0
        # x, y, vel, yaw, omega, curvature, cum_dist, dt, total_time_elapsed
        self.goal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.trajectory = None  # n, 2. todo: remove duplicates
        self.waypoint_kdtree = None

        self.dt = 0.0

        if self.trajectory is not None:
            self.waypoint_kdtree = trajectory_utils.generate_kd_tree(self.trajectory)

        # self.N_IND_SEARCH = search_index_number  # Search index number
        self.GOAL_DIS = goal_tolerance  # goal distance (stop permitted when dist to goal < dist_stop)
        self.MAXIMUM_SEARCH_RADIUS = 25.0  # maximum search radius (m)
        self.STOP_SPEED = stop_speed  # stop speed (stop permitted when speed < speed_stop)

    def calc_nearest_index(self, waypoints=None, state=None, current_index=None, num_neighbours=10,
                           min_search_radius=0.0, max_search_radius=30.0, use_euclidean_distance=True, workers=1):
        if waypoints is None:
            waypoints = self.trajectory[:, [self.trajectory_key_to_column['x'], self.trajectory_key_to_column['y']]]

        if state is None:
            state = self.state[:, [self.state_key_to_column['x'], self.state_key_to_column['y']]]

        if current_index is None:
            current_index = self.current_index

        if (min_search_radius is None) or (min_search_radius < 0.0):
            min_search_radius = self.GOAL_DIS

        if (max_search_radius is None) or (max_search_radius < 0.0):
            max_search_radius = self.MAXIMUM_SEARCH_RADIUS

        if self.waypoint_kdtree is None:
            self.waypoint_kdtree = trajectory_utils.generate_kd_tree(waypoints)
        indices, distances = trajectory_utils.find_closest_waypoints(waypoints[:, :2], state,
                                                                     current_index=current_index,
                                                                     num_neighbours=num_neighbours,
                                                                     min_search_radius=min_search_radius,
                                                                     max_search_radius=max_search_radius,
                                                                     use_euclidean_distance=use_euclidean_distance,
                                                                     workers=workers)

        return indices, distances

    def get_next_waypoint(self, position=None, target_point=None, lookahead=10.0, full_line=True):
        if position is None:
            position = self.state[:, [self.state_key_to_column['x'], self.state_key_to_column['y']]]

        if target_point is None:
            # todo: test this
            target_point = self.goal[:, [self.trajectory_key_to_column['x'], self.trajectory_key_to_column['y']]]

        # center the point around the robot
        center = position[:, :2] - position[:, :2]
        target = target_point[:, :2] - position[:, :2]

        # intersections = trajectory_utils.circle_line_segment_intersection(
        #         (0, 0), lookahead,
        #         center.flatten().tolist(), target.flatten().tolist(),
        #         full_line=full_line)
        # intersections = trajectory_utils.circle_line_segment_intersection2((0, 0), lookahead,
        #                                                                    center.flatten().tolist(),
        #                                                                    target.flatten().tolist(),
        #                                                                    full_line=full_line)
        intersections = trajectory_utils.circleSegmentIntersection(center.flatten().tolist(),
                                                                   target.flatten().tolist(), lookahead)
        if len(intersections) == 0:
            intersections = trajectory_utils.closestPointOnLine(center.flatten().tolist(), target.flatten().tolist())

        # restore the point to the global frame
        if len(intersections) > 0:
            intersections = np.array(intersections) + position[:, :2]
        return intersections

    def calc_ref_trajectory(self, state, trajectory, current_index, dt, prediction_horizon, lookahead_time=1.0, lookahead=15.0,
                            num_points_to_interpolate=0, target_speed=None):
        if state is None:
            state = self.state

        if trajectory is None:
            # todo: set as self.trajectory[current_index:, :]
            trajectory = self.trajectory

        if current_index is None:
            current_index = self.current_index

        if num_points_to_interpolate == 0 or num_points_to_interpolate is None:
            num_points_to_interpolate = prediction_horizon

        # todo: interpolate the trajectory (path, speed, yaw, dt, curvature)
        # notes
        # * the current index/goal point should only be updated once we reach a threshold

        # find the closest waypoint to the robot based on conditions
        # ind, distances = self.calc_nearest_index(waypoints=trajectory, state=state, current_index=self.current_index,
        #                                  num_neighbours=10, min_search_radius=self.GOAL_DIS, max_search_radius=self.MAXIMUM_SEARCH_RADIUS,
        #                                  use_euclidean_distance=True, workers=1)

        ind, distances = self.calc_nearest_index(waypoints=None, state=None, current_index=current_index,
                                                 num_neighbours=10, min_search_radius=self.GOAL_DIS,
                                                 max_search_radius=self.MAXIMUM_SEARCH_RADIUS,
                                                 use_euclidean_distance=True, workers=1)

        try:
            target_index = ind[0]  # returns None if at the end of the trajectory
        except IndexError:
            return current_index, None, None, None, None

        # find the next waypoint using the line intersection formula and compare to the above then pick one
        goal_waypoint0 = trajectory[target_index, [self.trajectory_key_to_column['x'], self.trajectory_key_to_column['y']]]  # trajectory[target_index, :]
        goal_waypoint1 = self.get_next_waypoint(position=state, target_point=trajectory[target_index, :].reshape(1, -1),
                                                lookahead=lookahead,
                                                full_line=True)

        goal_waypoint2 = trajectory_utils.get_target_point(
                lookahead=lookahead, circle_center=state[:, :2].flatten().tolist(),
                polyline=trajectory[:, [self.trajectory_key_to_column['x'], self.trajectory_key_to_column['y']]],
                method=2, full_line=True)

        # shift the waypoints to the rear axle
        temp_waypoints = trajectory[:, [self.trajectory_key_to_column['x'], self.trajectory_key_to_column['y']]] - state[:, [self.state_key_to_column['x'], self.state_key_to_column['y']]]
        intersection_temp = trajectory_utils.get_target_point(lookahead=lookahead, circle_center=(0, 0),
                                                           polyline=temp_waypoints[:, [self.trajectory_key_to_column['x'],
                                                                                   self.trajectory_key_to_column['y']]],
                                                           method=1, full_line=False)
        goal_waypoint3 = None
        if intersection_temp is not None:
            goal_waypoint3 = np.array(intersection_temp) + state[:,
                                                           [self.state_key_to_column['x'],
                                                            self.state_key_to_column['y']]]

        # get errors (x, y, psi). todo: do in mpc/purepursuit node
        psi_desired = trajectory[target_index, self.trajectory_key_to_column['yaw']]
        psi_current = state[:, self.state_key_to_column['yaw']]
        crosstrack_err, psi_error, lat_error, long_error = trajectory_utils.get_errors(
                state[:, [self.state_key_to_column['x'], self.state_key_to_column['y']]],
                trajectory[ind, :2].reshape(1, -1),
                psi_current,
                psi_desired
        )

        # get curvature. todo: do in mpc/purepursuit node

        # optionally convert to another reference frame, e.g Frenet, curvilinear or could do after smoothing/interpolation
        # https://github.com/berlala/genesis_path_follower/blob/master/scripts/gps_utils/ref_gps_traj.py#L125
        # the equation in the above link is the same as Rotation.from_euler('z', psi).inv().as_matrix()

        # find the reference 'horizon' points from the next waypoint based on distance, velocity, time, open loop prediction
        # e.g have a lookahead time (or prediction horizon) temp and find waypoints by adding the dt b/w waypoints
        # algorithm
        # step 1 (could be done once upon initialization or in each time step): compute 'dt' between waypoints using speed and distance
        # step 1 extra: could also have a lookahead distance and get dts based on speed so the prediction time is dynamic and scales based on current speed
        # step 1 extra extra: could also generate a velocity profile if the waypoints do not container dt, timestamps or velocities
        # step 2: accumulate next waypoints up to the prediction time
        # step 3: interpolate to get 'horizon' points

        # optionally, interpolate the distances
        # return the next idx, openloop prediction, (x, y, psi, vel, acc, etc) x horizon

        # interpolate the path around the current state
        interpolation_min = current_index
        interpolation_max = min(target_index + 10, len(trajectory) - 1)
        try:
            smooth_path = filters.smooth_and_interpolate_coordinates(
                    coordinates=trajectory[interpolation_min:interpolation_max,
                                [self.trajectory_key_to_column['x'], self.trajectory_key_to_column['y']]],
                    method='bspline', polynomial_order=3,
                    weight_smooth=0.1)
        except TypeError:
            # sometimes the interpolation fails if there are only <= 3 points due to duplicates
            pass

        # # todo: use this to add extra points for the reference trajectory
        # smooth_path = trajectory_utils.interpolate_trajectory(trajectory[interpolation_min:interpolation_max,
        #                     [self.trajectory_key_to_column['x'], self.trajectory_key_to_column['y']]],
        #                                                       num_points=prediction_horizon,
        #                                                       weight_smooth=0.5, polynomial_order=3)

        # below modifies the original trajectory if None was passed
        # trajectory[interpolation_min:interpolation_max, [self.trajectory_key_to_column['x'],
        #                                                  self.trajectory_key_to_column['y']]] = smooth_path

        # generate the reference trajectory
        reference_trajectory = trajectory_utils.generate_reference_trajectory_by_interpolation(
                trajectory, state,
                target_index,
                self.trajectory_key_to_column,
                horizon=prediction_horizon,
                v_target=target_speed, dt=dt)

        return target_index, reference_trajectory, goal_waypoint1, goal_waypoint2, goal_waypoint3

    def check_goal(self, x, y, vel, goal, tind, nind):
        """
        Todo:
            * refactor speed stopping.
            * enable looping
        :param state:
        :param goal:
        :param tind: target index
        :param nind: number of indices
        :return:
        """
        # check goal
        # dx = state.x - goal[0]
        # dy = state.y - goal[1]
        # d = math.hypot(dx, dy)
        d = trajectory_utils.get_distance(np.array([x, y]).reshape((1, -1)),
                                          goal[0:2].reshape(1, -1)).item()

        isgoal = (d <= self.GOAL_DIS)

        if abs(tind - nind) >= 5:
            isgoal = False

        # isstop = (abs(vel) <= self.STOP_SPEED)  # todo: refactor
        tolerance = 1.0
        isstop = abs(vel - self.STOP_SPEED) <= tolerance
        isstop = True

        if isgoal and isstop:  # and (tind >= nind)
            return True

        return False
