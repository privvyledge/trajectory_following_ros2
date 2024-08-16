"""
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
from scipy import interpolate, signal, fft, ndimage


class Trajectory(object):
    """docstring for ClassName"""

    def __init__(self, search_index_number=10, goal_tolerance=0.2, stop_speed=0.5 / 3.6):
        """Constructor for Trajectory"""
        self.N_IND_SEARCH = search_index_number  # Search index number
        self.GOAL_DIS = goal_tolerance  # goal distance (stop permitted when dist to goal < dist_stop)
        self.STOP_SPEED = stop_speed  # stop speed (stop permitted when speed < speed_stop)

    def getDistance(self, p1, p2):
        """
        Calculate distance
        :param p1: list, point1
        :param p2: list, point2
        :return: float, distance
        """
        distance = np.linalg.norm(p1 - p2, axis=-1)
        return distance

    def calc_nearest_index(self, state, cx, cy, cyaw, pind):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """
        dx = [state.x - icx for icx in cx[pind:(pind + self.N_IND_SEARCH)]]
        dy = [state.y - icy for icy in cy[pind:(pind + self.N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        # d2 = np.hypot(dx, dy)

        mind = min(d)

        ind = d.index(mind) + pind

        mind = math.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind

    def calc_nearest_index2(self, state, cx, cy, cyaw, pind):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point

        todo: use goal_threshold
        todo: when checking for distances, only remove points in front of the current one
        """
        dx = [state.x - icx for icx in cx[pind:(pind + self.N_IND_SEARCH)]]
        dy = [state.y - icy for icy in cy[pind:(pind + self.N_IND_SEARCH)]]

        # d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        d = np.hypot(dx, dy)

        # todo: fix as this is wrong
        ind_in_N = int(np.argmin(d))
        ind = pind + ind_in_N

        # dx = [state.x - icx for icx in cx[pind:(pind + self.N_IND_SEARCH)]]
        # dy = [state.y - icy for icy in cy[pind:(pind + self.N_IND_SEARCH)]]
        # d = np.hypot(dx, dy)

        # todo: set first index as point greater than self.GOAL_DIS
        # d_temp = np.array(d)  # todo: use numpy arrays for the calculation above
        # goal_candidates = np.where(d >= self.GOAL_DIS)[0]
        goal_candidates = np.where(d >= 0.05)[0]

        # ind = pind
        # mind = 0.
        # if len(goal_candidates) > 0:
        #     indices_greater_than_last = np.where(goal_candidates >= pind)[0]
        #     if len(indices_greater_than_last) > 0:
        #         goal_candidates = goal_candidates[indices_greater_than_last]
        #         ind = np.min(goal_candidates)
        #         # d = d_temp[goal_candidates]
        #         mind = d[ind]
        #
        #     else:
        #         ind = pind
        #         mind = 0.
        #
        # else:
        #     ind = pind
        #     mind = 0.

        mind = min(d)
        #
        # ind = d.index(mind) + pind
        #
        mind = math.sqrt(mind)
        #
        # dxl = cx[ind] - state.x
        # dyl = cy[ind] - state.y
        #
        # angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        # if angle < 0:
        #     mind *= -1

        # print(ind, ind1)
        return ind, mind

    def calc_nearest_index3(self, x, y, cx, cy, cyaw, pind, lookahead=0.4):
        pos = [x, y]
        trajectory = np.array([cx, cy]).T
        curr_distances = self.getDistance(pos, trajectory)

        # use points greater than lookahead as heuristic
        goal_candidates = np.where((curr_distances > lookahead))[0]  # or >=
        target_idx = pind
        if len(goal_candidates) > 0:
            indices_greater_than_last = np.where(goal_candidates >= pind)[0]

            if len(indices_greater_than_last) > 0:
                goal_candidates = goal_candidates[indices_greater_than_last]
                target_idx = np.min(goal_candidates)

        #     else:
        #         target_idx = self.last_idx
        #
        # else:
        #     target_idx = self.last_idx

        ind = target_idx
        min_distance = curr_distances[target_idx]
        return ind, min_distance

    def calc_ref_trajectory(self, x, y, cx, cy, cyaw, ck, sp, dl, pind, n_states=4, horizon_length=5, dt=0.2,
                            lookahead=0.4):
        """
        calc reference trajectory in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param state: current information
        :param ref_path: reference path: [x, y, yaw]
        :param sp: speed profile (designed speed strategy)
        :return: reference trajectory
        """
        xref = np.zeros((n_states, horizon_length + 1))
        dref = np.zeros((1, horizon_length + 1))
        ncourse = len(cx)

        ind, _ = self.calc_nearest_index3(x, y, cx, cy, cyaw, pind, lookahead=lookahead)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(horizon_length + 1):
            # travel += abs(state.vel) * dt
            # travel += self.GOAL_DIS
            travel += 0.3  # tested but doesn't really work in real time simulations

            # travel += 1

            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                xref[0, i] = cx[ind + dind]
                xref[1, i] = cy[ind + dind]
                xref[2, i] = sp[ind + dind]
                xref[3, i] = cyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = cx[ncourse - 1]
                xref[1, i] = cy[ncourse - 1]
                xref[2, i] = sp[ncourse - 1]
                xref[3, i] = cyaw[ncourse - 1]
                dref[0, i] = 0.0

        return xref, ind, dref

    def calc_ref_trajectory2(self, state, cx, cy, cyaw, ck, sp, dl, pind, n_states=4, horizon_length=5, dt=0.2):
        """
        calc reference trajectory in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param state: current information
        :param ref_path: reference path: [x, y, yaw]
        :param sp: speed profile (designed speed strategy)
        :return: reference trajectory
        """
        xref = np.zeros((n_states, horizon_length + 1))
        dref = np.zeros((1, horizon_length + 1))
        ncourse = len(cx)

        ind, _ = self.calc_nearest_index3(state, cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        xk = cx[ind]
        yk = cy[ind]
        vel_k = sp[ind]  # sp[ind] or state.vel
        psi_k = cyaw[ind]

        for i in range(horizon_length + 1):
            xref[0, i] = xk
            xref[1, i] = yk
            xref[2, i] = vel_k
            xref[3, i] = psi_k
            dref[0, i] = 0.0

            # todo: add option for different integration methods
            xk += (vel_k * math.cos(psi_k)) * dt
            yk += (vel_k * math.sin(psi_k)) * dt
            vel_k += 0.0 * dt
            psi_k += ((vel_k / 0.256) * math.tan(state.delta)) * dt

        return xref, ind, dref

    def calc_speed_profile(self, cx, cy, cyaw, target_speed):
        """
        design appropriate speed strategy
        :param cx: x of reference path [m]
        :param cy: y of reference path [m]
        :param cyaw: yaw of reference path [m]
        :param target_speed: target speed [m/s]
        :return: speed profile
        """

        speed_profile = [target_speed] * len(cx)
        direction = 1.0  # forward

        # Set stop point
        for i in range(len(cx) - 1):
            dx = cx[i + 1] - cx[i]
            dy = cy[i + 1] - cy[i]

            move_direction = math.atan2(dy, dx)

            if dx != 0.0 and dy != 0.0:
                dangle = abs(self.pi_2_pi(move_direction - cyaw[i]))
                if dangle >= math.pi / 4.0:
                    direction = -1.0
                else:
                    direction = 1.0

            if direction != 1.0:
                speed_profile[i] = -target_speed
            else:
                speed_profile[i] = target_speed

        speed_profile[-1] = 0.0

        return speed_profile

    def check_goal(self, x, y, vel, goal, tind, nind):
        """

        :param state:
        :param goal:
        :param tind: target index
        :param nind: number of indeices
        :return:
        """
        # check goal
        # dx = state.x - goal[0]
        # dy = state.y - goal[1]
        # d = math.hypot(dx, dy)
        d = self.getDistance([x, y], goal[0:2])

        isgoal = (d <= self.GOAL_DIS)

        if abs(tind - nind) >= 5:
            isgoal = False

        isstop = (abs(vel) <= self.STOP_SPEED)

        if isgoal and isstop:
            return True

        return False

    def pi_2_pi(self, angle):
        while (angle > math.pi):
            angle = angle - 2.0 * math.pi

        while (angle < -math.pi):
            angle = angle + 2.0 * math.pi

        return angle

    def smooth_yaw(self, yaw):

        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]

            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

        return yaw

    def smooth_and_interpolate_coordinates(self, coordinates, method='rgds',
                                           window_length=9, kernel_type='gaussian', polynomial_order=3, std_dev=2,
                                           weight_data=0.5, weight_smooth=0.5, tolerance=0.000001):
        unique, indices = np.unique(coordinates, axis=0, return_index=True)
        coordinates = coordinates[indices, :]
        x, y = coordinates[:, 0], coordinates[:, 1]
        # l = len(x)
        # t = np.linspace(0, 1, l - 2, endpoint=True)
        # t = np.append([0, 0, 0], t)
        # t = np.append(t, [1, 1, 1])
        # tck = [t, [x, y], 3]
        # u3 = np.linspace(0, 1, (max(l * 2, 70)), endpoint=True)
        # new_coordinates = interpolate.splev(u3, tck)

        if method == 'bspline':
            # scipy splprep
            interpolation_function, u = interpolate.splprep([x, y], s=weight_smooth, k=polynomial_order)
            # u = np.linspace(0, 1, num=50, endpoint=True)

            x_new, y_new = interpolate.splev(u, interpolation_function)
            new_coordinates = np.vstack((x_new, y_new)).T

            # # Scipy BSpline. Doesn't work
            # t, c, k = interpolation_function
            # c1 = np.asarray(c)
            # spline = interpolate.BSpline(t, c1.T, k, extrapolate=False)
            # new_coordinates = spline(coordinates[:, 0])

        elif method == 'moving_average':
            new_coordinates = self.moving_average(coordinates, window_length=window_length,
                                                  kernel_type=kernel_type, std_dev=std_dev)

        elif method == 'savgol':
            new_coordinates = self.savgol_filter(coordinates, window_length=window_length,
                                                 polynomial_order=polynomial_order)

        elif method == 'rgds':
            new_coordinates = self.recursive_gradient_descent_smoother(coordinates,
                                                                       weight_data=weight_data,
                                                                       weight_smooth=weight_smooth, tolerance=tolerance)

        # # fft. Doesn't work
        # w = fft.rfft(y)
        # f = fft.rfftfreq(window_length, x[1]-x[0])
        # spectrum = w ** 2
        # cutoff_idx = spectrum < (spectrum.max() / 1)
        # w2 = w.copy()
        # w2[cutoff_idx] = 0
        # y2 = fft.irfft(w2)

        return new_coordinates

    ''' Smoothing functions '''
    def moving_average(self, noisy_points, window_length=10, kernel_type='gaussian', std_dev=2):
        """

        :param noisy_points:
        :param window_length:
        :param kernel_type: gaussian, hann, box
        :param std_dev:
        :return:
        """
        smoothed_points = np.zeros_like(noisy_points)
        kernels = {'gaussian': signal.windows.gaussian(window_length, std=std_dev),
                   'hann': signal.windows.hann(window_length),
                   'box': np.ones(window_length) / window_length}
        weights = kernels.setdefault(kernel_type, 'gaussian')

        for dim in range(noisy_points.shape[1]):
            column_data = noisy_points[:, dim]
            if kernel_type == 'box':
                # uniform_filter is faster than both cumsum and convolve (see https://stackoverflow.com/a/43200476)
                smoothed_points[:, dim] = ndimage.uniform_filter1d(column_data, window_length,
                                                                   mode='nearest')

                # smoothed_points[:, dim] = np.convolve(column_data, weights, mode='same')
                # smoothed_points[:, dim] = signal.convolve(column_data, weights, mode='same') # same as above

                # # cumsum method is faster than the convolve method
                # cumsum = np.cumsum(np.insert(column_data, 0, 0))
                # smoothed_points = (cumsum[window_length:] - cumsum[:-window_length]) / window_length

            else:
                smoothed_points[:, dim] = np.convolve(column_data, weights, mode='same') / sum(weights)

        return smoothed_points

    def savgol_filter(self, noisy_points, window_length=10, polynomial_order=3):
        smoothed_points = np.zeros_like(noisy_points)
        for dim in range(noisy_points.shape[1]):
            column_data = noisy_points[:, dim]
            smoothed_points[:, dim] = signal.savgol_filter(column_data, window_length=window_length,
                                                           polyorder=polynomial_order)
        return smoothed_points

    def recursive_gradient_descent_smoother(self, path, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001):
        """
        A model free recursive smoother.
        Creates a smooth path for a n-dimensional series of coordinates.
        Arguments:
            path: List containing coordinates of a path
            weight_data: Float, how much weight to update the data (alpha)
            weight_smooth: Float, how much weight to smooth the coordinates (beta).
            tolerance: Float, how much change per iteration is necessary to keep iterating.
        Output:
            new: List containing smoothed coordinates.
        """
        new = deepcopy(path)
        dims = len(path[0])
        change = tolerance

        while change >= tolerance:
            change = 0.0
            for i in range(1, len(new) - 1):
                for j in range(dims):
                    x_i = path[i][j]
                    y_i, y_prev, y_next = new[i][j], new[i - 1][j], new[i + 1][j]

                    y_i_saved = y_i
                    y_i += weight_data * (x_i - y_i) + weight_smooth * (y_next + y_prev - (2 * y_i))
                    new[i][j] = y_i

                    change += abs(y_i - y_i_saved)

        return new

    def generate_velocity_profile(self):
        # see: https://www.mathworks.com/help/nav/ref/velocitycommand.html
        pass

    def get_desired_psi(self):
        # see: https://www.mathworks.com/help/nav/ref/headingfromxy.html
        # https://gitlab.com/autowarefoundation/autoware.ai/core_planning/-/blob/a056c3908676e3bf8f39daadb1699ab946dd55ca/mpc_follower/src/mpc_utils.cpp#L150
        pass

    def get_desired_curvature(self):
        # see: https://www.mathworks.com/help/nav/ref/referencepathfrenet.html
        # https://gitlab.com/autowarefoundation/autoware.ai/core_planning/-/blob/a056c3908676e3bf8f39daadb1699ab946dd55ca/mpc_follower/src/mpc_utils.cpp#L168
        # https://github.com/berlala/genesis_path_follower/blob/master/scripts/gps_utils/ref_gps_traj.py#L35
        pass
