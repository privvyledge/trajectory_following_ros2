"""
Todo:
    * implement circle line intersection [done]
    * switch pose, goal, waypoints to include x, y, vel, yaw, etc. [done]
    * implement interpolation
    * rename waypoints to trajectory: x, y, vel, yaw, curvature, cum_dist, arc_length, time, dt [done]
    * plot
    * rework functions here
    * time methods in functions and choose the faster methods
    * integrate functions from Autowares https://gitlab.com/autowarefoundation/autoware.ai/core_planning/-/blob/a056c3908676e3bf8f39daadb1699ab946dd55ca/mpc_follower/src/mpc_utils.cpp. See autoware_mpc_utils.py
    * implement lateral, longitudinal and angular error threshold to substitute the goal tolerance
"""
from copy import deepcopy
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal, fft, ndimage
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.signal import filtfilt
from scipy.interpolate import splprep, splev, interp1d, CubicSpline
import pandas as pd  # todo: remove

import trajectory_following_ros2.utils.filters as filters


def get_distance(node1, node2, metric='euclidean'):
    """
    Calculate distance
    :param p1: list, point1
    :param p2: list, point2
    :return: float, distance
    """
    distances = cdist(node1, node2, metric=metric).flatten()

    # # method2
    # distances = np.linalg.norm(node1 - node2, axis=-1)
    # # method3
    # distances = np.hypot(node1[0, 0] - node2[:, 0], node1[0, 1] - node2[:, 1])
    #
    # dx = [node1[0] - icx for icx in node2[:, 0]]
    # dy = [node1[1] - icy for icy in node2[:, 1]]
    # # method4
    # distances = np.hypot(dx, dy)
    # # method5
    # distances = [math.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]

    return distances


def generate_kd_tree(waypoints=None):
    return KDTree(waypoints)


def find_closest_waypoints(waypoints, position, num_neighbours=10,
                           current_index=0, min_search_radius=0.0, max_search_radius=np.inf,
                           use_euclidean_distance=True, workers=1):
    # use min_search_radius as goal threshold and max_search_radius to restrict the waypoints searched
    # todo: could search only from >=current_index to speed up
    # todo: should we find the current_index first or just keep track of that?
    indices = np.arange(len(waypoints))
    distances = get_distance(position, waypoints)

    # # restrict to indices greater than the current index distance + min_search_radius and current_index distance + max_search_radius
    # if current_index is not None:
    #     # indices = indices[indices >= current_index]
    #     # distances = distances[indices >= current_index]

    # find the indices with the shortest distance
    indices = indices[(indices >= current_index) & (min_search_radius <= distances) & (distances <= max_search_radius)]
    # distances = distances[indices]

    # restrict to num_neighbours
    try:
        indices = indices[:num_neighbours]
        distances = distances[indices]
        # distances = distances[:num_neighbours]
    except IndexError:
        # distances = distances
        # indices = indices
        pass

    return indices, distances


def find_closest_waypoints_kdtree(waypoints_kd_tree, position, num_neighbours=1,
                                  current_index=0, min_search_radius=0.0, max_search_radius=np.inf,
                                  use_euclidean_distance=True, workers=1, create_kdtree=False, waypoints=None):
    """
    The returned indices and distances are not guaranteed to be sorted in ascending order.
    Same as above but using a KDTree
    :param waypoints_kd_tree:
    :param position:
    :param num_neighbours:
    :param current_index:
    :param min_search_radius:
    :param max_search_radius:
    :param use_euclidean_distance:
    :param workers:
    :param create_kdtree:
    :param waypoints:
    :return:
    """
    if create_kdtree:
        if waypoints is not None:
            waypoints_kd_tree = generate_kd_tree(waypoints)

    if num_neighbours == 1:
        num_neighbours = [num_neighbours]

    p = 1  # Manhattan distance
    if use_euclidean_distance:
        p = 2

    distances, indices = None, None
    # query the nearest neighbour
    distances, indices = waypoints_kd_tree.query(position,
                                                 k=num_neighbours, distance_upper_bound=max_search_radius, p=p,
                                                 workers=workers)

    # remove invalid distances if < num_neighbours have distances < search_radius
    # and restrict to only distances >= min_search_radius distance
    if len(distances) > 0:
        # indices_in_between = indices[(min_search_radius <= distances) & (distances <= max_search_radius)]
        # distances_in_between = distances[(min_search_radius <= distances) & (distances <= max_search_radius)]
        # indices_greater_than_current = indices_in_between[indices_in_between >= current_index]
        # distances_greater_than_current = distances_in_between[indices_in_between >= current_index]

        indices_greater_than_current = indices[indices >= current_index]
        distances_greater_than_current = distances[indices >= current_index]
        indices_in_between = indices_greater_than_current[(min_search_radius <= distances_greater_than_current) & (
                distances_greater_than_current <= max_search_radius)]
        distances_in_between = distances_greater_than_current[(min_search_radius <= distances_greater_than_current) & (
                distances_greater_than_current <= max_search_radius)]
        indices_in_between = indices_in_between[:num_neighbours]
        distances_in_between = distances_in_between[:num_neighbours]

        indices = indices_in_between
        distances = distances_in_between
    return indices, distances


def get_angle_from_position(position):
    """
    Returns an angle given a position vector. atan2(y, x).
    Can be used to generate desired angles if only x, y are available
    :param position:
    :return:
    """
    if isinstance(position, list) or isinstance(position, tuple) or len(position.shape) == 1:
        return np.arctan2(position[1], position[0])
    return np.arctan2(position[:, 1], position[:, 0])


def fix_angle_reference(angle_ref, angle_init):
    # similar to smooth yaw. todo: test and compare
    # This function returns a "smoothened" angle_ref wrt angle_init so there are no jumps.
    diff_angle = angle_ref - angle_init
    diff_angle = normalize_angle(diff_angle, minus_pi_to_pi=True, pi_is_negative=True, degrees=False)
    diff_angle = np.unwrap(diff_angle)  # removes jumps greater than pi
    return angle_init + diff_angle


def get_angle_using_model(*args):
    # todo
    pass


def normalize_angle(angle, minus_pi_to_pi=True, pi_is_negative=True, degrees=False):
    """

    :param angle:
    :param minus_pi_to_pi: If False normalize between [0, 2pi),
        If True and pi_is_negative is True normalize between [-pi, pi),
        elif True and pi_is_negative is False normalize between [-pi, pi]
    :param pi_is_negative:
    :param degrees: If the input and output are in degrees
    :return:
    """
    if isinstance(angle, float) or isinstance(angle, int):
        is_float = True
    else:
        is_float = False

    angle = np.asarray(angle).flatten()
    if degrees:
        angle = np.deg2rad(angle)

    if minus_pi_to_pi:
        normalized_angle = normalize_angle_from_minus_pi_to_pi(angle, pi_is_negative=pi_is_negative)
    else:
        normalized_angle = normalize_angle_from_zero_to_2pi(angle)

    if degrees:
        normalized_angle = np.rad2deg(normalized_angle)

    if is_float:
        return normalized_angle.item()
    return normalized_angle


def normalize_angle_from_zero_to_2pi(angle):
    """
    map from 0 to 2 PI, [0, 2PI)
    :param angle:
    :return:
    """
    # method 1
    normalized_angle = angle - 2.0 * np.pi * np.floor(angle / (2.0 * np.pi))

    # # method 2
    # normalized_angle = np.fmod(angle, 2.0 * np.pi)
    # normalized_angle = normalized_angle if normalized_angle >= 0 else normalized_angle + 2.0 * np.pi
    #
    # # method 3. fmod(a, b)
    # normalized_angle = np.fmod(angle, 2.0 * np.pi)
    # normalized_angle = np.fmod(normalized_angle + 2.0 * np.pi, 2.0 * np.pi)

    return normalized_angle


def normalize_angle_from_minus_pi_to_pi(angle, pi_is_negative=True):
    """
    Maps from -pi to pi, 0 maps to 0, 360 maps to 0, 180 maps to -180
    :param angle:
    :param pi_is_negative: if True normalizes to [-pi, pi), else normalizes to [-pi, pi]
    :return:
    """
    if pi_is_negative:
        # normalize from [-pi, pi), 180 -> -180
        # todo: investigate using the below formulation since using fmod can introduce discontinuities
        # normalized_angle = np.arctan2(np.sin(angle), np.cos(angle))

        # method 1
        normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi  # normalizes to [-pi, pi)

        # # method 2  # normalizes to [-pi, pi)
        # normalized_angle = angle - 2.0 * np.pi * np.floor((angle + np.pi) / (2.0 * np.pi))

        # # method 3
        # normalized_angle = angle
        # if normalized_angle >= np.pi:
        #     normalized_angle = normalized_angle - 2.0 * np.pi  # normalizes to [-pi, pi)
        # elif normalized_angle <= -np.pi:
        #     normalized_angle = normalized_angle + 2.0 * np.pi
        return normalized_angle

    # normalize from [-pi, pi], 180 -> 180
    # method 4:   # normalizes to [-pi, pi], 180
    normalized_angle = np.arctan2(np.sin(angle), np.cos(angle))  # normalizes to [-pi, pi]
    # method 5:   # normalizes to [-pi, pi],
    normalized_angle = angle
    while normalized_angle > math.pi:
        normalized_angle = angle - 2.0 * np.pi
    while normalized_angle < -math.pi:
        normalized_angle = angle + 2.0 * np.pi
    return normalized_angle


def get_errors(pose, goal, psi_current, psi_desired):
    # returns longitudinal, lateral and angular errors
    ref_to_axle = pose[:, :2] - goal[:, :2]
    crosstrack_vector = np.array(
            [np.sin(psi_desired), -np.cos(psi_desired)]
    )
    crosstrack_err = ref_to_axle.dot(crosstrack_vector)

    # angular error
    psi_error = psi_current - psi_desired  # pose[3] - goal[3]
    psi_error = normalize_angle(psi_error, minus_pi_to_pi=True, pi_is_negative=True, degrees=True)
    err_x = pose[:, 0] - goal[:, 0]
    err_y = pose[:, 1] - goal[:, 1]
    lat_error = -np.sin(psi_desired) * err_x + np.cos(psi_desired) * err_y

    long_error = None
    return crosstrack_err, psi_error, lat_error, long_error


def get_arc_lengths(waypoints):
    # same as cumulative_distance_along_path. todo: time and compare
    d = np.diff(waypoints, axis=0)
    consecutive_diff = np.sqrt(np.sum(np.power(d, 2), axis=1))
    # consecutive_diff = np.linalg.norm(d, axis=-1)
    dists_cum = np.cumsum(consecutive_diff)
    dists_cum = np.insert(dists_cum, 0, 0.0)
    return dists_cum


def cumulative_distance_along_path(waypoints):
    # get_arc_lengths
    # todo: time and compare, also
    # method 1
    # Xs = waypoints[:, 0]
    # Ys = waypoints[:, 1]
    cdists = np.zeros(len(waypoints))  # cumulative distance along path (m, aka "s" in Frenet formulation)
    d = np.linalg.norm(waypoints[:-1, :] - waypoints[1:, :], axis=-1)
    cdists[1:] = np.cumsum(d)  # s = s_prev + dist(z[i], z[i-1])

    # # method2: the below implentation is equivalent but slower. Use the vectorized implementation above.
    # Xs = []  # global X position (m, wrt to origin at LON0, LAT0)
    # Ys = []  # global Y position (m, wrt to origin at LON0, LAT0)
    # cdists = []  # cumulative distance along path (m, aka "s" in Frenet formulation)
    #
    # for i in range(len(waypoints)):
    #     X, Y = waypoints[i, :]
    #
    #     if len(Xs) == 0:  # i.e. the first point on the trajectory
    #         cdists.append(0.0)  # s = 0
    #
    #     else:  # later points on the trajectory
    #         d = math.sqrt((X - Xs[-1]) ** 2 + (Y - Ys[-1]) ** 2) + cdists[-1]
    #         cdists.append(d)  # s = s_prev + dist(z[i], z[i-1])
    #     Xs.append(X)
    #     Ys.append(Y)

    return cdists


def calculate_current_arc_length(current_speed, normalized_yaw_diff, dt):
    # Compute velocity along path
    s_dot = current_speed * np.cos(normalized_yaw_diff)
    # distance travelled along reference path
    s = s_dot * dt
    return s


def calculate_curvature_all(cdists, psis, smooth=True):
    """
    Calculates the curvature of every point and optionally smoothes them
    :param cdists:
    :param psis:
    :return:
    """
    # method1: https://github.com/berlala/genesis_path_follower/blob/master/scripts/gps_utils/ref_gps_traj.py#L35
    # This function estimates curvature using finite differences (curv = dpsi/dcdist).
    diff_dists = np.diff(cdists)
    diff_psis = np.diff(np.unwrap(psis))

    assert np.max(np.abs(diff_psis)) < np.pi, "Detected a jump in the angle difference."

    curv_raw = diff_psis / np.maximum(diff_dists, 0.1)  # use diff_dists where greater than 10 cm
    curv_raw = np.insert(curv_raw, len(curv_raw), curv_raw[-1])  # curvature at last waypoint

    # Curvature Filtering
    if smooth:
        curv_filt = filtfilt(np.ones((11,)) / 11, 1, curv_raw)
        return curv_filt
    return curv_raw


def calculate_curvature_single(waypoints, goal_index):
    """
    Calculate the curvature of a point using the previous two points
    :param waypoints:
    :param goal_index:
    :return:
    """
    # method 2: # https://github.com/ai-winter/python_motion_planning/blob/master/python_motion_planning/local_planner/local_planner.py#L154
    if goal_index == 1:
        goal_index = goal_index + 1
    idx_prev = goal_index - 1
    idx_pprev = idx_prev - 1

    a = get_distance(waypoints[idx_prev, :].reshape(1, -1), waypoints[goal_index, :].reshape(1, -1)).item()
    b = get_distance(waypoints[idx_pprev, :].reshape(1, -1), waypoints[goal_index, :].reshape(1, -1)).item()
    c = get_distance(waypoints[idx_pprev, :].reshape(1, -1), waypoints[idx_prev, :].reshape(1, -1)).item()
    try:
        cosB = (a * a + c * c - b * b) / (2 * a * c)
    except ZeroDivisionError:
        # fails if a or c is zero (i.e. the 2 previous points are the same caused by duplicates)
        return 0.0

    sinB = np.sin(np.arccos(cosB))

    cross = (waypoints[idx_prev][0] - waypoints[idx_pprev][0]) * \
            (waypoints[goal_index][1] - waypoints[idx_pprev][1]) - \
            (waypoints[idx_prev][1] - waypoints[idx_pprev][1]) * \
            (waypoints[goal_index][0] - waypoints[idx_pprev][0])
    kappa = math.copysign(2 * sinB / b, cross)

    return kappa


def calc_path_relative_time(path, speeds, min_dt=0.001):
    """
    Calculate the relative time for each point in the path as well as dt.
    Note: for now the dts are wrong. Todo: fix
    Todo: interpolate
    :param path:
    :return:
    """
    t = 0.0
    path_time = []
    path_dt = [0.0]
    # path_time.clear()
    path_time.append(t)
    # todo: speed up by vectorizing
    for i in range(len(path) - 1):
        x0 = path[i, 0]
        y0 = path[i, 1]
        z0 = 0.0  # path[i, 0]
        x1 = path[i + 1, 0]
        y1 = path[i + 1, 1]
        z1 = 0.0  # path[i + 1, 0]
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        dist = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        v = max(abs(speeds[i]), 0.1)  # 1.0, 0.1. to avoid zero division error. np.finfo(float).eps
        dt = max(dist / v, min_dt)
        t += dt
        path_time.append(t)
        path_dt.append(dt)

    # dists = np.linalg.norm(waypoints[:-1, :2] - waypoints[1:, :2], axis=-1)
    # t = np.cumsum(dists / np.maximum(speeds, 0.1))
    return path_time, path_dt


def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    # https://github.com/thomasfermi/Algorithms-for-Automated-Driving/blob/master/code/solutions/control/get_target_point.py
    # https://github.com/ai-winter/python_motion_planning/blob/master/python_motion_planning/local_planner/local_planner.py#L143 | https://github.com/ai-winter/python_motion_planning/blob/master/python_motion_planning/utils/helper/math_helper.py#L11
    #

    # Find the intersection between a circle and a line segment.
    # :param circle_center: The (x, y) location of the center of the circle.
    # :param circle_radius: The radius of the circle.
    # :param pt1: The (x, y) location of the first point of the segment.
    # :param pt2: The (x, y) location of the second point of the segment.
    # :param full_line: True to find intersections along the full line, not just in the segment. False to only find intersections in the line segment.
    # :param tangent_tol: Numerical tolerance at which we decide the intersection are close enough to consider it a tangent.
    # :return: Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.
    # :rtype: list

    circle_center = np.array(circle_center)
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)

    # Vector in the direction of the line
    d = np.subtract(pt2, pt1)
    f = np.subtract(pt1, circle_center)

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - circle_radius ** 2

    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        return []  # no intersection

    # two possible solutions
    discriminant = np.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)

    # check if the first intersection is on the segment
    if not full_line and (t1 < 0 or t1 > 1):
        t1 = None

    # check if the second intersection is on the segment
    if not full_line and (t2 < 0 or t2 > 1):
        t2 = None

    if t1 is not None and t2 is not None:
        return [pt1 + t1 * d, pt1 + t2 * d]
    elif t1 is not None:
        return [pt1 + t1 * d]
    elif t2 is not None:
        return [pt1 + t2 * d]
    return []


def circle_line_segment_intersection2(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2) ** .5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment. todo: catch ZeroDivisionError
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant ** .5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant ** .5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                      intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections


def circleSegmentIntersection(p1: tuple, p2: tuple, r: float) -> list:
    # generate comments for this function
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]

    dx, dy = x2 - x1, y2 - y1
    dr2 = dx * dx + dy * dy
    D = x1 * y2 - x2 * y1

    # the first element is the point within segment
    d1 = x1 * x1 + y1 * y1
    d2 = x2 * x2 + y2 * y2
    dd = d2 - d1

    delta_2 = r * r * dr2 - D * D
    if delta_2 < 0:  # no intersection
        return []

    delta = math.sqrt(delta_2)
    if (delta == 0):
        return [(D * dy / dr2, -D * dx / dr2)]
    else:  # delta > 0
        return [
            ((D * dy + math.copysign(1.0, dd) * dx * delta) / dr2,
             (-D * dx + math.copysign(1.0, dd) * dy * delta) / dr2),
            ((D * dy - math.copysign(1.0, dd) * dx * delta) / dr2,
             (-D * dx - math.copysign(1.0, dd) * dy * delta) / dr2)
        ]


def closestPointOnLine(a: tuple, b: tuple, p: tuple = (0.0, 0.0)) -> tuple:
    """
    Find the closest intersection point (foot of a perpendicular) between point p and the line ab.

    Parameters:
        a (tuple): point a of the line
        b (tuple): point b of the line
        p (tuple): point p to find the closest intersection point

    References:
        [1] method 2 of https://www.youtube.com/watch?v=TPDgB6136ZE
    """
    ap = (p[0] - a[0], p[1] - a[1])
    ab = (b[0] - a[0], b[1] - a[1])
    af_coef = (ap[0] * ab[0] + ap[1] * ab[1]) / (ab[0] ** 2 + ab[1] ** 2)
    af = (af_coef * ab[0], af_coef * ab[1])
    f = (a[0] + af[0], a[1] + af[1])
    return f


def get_target_point(lookahead, polyline, circle_center=(0, 0), full_line=False, method=1):
    """ Determines the target point for the pure pursuit controller

    Parameters
    ----------
    lookahead : float
        The target point is on a circle of radius `lookahead`
        The circle's center is (0,0)
    polyline: array_like, shape (M,2)
        A list of 2d points that defines a polyline.

    Returns:
    --------
    target_point: numpy array, shape (,2)
        Point with positive x-coordinate where the circle of radius `lookahead`
        and the polyline intersect.
        Return None if there is no such point.
        If there are multiple such points, return the one that the polyline
        visits first.
    """
    intersections = []
    for j in range(len(polyline) - 1):
        pt1 = polyline[j]
        pt2 = polyline[j + 1]
        if method == 1:
            intersections += circle_line_segment_intersection(circle_center, lookahead, pt1, pt2, full_line=full_line)
        else:
            intersections += circle_line_segment_intersection2(circle_center, lookahead, pt1, pt2, full_line=full_line)
    filtered = [p for p in intersections if p[0] > 0]
    if len(filtered) == 0:
        return None
    return filtered[0]


def convert_error_to_frenet_frame(pose, goal, psi_current, psi_desired, cdists, index, r_global_to_frenet):
    # similar to get_errors
    error_xy = pose - goal  # xy deviation (global frame)

    error_frenet = np.dot(r_global_to_frenet, error_xy[0, :])  # e_s, e_y deviation (Frenet frame)

    s = cdists[index]
    e_s = error_frenet[0]  # longitudinal deviation
    e_y = error_frenet[1]  # lateral deviation
    psi_error = psi_current - psi_desired  #  pose[3] - goal[3]
    psi_error = normalize_angle(psi_error, minus_pi_to_pi=True, pi_is_negative=True, degrees=True)
    return s, e_s, e_y, psi_error


def convert_to_frenet_frame(car_pose, ref_path):
    """
    todo: test as this was generated by GitHub copilot
    Convert the global car pose to Frenet frame coordinates.

    Args:
        car_pose (dict): A dictionary with keys 'position' and 'orientation' for the car.
        ref_path (np.ndarray): An array of reference path points with shape (N, 3) where each point is (x, y, yaw).

    Returns:
        tuple: Frenet coordinates (s, d).
    """
    # Extract car position and orientation
    car_x = car_pose['position']['x']
    car_y = car_pose['position']['y']
    car_yaw = Rotation.from_quat([
        car_pose['orientation']['x'],
        car_pose['orientation']['y'],
        car_pose['orientation']['z'],
        car_pose['orientation']['w']
    ]).as_euler('xyz')[2]

    # Create KDTree for the reference path
    ref_tree = KDTree(ref_path[:, :2])

    # Find the closest point on the reference path
    _, idx = ref_tree.query([car_x, car_y])
    closest_point = ref_path[idx]

    # Calculate the longitudinal distance (s)
    s = np.linalg.norm(ref_path[:idx + 1, :2] - ref_path[0, :2], axis=1).sum()

    # Calculate the lateral distance (d)
    ref_x, ref_y, ref_yaw = closest_point
    dx = car_x - ref_x
    dy = car_y - ref_y
    d = -np.sin(ref_yaw) * dx + np.cos(ref_yaw) * dy

    return s, d


def interpolate_trajectory(waypoints, num_points=100, weight_smooth=0.5, polynomial_order=3, remove_duplicates=True):
    """
    Interpolate the trajectory using spline interpolation.

    Use this to interpolate the reference path of horizon length.
    Right now is not good for interpolating the entire trajectory.

    Args:
        waypoints (np.ndarray): An array of waypoints with shape (N, 2) where each point is (x, y).
        num_points (int): The desired number of interpolated points.

    Returns:
        np.ndarray: An array of interpolated trajectory points with shape (num_points, 2).
    """
    # Extract x and y coordinates
    if remove_duplicates:
        waypoints, indices = np.unique(waypoints[:, :2], axis=0, return_index=True)
        x = waypoints[:, 0]
        y = waypoints[:, 1]

    # Perform spline interpolation
    tck, u = splprep([x, y], s=weight_smooth, k=polynomial_order)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck)

    # Combine x and y coordinates into a single array
    interpolated_trajectory = np.vstack((x_new, y_new)).T

    return interpolated_trajectory


def interpolate_path(pts, arc_lengths_arr, smooth_value=0.1, scale=2, derivative_order=0, remove_duplicates=True):
    """
    Use this to interpolate the reference path of horizon length.
    Right now is not good for interpolating the entire trajectory.

    Use interpolate_trajectory instead
    :param pts:
    :param arc_lengths_arr:
    :param smooth_value:
    :param scale:
    :param derivative_order:
    :return:
    """
    # https://github.com/nirajbasnet/Nonlinear_MPCC_for_autonomous_racing/blob/master/nonlinear_mpc_casadi/scripts/Nonlinear_MPC_node.py#L217
    # tck represents vector of knots, the B-spline coefficients, and the degree of the spline.

    # BSpline interpolation only works for unique values of x
    if remove_duplicates:
        pts, indices = np.unique(pts, axis=0, return_index=True)

    tck, u = splprep(pts.T, u=arc_lengths_arr, s=smooth_value, per=1)

    # x, y = pts[:, 0], pts[:, 1]
    # tck, u = splprep([x, y], u=arc_lengths_arr, s=smooth_value, per=1)
    u_new = np.linspace(u.min(), u.max(), len(pts) * scale)
    x_new, y_new = splev(u_new, tck, der=derivative_order)

    interp_points = np.concatenate((x_new.reshape((-1, 1)), y_new.reshape((-1, 1))), axis=1)
    return interp_points, tck


# def get_interpolated_path_casadi(self, label_x, label_y, pts, arc_lengths_arr):
#     u = arc_lengths_arr
#     V_X = pts[:, 0]
#     V_Y = pts[:, 1]
#     lut_x = interpolant(label_x, 'bspline', [u], V_X)
#     lut_y = interpolant(label_y, 'bspline', [u], V_Y)
#     return lut_x, lut_y


def dynamic_smoothing_velocity(start_idx, start_vel, acc_lim, tau, waypoints):
    """
    Used to generate a smooth velocity profile for the waypoints.
    Can be run at each timestep or for the whole trajectory.
    Todo: speed up by vectorization
    :param start_idx:
    :param start_vel:
    :param acc_lim:
    :param tau:
    :param waypoints:
    :return:
    """
    curr_v = start_vel
    vel_list = np.zeros((len(waypoints)))
    vel_list[start_idx] = start_vel

    if len(waypoints[:, 2]) > 1:
        vel_list[start_idx + 1] = start_vel

    for i in range(start_idx + 2, len(waypoints[:, 0])):
        ds = np.linalg.norm(waypoints[i, :2] - waypoints[i - 1, :2], axis=-1)  # calc_distance_2d(traj, i, i - 1)
        dt = ds / max(abs(curr_v), np.finfo(float).eps)
        a = tau / max(tau + dt, np.finfo(float).eps)
        updated_v = a * curr_v + (1.0 - a) * waypoints[i, 2]
        dv = max(-acc_lim * dt, min(acc_lim * dt, updated_v - curr_v))
        curr_v += dv
        vel_list[i] = curr_v

    path_times, path_dts = calc_path_relative_time(waypoints, vel_list)
    return vel_list, path_times


def interpolate_speed(waypoints, method=1):
    """
    Interpolates the speed between waypoints. For now does not take into account acceleration and jerk limits.
    Todo: setup velocity profile, e.g trapezoidal, s-curve, etc
    :param waypoints:
    :return:
    """
    # Extract x, y coordinates and velocities
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    velocities = waypoints[:, 2]
    speeds = waypoints[:, 2]

    # Calculate cumulative distance along the path
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    if method == 1:
        interpolation_function, b_spline_coeffs = splprep([np.arange(speeds.shape[0]), speeds], s=0.3, k=3)
        _, sp = splev(b_spline_coeffs, interpolation_function)
        return sp

    elif method == 2:
        # Interpolate velocities along the cumulative distance
        interp_func = interp1d(cumulative_distances, velocities, kind='linear', fill_value="extrapolate")
        interpolated_velocities = interp_func(cumulative_distances)
        return interpolated_velocities

    elif method == 3:
        # # Interpolate velocities along the cumulative distance using CubicSpline
        # interp_func = interp1d(range(len(waypoints)), velocities, kind='cubic', fill_value="extrapolate")
        # interpolated_velocities = interp_func(range(len(waypoints)))

        cs = CubicSpline(range(len(waypoints)), velocities)
        # Generate interpolated velocities
        interpolated_velocities = cs(range(len(waypoints)))

        return interpolated_velocities


def trapezoidal_velocity_profile(distance, max_vel, accel, decel):
    # Time to accelerate to max velocity
    t_accel = max_vel / accel
    d_accel = 0.5 * accel * t_accel ** 2

    # Time to decelerate from max velocity
    t_decel = max_vel / decel
    d_decel = 0.5 * decel * t_decel ** 2

    # Check if the distance is sufficient to reach max velocity
    if d_accel + d_decel > distance:
        # Triangular profile
        t_accel = np.sqrt(2 * distance / (accel + (accel ** 2 / decel)))
        t_decel = t_accel * (accel / decel)
        t_constant = 0
    else:
        # Trapezoidal profile
        d_constant = distance - (d_accel + d_decel)
        t_constant = d_constant / max_vel

    # Total time
    total_time = t_accel + t_constant + t_decel

    # Generate time points
    time_points = np.linspace(0, total_time, num=1000)

    # Generate velocity profile
    velocity_profile = np.zeros_like(time_points)
    for i, t in enumerate(time_points):
        if t < t_accel:
            velocity_profile[i] = accel * t
        elif t < t_accel + t_constant:
            velocity_profile[i] = max_vel
        else:
            velocity_profile[i] = max_vel - decel * (t - t_accel - t_constant)

    return time_points, velocity_profile


def s_curve_velocity_profile(distance, max_vel, accel, decel, jerk):
    # Time to reach maximum acceleration
    t_accel = accel / jerk
    d_accel = 0.5 * jerk * t_accel ** 3

    # Time to reach maximum velocity
    t_vel = (max_vel - accel * t_accel) / accel
    d_vel = accel * t_vel ** 2 / 2 + accel * t_accel * t_vel

    # Time to decelerate to zero velocity
    t_decel = decel / jerk
    d_decel = 0.5 * jerk * t_decel ** 3

    # Check if the distance is sufficient to reach max velocity
    if d_accel + d_vel + d_decel > distance:
        # Adjust profile to fit within the distance
        t_accel = (distance / (2 * jerk)) ** (1 / 3)
        t_decel = t_accel
        t_vel = 0
    else:
        # S-curve profile
        d_constant = distance - (d_accel + d_vel + d_decel)
        t_constant = d_constant / max_vel

    # Total time
    total_time = 2 * t_accel + t_vel + t_constant + 2 * t_decel

    # Generate time points
    time_points = np.linspace(0, total_time, num=1000)

    # Generate velocity profile
    velocity_profile = np.zeros_like(time_points)
    for i, t in enumerate(time_points):
        if t < t_accel:
            velocity_profile[i] = 0.5 * jerk * t ** 2
        elif t < t_accel + t_vel:
            velocity_profile[i] = accel * (t - t_accel) + 0.5 * jerk * t_accel ** 2
        elif t < t_accel + t_vel + t_constant:
            velocity_profile[i] = max_vel
        elif t < t_accel + t_vel + t_constant + t_decel:
            velocity_profile[i] = max_vel - 0.5 * jerk * (t - t_accel - t_vel - t_constant) ** 2
        else:
            velocity_profile[i] = max_vel - accel * (
                        t - t_accel - t_vel - t_constant - t_decel) - 0.5 * jerk * t_decel ** 2

    return time_points, velocity_profile


def generate_reference_trajectory_by_interpolation(trajectory, init_pose, closest_index, waypoint_keys_to_columns,
                                                   horizon, v_target=None, dt=0.02):
    # (3) Find the reference trajectory using distance or time interpolation.
    #     WARNING: this function does not correctly handle cases where the car is far from the recorded path!
    #     Ill-defined behavior/speed.
    #     Could use the actual minimum distance and add appropriate logic to handle this edge case.

    # if dt == 0.0:
    #     dt = dts[closest_index]

    yaw_init = init_pose[:, 3].item()
    if v_target is not None:
        # Given a velocity reference, use the cumulative distance for interpolation.
        start_dist = trajectory[closest_index, waypoint_keys_to_columns['cum_dist']]
        # start_dist = cdists[closest_index]
        interp_by_key = 'cum_dist'
        interp_to_fit = [h * dt * v_target + start_dist for h in range(0, horizon + 1)]
    else:
        # No velocity reference provided.  So use timestamp for interpolation.
        # start_tm = path_times[closest_index]
        start_tm = trajectory[closest_index, waypoint_keys_to_columns['total_time_elapsed']]
        interp_by_key = 'total_time_elapsed'
        interp_to_fit = [h * dt + start_tm for h in range(0, horizon + 1)]  # or range(1, horizon + 2)

    waypoint_dict = {}
    for waypoint_key in ['x', 'y', 'yaw', 'cum_dist', 'curvature']:
        waypoint_dict[waypoint_key + '_ref'] = np.interp(interp_to_fit,
                                                         trajectory[:, waypoint_keys_to_columns[interp_by_key]],
                                                         trajectory[:, waypoint_keys_to_columns[waypoint_key]])
        if waypoint_key == 'yaw':
            waypoint_dict['yaw_ref'] = fix_angle_reference(waypoint_dict['yaw_ref'], yaw_init)

    # Reference velocity found by approximation using ds/dt finite differencing.
    waypoint_dict['vel_ref'] = np.diff(waypoint_dict['cum_dist_ref']) / dt
    waypoint_dict['vel_ref'] = np.insert(waypoint_dict['vel_ref'], len(waypoint_dict['vel_ref']),
                                       waypoint_dict['vel_ref'][-1])  # v_{N-1} = v_N
    return waypoint_dict


if __name__ == '__main__':
    # Load waypoints
    waypoints_path = '../../data/carla_waypoints.csv'
    # trajectory_keys = ['x', 'y', 'speed', 'yaw', 'omega', 'dt', 'total_time_elapsed']
    trajectory_keys = ['x', 'y', 'speed', 'yaw', 'omega', 'curvature', 'cum_dist', 'dt', 'total_time_elapsed']

    csv_df = pd.read_csv(waypoints_path, skipinitialspace=True)
    waypoints = csv_df.loc[:,
                ["x", "y", "speed", "yaw", "omega", "dt", "total_time_elapsed"]].to_numpy()  # rename to trajectory

    trajectory = np.zeros((len(waypoints), len(trajectory_keys)))
    trajectory_key_to_column = {key: i for i, key in enumerate(trajectory_keys)}

    # set current position
    current_index = 43
    current_state = csv_df.loc[current_index, ["x", "y", "speed", "yaw", "omega", "dt", "total_time_elapsed"]].to_numpy(
            dtype='float').reshape(1, -1)
    current_state_list = current_state.flatten().tolist()
    psi_current = current_state[:, 3].item()

    # generate smooth velocity profile
    velocity_time_constant = 0.3
    # use Autowares velocity smoother as it accounts for accel limits
    smooth_velocity, path_times = dynamic_smoothing_velocity(0, waypoints[0, 2], 3.0,
                                                             velocity_time_constant, waypoints)
    smooth_velocity2 = interpolate_speed(waypoints, method=1)
    smooth_velocity3 = interpolate_speed(waypoints, method=2)
    smooth_velocity4 = interpolate_speed(waypoints, method=3)

    waypoints[:, 2] = smooth_velocity

    # # plot the raw velocity against the smooth velocity
    # plt.plot(waypoints[:, 6], waypoints[:, 2], 'r', label='Raw Velocity')
    # plt.plot(waypoints[:, 6], smooth_velocity, 'b', label='Autoware Smoother')
    # plt.plot(waypoints[:, 6], smooth_velocity2, 'go', label='BSpline Smoother')
    # plt.plot(waypoints[:, 6], smooth_velocity3, 'bo', label='Linear Smoother')
    # plt.plot(waypoints[:, 6], smooth_velocity4, 'r*', label='CubicSpline Smoother')
    # plt.legend()
    # plt.show()

    # estimate time and dt
    relative_times, relative_dts = calc_path_relative_time(waypoints, waypoints[:, 2])
    # interpolate time and dt using linear interpolation
    # interpolated_times = np.interp(range(len(waypoints)), range(len(waypoints)), relative_times)
    # interpolated_dts = np.interp(range(len(waypoints)), range(len(waypoints)), relative_dts)

    interp = interpolate.interp1d(range(len(waypoints)), relative_times, kind="slinear", fill_value='extrapolate')
    interpolated_times = interp(range(len(waypoints)))

    interp = interpolate.interp1d(range(len(waypoints)), relative_dts, kind="slinear", fill_value='extrapolate')
    interpolated_dts = interp(range(len(waypoints)))

    # get angles
    estimated_angle_list = get_angle_from_position(current_state_list[:2])
    estimated_angle = get_angle_from_position(current_state[:, :2])
    estimated_angles = get_angle_from_position(waypoints[:, :2])
    estimated_angles_normalized = normalize_angle(estimated_angles, minus_pi_to_pi=True, pi_is_negative=True,
                                                  degrees=True)

    # get arc lengths. Note: arc_lengths and cumulative_distance_along_path are the same
    arc_lengths = get_arc_lengths(waypoints[:, :2])
    # cumulative_distance_along_path(waypoints)
    cdists = cumulative_distance_along_path(waypoints[:, :2])

    # interpolate/filter the path
    smooth_path = interpolate_trajectory(waypoints[20:70, :], num_points=100, weight_smooth=0.5, polynomial_order=3)  # todo: test
    interpolated_paths, tck = interpolate_path(waypoints[20:70, :2], None, smooth_value=0.1, scale=2,
                                               derivative_order=0)

    rgds_path = filters.smooth_and_interpolate_coordinates(coordinates=waypoints[:, :2],
                                                           method='rgds', weight_data=0.5,
                                                           weight_smooth=0.5)

    savgol_path = filters.smooth_and_interpolate_coordinates(coordinates=waypoints[:, :2],
                                                             method='savgol', window_length=9,
                                                             polynomial_order=3)

    bspline_path = filters.smooth_and_interpolate_coordinates(coordinates=waypoints[:, :2],
                                                              method='bspline', polynomial_order=3,
                                                              weight_smooth=0.1)
    mvavg_box_path = filters.smooth_and_interpolate_coordinates(coordinates=waypoints[:, :2],
                                                                method='moving_average',
                                                                window_length=9, kernel_type='box')
    mvavg_hann_path = filters.smooth_and_interpolate_coordinates(coordinates=waypoints[:, :2],
                                                                 method='moving_average',
                                                                 window_length=9, kernel_type='hann')

    # plt.plot(waypoints[:, 0], waypoints[:, 1], 'r*', label='Waypoints')
    # plt.plot(smooth_path[:, 0], smooth_path[:, 1], 'bo')
    # plt.plot(interpolated_paths[:, 0], interpolated_paths[:, 1], 'g*')
    # plt.plot(rgds_path[:, 0], rgds_path[:, 1], 'go')
    # plt.plot(savgol_path[:, 0], savgol_path[:, 1], 'yo')
    # plt.plot(bspline_path[:, 0], bspline_path[:, 1], 'black')
    # plt.plot(mvavg_box_path[:, 0], mvavg_box_path[:, 1], 'orange')
    # plt.plot(mvavg_hann_path[:, 0], mvavg_hann_path[:, 1], 'y*')
    # plt.legend()
    # plt.show()

    # calculate curvatures
    curvs_all = calculate_curvature_all(cdists, waypoints[:, 3], smooth=True)
    curvature_at_index = calculate_curvature_single(waypoints[:, :2], goal_index=current_index)

    # plot the estimated dt against the actual timestamps
    plt.plot(range(len(waypoints)), interpolated_times, 'r*', label='Interpolated times')
    # plt.plot(range(len(waypoints)), relative_times, 'bo', label='Estimated times')
    plt.plot(range(len(waypoints)), waypoints[:, 6], 'g*', label='Actual times')
    plt.legend()
    plt.show()

    # plot the estimated dt against the actual dt
    plt.plot(range(len(waypoints)), interpolated_dts, 'r*', label='Interpolated dt')
    # plt.plot(range(len(waypoints)), relative_dts, 'bo', label='Estimated dt')
    plt.plot(range(len(waypoints)), waypoints[:, 5], 'g*', label='Actual dt')
    plt.legend()
    plt.show()

    # update trajectory array. ['x', 'y', 'speed', 'yaw', 'omega', 'curvature', 'cum_dist', 'dt', 'total_time_elapsed']
    # todo: update with smooth/interpolated values using a flag
    # trajectory[:, :5] = waypoints[:, :5]  # x, y, speed, yaw, omega
    # trajectory[:, [7, 8]] = waypoints[:, [5, 6]]  # dt, total_time_elapsed
    trajectory[:, trajectory_key_to_column['x']] = waypoints[:, 0]
    trajectory[:, trajectory_key_to_column['y']] = waypoints[:, 1]
    trajectory[:, trajectory_key_to_column['speed']] = waypoints[:, 2]
    trajectory[:, trajectory_key_to_column['yaw']] = waypoints[:, 3]
    trajectory[:, trajectory_key_to_column['omega']] = waypoints[:, 4]
    trajectory[:, trajectory_key_to_column['curvature']] = curvs_all
    trajectory[:, trajectory_key_to_column['cum_dist']] = cdists
    trajectory[:, trajectory_key_to_column['dt']] = waypoints[:, 5]
    trajectory[:, trajectory_key_to_column['total_time_elapsed']] = waypoints[:, 6]

    # generate the kdtree from the waypoints
    waypoints_kdtree = KDTree(waypoints[:, :2])

    # find closest waypoint using euclidean distance
    ii, dd = find_closest_waypoints(waypoints[:, :2], current_state[:, :2], current_index=current_index,
                                    num_neighbours=50, min_search_radius=0.0, max_search_radius=30.0,
                                    use_euclidean_distance=True,
                                    workers=1)

    # find the closest waypoint using kdtree nearest neighbour search
    ii2, dd2 = find_closest_waypoints_kdtree(waypoints_kdtree, current_state[:, :2], current_index=current_index,
                                             num_neighbours=50, min_search_radius=0.0, max_search_radius=30.0,
                                             use_euclidean_distance=True, workers=1)

    # get circle line intersection
    # first transform point to the robots frame so it's centered around the rear axle, e.g (0, 0)
    # for example, goal = goal - current_state
    lookahead = 15.0
    pt1 = current_state[:, :2] - current_state[:, :2]
    pt2 = waypoints[ii[1], :2] - current_state[:, :2]
    full_line = True
    intersections = circle_line_segment_intersection((0, 0), lookahead,
                                                     pt1.flatten().tolist(), pt2.flatten().tolist(),
                                                     full_line=full_line)
    intersections2 = circle_line_segment_intersection2((0, 0), lookahead,
                                                       pt1.flatten().tolist(), pt2.flatten().tolist(),
                                                       full_line=full_line)
    intersections3 = circleSegmentIntersection(pt1.flatten().tolist(), pt2.flatten().tolist(), lookahead)

    # same for method1 and method2
    if len(intersections3) == 0:
        intersections3 = closestPointOnLine(pt1.flatten().tolist(), pt2.flatten().tolist())

    # restore the point to the global frame
    if len(intersections3) > 0:
        intersections3 = np.array(intersections3) + current_state[:, :2]

    # get target point using circle line intersection
    target_point = get_target_point(lookahead=lookahead, polyline=waypoints[:, :2], method=1, full_line=full_line)
    target_point2 = get_target_point(lookahead=lookahead, polyline=waypoints[:, :2], method=2, full_line=full_line)

    # calculate errors
    psi_desired = waypoints[ii[0], 3]
    crosstrack_err, psi_error, lat_error, long_error = get_errors(current_state[:, :2],
                                                                  waypoints[ii[0], :2].reshape(1, -1),
                                                                  psi_current,
                                                                  psi_desired)

    # get errors in Frenet frame. psi_rotates from the frenet to the global frame
    rot_frenet_to_global_object = Rotation.from_euler('z', psi_desired).as_matrix()
    # rot_global_to_frenet = np.array([[np.cos(psi_desired), np.sin(psi_desired)],
    #                                  [-np.sin(psi_desired), np.cos(psi_desired)]])
    rot_global_to_frenet = Rotation.from_matrix(rot_frenet_to_global_object).inv().as_matrix()

    s, e_y, e_s, psi_error2 = convert_error_to_frenet_frame(current_state[:, :2], waypoints[ii[1], :2],
                                                            psi_current, psi_desired, cdists, current_index,
                                                            rot_global_to_frenet[:2, :2])

    # generate reference trajectory
    reference_trajectory = generate_reference_trajectory_by_interpolation(trajectory, current_state, current_index,
                                                                          trajectory_key_to_column, horizon=20,
                                                                          v_target=None, dt=0.02)
    reference_trajectory2 = generate_reference_trajectory_by_interpolation(trajectory, current_state, current_index,
                                                                           trajectory_key_to_column, horizon=20,
                                                                           v_target=8.9, dt=0.02)

    print("Done")
