#!/usr/bin/env python3
r"""Minimal go-to-goal planner: turn a goal pose into a `nav_msgs/Path` the controller drives.

The controller consumes a Path, never a goal -- it has no planner of its own. Nav2
supplies that plan in a real stack; this node stands in for it so the go-to-goal
path can be exercised against the simulator with no Nav2 install. It subscribes to
a `geometry_msgs/PoseStamped` goal (RViz "2D Goal Pose" publishes exactly this on
/goal_pose) and publishes a plan from the vehicle's *current* pose to that goal.

The plan is a cubic Hermite curve honouring both the start and goal headings,
resampled at uniform arc length. There is no map, no costmap and no obstacle
check: the curve is smooth and heading-consistent, nothing more. It exists to
feed the controller a plan that starts where the vehicle actually is, which is
the property a static test path cannot have -- a replan that starts at a fixed
origin puts the vehicle metres off its own path the instant it switches.

The Hermite tangent scale trades corner-cutting against curvature: too small and
the curve leaves the start heading abruptly, too large and it loops. Curvature is
reported per plan so an infeasible goal (tighter than the vehicle can turn) is
visible before the solver has to refuse it.

Usage:
    python3 goal_to_plan.py --ros-args -p plan_topic:=trajectory/path
    ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \\
      "{header: {frame_id: 'odom'}, pose: {position: {x: 6.0, y: 3.0}, \\
        orientation: {z: 0.383, w: 0.924}}}"
"""
import math

import numpy as np
import rclpy
import tf_transformations
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile


def _mod2pi(a):
    return a - 2.0 * math.pi * math.floor(a / (2.0 * math.pi))


def _dubins_words(alpha, beta, d):
    """The six Dubins word candidates as (modes, (t, p, q)) in normalized length."""
    sa, sb, ca, cb = math.sin(alpha), math.sin(beta), math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)
    out = []

    p_sq = 2 + d * d - 2 * c_ab + 2 * d * (sa - sb)
    if p_sq >= 0:
        tmp = math.atan2(cb - ca, d + sa - sb)
        out.append(('LSL', (_mod2pi(-alpha + tmp), math.sqrt(p_sq), _mod2pi(beta - tmp))))

    p_sq = 2 + d * d - 2 * c_ab + 2 * d * (sb - sa)
    if p_sq >= 0:
        tmp = math.atan2(ca - cb, d - sa + sb)
        out.append(('RSR', (_mod2pi(alpha - tmp), math.sqrt(p_sq), _mod2pi(-beta + tmp))))

    p_sq = -2 + d * d + 2 * c_ab + 2 * d * (sa + sb)
    if p_sq >= 0:
        p = math.sqrt(p_sq)
        tmp = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2.0, p)
        out.append(('LSR', (_mod2pi(-alpha + tmp), p, _mod2pi(-beta + tmp))))

    p_sq = d * d - 2 + 2 * c_ab - 2 * d * (sa + sb)
    if p_sq >= 0:
        p = math.sqrt(p_sq)
        tmp = math.atan2(ca + cb, d - sa - sb) - math.atan2(2.0, p)
        out.append(('RSL', (_mod2pi(alpha - tmp), p, _mod2pi(beta - tmp))))

    tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        out.append(('RLR', (t, p, _mod2pi(alpha - beta - t + p))))

    tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (sb - sa)) / 8.0
    if abs(tmp) <= 1:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(-alpha + math.atan2(-ca + cb, d + sa - sb) + p / 2.0)
        out.append(('LRL', (t, p, _mod2pi(beta - alpha - t + p))))

    return out


def _integrate_word(start, modes, lengths, radius, spacing):
    """Exactly integrate a Dubins word, sampling at ~`spacing` metres."""
    x, y, yaw = start
    pts = [(x, y, yaw)]
    for mode, seg in zip(modes, lengths):
        if seg <= 0.0:
            continue
        steps = max(1, int(math.ceil(seg * radius / spacing)))
        du = seg / steps
        for _ in range(steps):
            if mode == 'S':
                x += radius * du * math.cos(yaw)
                y += radius * du * math.sin(yaw)
            elif mode == 'L':
                nyaw = yaw + du
                x += radius * (math.sin(nyaw) - math.sin(yaw))
                y -= radius * (math.cos(nyaw) - math.cos(yaw))
                yaw = nyaw
            else:  # 'R'
                nyaw = yaw - du
                x += radius * (math.sin(yaw) - math.sin(nyaw))
                y -= radius * (math.cos(nyaw) - math.cos(yaw))
                yaw = nyaw
            pts.append((x, y, yaw))
    return np.array(pts)


def dubins_plan(start, goal, spacing, radius):
    """Shortest forward-only path between two poses that never turns tighter than `radius`.

    Returns (plan, length). A Dubins path is the correct primitive for a car that
    cannot reverse: unlike an interpolating spline it *respects* the turning limit
    rather than demanding a radius the vehicle has to cut. Words are accepted only
    after integrating them and checking the endpoint really lands on the goal, so a
    formula slip degrades to "no word found" instead of a silently wrong path.
    """
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    dist = math.hypot(dx, dy)
    d = dist / radius
    theta = math.atan2(dy, dx) if dist > 1e-9 else 0.0
    alpha, beta = _mod2pi(start[2] - theta), _mod2pi(goal[2] - theta)

    best = None
    for modes, lengths in _dubins_words(alpha, beta, d):
        if any(seg < 0 for seg in lengths):
            continue
        pts = _integrate_word(start, modes, lengths, radius, spacing)
        end = pts[-1]
        pos_err = math.hypot(end[0] - goal[0], end[1] - goal[1])
        yaw_err = abs((end[2] - goal[2] + math.pi) % (2 * math.pi) - math.pi)
        if pos_err > 1e-6 or yaw_err > 1e-6:
            continue  # formula/branch mismatch -- reject rather than emit a wrong path
        total = sum(lengths) * radius
        if best is None or total < best[1]:
            best = (pts, total, modes)
    if best is None:
        return None, 0.0, ''
    return best[0], best[1], best[2]


def hermite_plan(start, goal, spacing, tangent_scale):
    """Resample a cubic Hermite curve between two (x, y, yaw) poses at uniform arc length.

    Returns an (M, 3) array of [x, y, yaw]; yaw follows the curve tangent so the
    reference heading is consistent with the geometry the controller tracks.
    """
    x0, y0, yaw0 = start
    x1, y1, yaw1 = goal
    d = math.hypot(x1 - x0, y1 - y0) * tangent_scale
    p0, p1 = np.array([x0, y0]), np.array([x1, y1])
    t0 = d * np.array([math.cos(yaw0), math.sin(yaw0)])
    t1 = d * np.array([math.cos(yaw1), math.sin(yaw1)])

    # Dense sampling first: the Hermite parameter is not arc length, so uniform
    # spacing has to come from a re-interpolation against the measured arc length.
    s = np.linspace(0.0, 1.0, 1000)[:, None]
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    dense = h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1

    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total < spacing:
        return None, 0.0

    targets = np.arange(0.0, total, spacing)
    targets = np.append(targets, total)
    xy = np.column_stack([np.interp(targets, arc, dense[:, 0]),
                          np.interp(targets, arc, dense[:, 1])])

    tangents = np.gradient(xy, axis=0)
    yaw = np.arctan2(tangents[:, 1], tangents[:, 0])
    # Pin the endpoint headings: the one-sided gradient there is a poorer estimate
    # than the boundary condition the curve was built from, and a start-heading
    # error is exactly what the vehicle cannot work off at standstill.
    yaw[0], yaw[-1] = yaw0, yaw1

    # Peak curvature of the dense curve, i.e. the tightest radius the plan demands.
    dxy = np.gradient(dense, axis=0)
    ddxy = np.gradient(dxy, axis=0)
    speed = np.linalg.norm(dxy, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.abs(dxy[:, 0] * ddxy[:, 1] - dxy[:, 1] * ddxy[:, 0]) / speed ** 3
    max_kappa = float(np.nanmax(kappa))
    return np.column_stack([xy, yaw]), max_kappa


class GoalToPlan(Node):
    """Publish a Path from the live vehicle pose to each received goal pose."""

    def __init__(self):
        super().__init__('goal_to_plan')
        self.declare_parameter('goal_topic', '/goal_pose')
        self.declare_parameter('plan_topic', 'trajectory/path')
        self.declare_parameter('odom_topic', 'odometry/local')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('spacing', 0.1)
        self.declare_parameter('tangent_scale', 0.7)
        self.declare_parameter('plan_qos', 'volatile')
        self.declare_parameter('plan_type', 'dubins')
        self.declare_parameter('min_turn_radius', 0.5)

        gp = (lambda n: self.get_parameter(n).value)
        self.frame_id = gp('frame_id')
        self.spacing = float(gp('spacing'))
        self.tangent_scale = float(gp('tangent_scale'))
        self.min_turn_radius = float(gp('min_turn_radius'))
        self.plan_type = gp('plan_type')
        if self.plan_type == 'dubins' and self.min_turn_radius <= 0.0:
            self.get_logger().warn(
                'plan_type=dubins needs min_turn_radius > 0; falling back to hermite.')
            self.plan_type = 'hermite'

        durability = (DurabilityPolicy.TRANSIENT_LOCAL
                      if gp('plan_qos') == 'transient_local' else DurabilityPolicy.VOLATILE)
        self.plan_pub = self.create_publisher(
            Path, gp('plan_topic'), QoSProfile(depth=1, durability=durability))
        self.create_subscription(Odometry, gp('odom_topic'), self._odom_callback, 1)
        self.create_subscription(PoseStamped, gp('goal_topic'), self._goal_callback, 1)

        self._pose = None
        self.get_logger().info(
            f"goal_to_plan: {gp('goal_topic')} -> {gp('plan_topic')} "
            f"({gp('plan_qos')}), frame {self.frame_id}")

    def _odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self._pose = (p.x, p.y, yaw)

    def _goal_callback(self, msg: PoseStamped):
        if self._pose is None:
            self.get_logger().warn('Goal received before any odometry; ignoring.')
            return

        q = msg.pose.orientation
        _, _, goal_yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        goal = (msg.pose.position.x, msg.pose.position.y, goal_yaw)

        if self.plan_type == 'dubins':
            plan, length, word = dubins_plan(
                self._pose, goal, self.spacing, self.min_turn_radius)
            if plan is None:
                self.get_logger().warn('No Dubins solution for this goal; ignoring.')
                return
            max_kappa = 1.0 / self.min_turn_radius
            detail = f'{word}, {length:.2f} m'
        else:
            plan, max_kappa = hermite_plan(
                self._pose, goal, self.spacing, self.tangent_scale)
            if plan is None:
                self.get_logger().warn('Goal is closer than one waypoint spacing; ignoring.')
                return
            detail = f'min radius {1.0 / max_kappa if max_kappa > 0 else float("inf"):.2f} m'

        # A VOLATILE plan published before the controller has matched is simply lost,
        # and the run then looks like a controller that ignored the goal.
        if self.plan_pub.get_subscription_count() == 0:
            self.get_logger().warn(
                'No subscriber on the plan topic; publishing anyway, but a volatile '
                'plan sent before the controller matches will be dropped.')

        if self.min_turn_radius > 0.0 and max_kappa > 1.0 / self.min_turn_radius:
            self.get_logger().warn(
                f'Plan demands radius {1.0 / max_kappa:.2f} m, tighter than the '
                f'{self.min_turn_radius:.2f} m the vehicle can turn; it will cut the curve.')

        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y, yaw in plan:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            qz = tf_transformations.quaternion_from_euler(0.0, 0.0, float(yaw))
            (pose.pose.orientation.x, pose.pose.orientation.y,
             pose.pose.orientation.z, pose.pose.orientation.w) = [float(v) for v in qz]
            path.poses.append(pose)
        self.plan_pub.publish(path)

        self.get_logger().info(
            f'Published {self.plan_type} plan: {len(path.poses)} poses, '
            f'({self._pose[0]:.2f}, {self._pose[1]:.2f}) -> ({goal[0]:.2f}, {goal[1]:.2f}), '
            f'{detail}')


def main(args=None):
    rclpy.init(args=args)
    node = GoalToPlan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
