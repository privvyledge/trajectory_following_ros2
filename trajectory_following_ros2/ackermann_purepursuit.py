import math
import sys
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from std_msgs.msg import Float32

from trajectory_following_ros2.base_tracker import BaseTrajectoryTracker
from trajectory_following_ros2.backends.base_solver import BaseSolver


class AckermannPurePursuit(BaseTrajectoryTracker):
    """
    Pure Pursuit geometric path tracking controller.

    Overrides the MPC control loop entirely; reuses the base class
    pub/sub infrastructure (odom, path, speed callbacks, publishers).
    """

    def __init__(self):
        super().__init__('purepursuit')
        # Re-create yaw_rate publisher under purepursuit namespace
        self.yaw_rate_pub = self.create_publisher(Float32, 'purepursuit/des_yaw_rate', 1)
        self.get_logger().info('purepursuit node started.')

    # ------------------------------------------------------------------
    # Base class abstract/override points
    # ------------------------------------------------------------------

    def _declare_backend_parameters(self):
        # PP overrides the odom default to match the old node's default
        # (odom_topic is already declared in base, but we need to override
        #  the default. Easiest: just set it in our __init__ after super().__init__.)
        self.declare_parameter('lookahead_distance', 0.4)
        self.declare_parameter('min_lookahead', 0.3)
        self.declare_parameter('max_lookahead', 10.452)
        self.declare_parameter('adaptive_lookahead_gain', 0.4)
        self.declare_parameter('use_adaptive_lookahead', False)
        self.declare_parameter('speed_Kp', 2.0)
        self.declare_parameter('speed_Ki', 0.2)
        self.declare_parameter('speed_Kd', 0.0)

    def _init_solver(self) -> Optional[BaseSolver]:
        # Read PP-specific params here (after declare)
        self.desired_lookahead = self.get_parameter('lookahead_distance').value
        self.lookahead = self.desired_lookahead
        self.MIN_LOOKAHEAD = self.get_parameter('min_lookahead').value
        self.MAX_LOOKAHEAD = self.get_parameter('max_lookahead').value
        self.lookahead_gain = self.get_parameter('adaptive_lookahead_gain').value
        self.use_adaptive_lookahead = self.get_parameter('use_adaptive_lookahead').value

        self.speed_Kp = self.get_parameter('speed_Kp').value
        self.speed_Ki = self.get_parameter('speed_Ki').value
        self.speed_Kd = self.get_parameter('speed_Kd').value
        self._speed_Pterm = 0.0
        self._speed_Iterm = 0.0
        self._speed_Dterm = 0.0
        self._speed_last_error = None

        self.desired_steering_angle = 0.0
        self.alpha = 0.0
        self.goal: Optional[np.ndarray] = None

        return None  # PP has no MPC solver

    # ------------------------------------------------------------------
    # PP control loop (replaces MPC timer callback)
    # ------------------------------------------------------------------

    def _control_timer_callback(self):
        # Stale-odom guard
        if not self._odom_is_fresh():
            if self.initial_pose_received:
                self.get_logger().warn(
                    'Stale odometry, skipping PP step', throttle_duration_sec=1.0)
                self._publish_zero_command()
            return

        # Lazy trajectory initialization
        if self.path_received and self.desired_speed_received and not self.trajectory_initialized:
            self._init_trajectory()

        if not (self.initial_pose_received and self.path_received):
            return

        # Goal check
        if self.trajectory.check_goal(
                self.x, self.y, self.speed,
                self.final_goal, self.current_idx, self.path.shape[0]):
            self.get_logger().info('Final goal reached.')
            self._publish_zero_command()
            return

        # Adaptive lookahead
        if self.use_adaptive_lookahead:
            lookahead = self.lookahead_gain * self.speed + self.desired_lookahead
            self.lookahead = float(np.clip(lookahead, self.MIN_LOOKAHEAD, self.MAX_LOOKAHEAD))

        # Update trajectory state and find target point
        self._update_trajectory_state(self.x, self.y, self.speed, self.yaw, self.omega)

        self.current_idx, ref_traj, _, _, _ = self.trajectory.calc_ref_trajectory(
            state=None, trajectory=None, current_index=None,
            dt=self.sample_time, prediction_horizon=50,
            lookahead_time=1.0, lookahead=self.lookahead,
            num_points_to_interpolate=50)

        if ref_traj is None:
            self.get_logger().info('End of trajectory reached.')
            self._publish_zero_command()
            return

        self.goal = self.path[self.current_idx, :]

        # Pure Pursuit steering law
        self.alpha = math.atan2(self.goal[1] - self.y,
                                self.goal[0] - self.x) - self.yaw
        steering = math.atan2(
            2.0 * self.WHEELBASE * math.sin(self.alpha), self.lookahead)
        self.desired_steering_angle = float(
            np.clip(steering, self.MIN_STEER_ANGLE, self.MAX_STEER_ANGLE))

        # Speed command
        target_speed = self._get_target_speed() or 0.0
        speed_cmd: float
        if abs(self.speed_Kp) > 0.0:
            speed_cmd = self._pid_speed(target_speed, self.speed)
        else:
            speed_cmd = target_speed
        speed_cmd = float(np.clip(speed_cmd, -self.MAX_SPEED, self.MAX_SPEED))

        # Publish
        self.delta_cmd = self.desired_steering_angle
        self.velocity_cmd = speed_cmd
        self.acc_cmd = 0.0
        self._publish_command()

        if self.publish_twist_topic:
            self._publish_twist()

        self.run_count += 1
        self.update_queue(self.goal_queue, self.goal[:2].tolist())

    # ------------------------------------------------------------------
    # PID speed controller
    # ------------------------------------------------------------------

    def _pid_speed(self, target: float, measured: float) -> float:
        error = target - measured
        self._speed_Pterm = self.speed_Kp * error
        self._speed_Iterm += self.speed_Ki * error * self.sample_time
        if self._speed_last_error is not None:
            self._speed_Dterm = (
                self.speed_Kd * (error - self._speed_last_error) / self.sample_time)
        self._speed_last_error = error
        return self._speed_Pterm + self._speed_Iterm + self._speed_Dterm


def main(args=None):
    rclpy.init(args=args)
    try:
        node = AckermannPurePursuit()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
