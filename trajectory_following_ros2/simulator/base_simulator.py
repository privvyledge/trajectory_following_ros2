"""
Abstract base class for kinematic bicycle simulator ROS 2 nodes.

Owns all shared ROS 2 infrastructure: parameter declarations, state variables,
Ackermann command subscription, odometry/acceleration/TF publishing, and the
simulation timer callback. Items 7a (actuation lag), 7b (steering rate clamp),
and 7e (Gaussian sensor noise) are implemented once here.

Subclasses implement two abstract methods:
  _build_integrator() -- set up backend-specific integrator once in __init__
  _step(x, u) -> np.ndarray -- advance state by one dt
"""
import math
from abc import ABC, abstractmethod

import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
import tf_transformations
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import AccelWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped


class BaseSimulator(Node, ABC):
    """Shared ROS 2 boilerplate for kinematic bicycle simulator nodes."""

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._declare_parameters()
        self._read_parameters()
        self._init_state()
        self._build_integrator()
        self._setup_pub_sub()
        self.get_logger().info(f'{node_name} node started.')

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _declare_parameters(self):
        pd = ParameterDescriptor
        self.declare_parameter(
            'robot_frame', 'base_link',
            pd(description='Local/body frame of the vehicle.'))
        self.declare_parameter(
            'global_frame', 'odom',
            pd(description='Global/world frame (odom or map).'))
        self.declare_parameter('update_rate', 50.0)
        self.declare_parameter('wheelbase', 0.256)
        self.declare_parameter(
            'min_steer', -30.0,
            pd(description='Minimum steering angle (degrees).'))
        self.declare_parameter(
            'max_steer', 30.0,
            pd(description='Maximum steering angle (degrees).'))
        self.declare_parameter(
            'max_steer_rate', 60.0 / 0.17,
            pd(description='Maximum steering rate (degrees/s).'))
        self.declare_parameter('max_speed', 100.5)
        self.declare_parameter('min_speed', -100.5)
        self.declare_parameter('max_accel', 30.0)
        self.declare_parameter('max_decel', -30.0)
        self.declare_parameter('saturate_input', True)
        self.declare_parameter('n_states', 4)
        self.declare_parameter('n_inputs', 2)
        self.declare_parameter('odom_topic', 'odometry/filtered')
        self.declare_parameter('acceleration_topic', 'accel/local')
        self.declare_parameter('ackermann_cmd_topic', 'drive')
        self.declare_parameter('debug', False)
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw', 0.0)
        self.declare_parameter('initial_speed', 0.0)
        # Item 7a: first-order lag time constants (s); 0.0 = instantaneous
        self.declare_parameter('steering_time_constant', 0.0)
        self.declare_parameter('acceleration_time_constant', 0.0)
        # Item 7e: Gaussian sensor noise std devs; 0.0 = no noise
        self.declare_parameter('noise_std_x', 0.0)
        self.declare_parameter('noise_std_y', 0.0)
        self.declare_parameter('noise_std_v', 0.0)
        self.declare_parameter('noise_std_psi', 0.0)

    def _read_parameters(self):
        def gp(name):
            return self.get_parameter(name).value

        self.robot_frame = gp('robot_frame')
        self.global_frame = gp('global_frame')
        self.update_rate = gp('update_rate')
        self.WHEELBASE = gp('wheelbase')
        self.MAX_STEER_ANGLE = math.radians(gp('max_steer'))
        self.MIN_STEER_ANGLE = math.radians(gp('min_steer'))
        self.MAX_STEER_RATE = math.radians(gp('max_steer_rate'))
        self.MAX_SPEED = gp('max_speed')
        self.MIN_SPEED = gp('min_speed')
        self.MAX_ACCEL = gp('max_accel')
        self.MAX_DECEL = gp('max_decel')
        self.saturate_input = gp('saturate_input')
        self.NX = gp('n_states')
        self.NU = gp('n_inputs')
        self.odom_topic = gp('odom_topic')
        self.acceleration_topic = gp('acceleration_topic')
        self.ackermann_cmd_topic = gp('ackermann_cmd_topic')
        self.debug = gp('debug')
        self.dt = self.sample_time = 1.0 / self.update_rate
        self._tau_steer = gp('steering_time_constant')
        self._tau_accel = gp('acceleration_time_constant')
        self._noise_std_x = gp('noise_std_x')
        self._noise_std_y = gp('noise_std_y')
        self._noise_std_v = gp('noise_std_v')
        self._noise_std_psi = gp('noise_std_psi')

    def _init_state(self):
        def gp(name):
            return self.get_parameter(name).value

        self.x = gp('initial_x')
        self.y = gp('initial_y')
        self.yaw = gp('initial_yaw')
        self.speed = gp('initial_speed')
        self.speed_prev = 0.0
        self.yaw_rate = 0.0
        self.acceleration = 0.0
        self.rear_x = self.x - (self.WHEELBASE / 2) * math.cos(self.yaw)
        self.rear_y = self.y - (self.WHEELBASE / 2) * math.sin(self.yaw)

        self.acc_cmd = 0.0
        self.delta_cmd = 0.0
        self.velocity_cmd = 0.0
        self.zk = np.zeros((self.NX, 1))  # column [x, y, v, psi]
        self.zk[0, 0] = self.x
        self.zk[1, 0] = self.y
        self.zk[2, 0] = self.speed
        self.zk[3, 0] = self.yaw
        self.uk = np.zeros((self.NU, 1))  # column [accel, delta]
        self.u_prev = np.zeros((self.NU, 1))

        # Item 7a: lag filter state (tracks actual applied actuator value)
        self._delta_actual = 0.0
        self._accel_actual = 0.0

    def _setup_pub_sub(self):
        self.br = TransformBroadcaster(self)
        self.ackermann_cmd_sub = self.create_subscription(
            AckermannDriveStamped, self.ackermann_cmd_topic,
            self.ackermann_cmd_callback, 1)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 1)
        self.accel_pub = self.create_publisher(
            AccelWithCovarianceStamped, self.acceleration_topic, 1)
        self.simulation_timer = self.create_timer(
            self.sample_time, self.update_timer_callback)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_integrator(self):
        """Set up backend-specific integrator. Called once during __init__."""

    @abstractmethod
    def _step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Advance state one dt. x: (4,) [x,y,v,psi], u: (2,) [accel,delta]. Returns (4,)."""

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def ackermann_cmd_callback(self, data):
        self.u_prev[:, :] = self.uk
        self.delta_cmd = data.drive.steering_angle
        self.velocity_cmd = data.drive.speed
        if data.drive.acceleration != 0.0:
            self.acc_cmd = data.drive.acceleration
        else:
            self.acc_cmd = (self.velocity_cmd - self.speed) / self.sample_time
        self.uk[0, 0] = self.acc_cmd
        self.uk[1, 0] = self.delta_cmd

    def update_timer_callback(self):
        # Zero-order hold: reuses last uk if no new command arrived.
        uk = self.uk.copy()

        # Item 7b: clamp steering rate before applying lag
        max_delta_step = self.MAX_STEER_RATE * self.sample_time
        uk[1, 0] = self.u_prev[1, 0] + np.clip(
            uk[1, 0] - self.u_prev[1, 0], -max_delta_step, max_delta_step)

        # Item 7a: first-order actuation lag
        if self._tau_steer > 0.0:
            self._delta_actual += (self.sample_time / self._tau_steer) * (
                uk[1, 0] - self._delta_actual)
        else:
            self._delta_actual = uk[1, 0]

        if self._tau_accel > 0.0:
            self._accel_actual += (self.sample_time / self._tau_accel) * (
                uk[0, 0] - self._accel_actual)
        else:
            self._accel_actual = uk[0, 0]

        u_eff = np.array([self._accel_actual, self._delta_actual])
        x_next = self._step(self.zk.flatten(), u_eff)
        self.zk[:, 0] = x_next

        self.x = x_next[0]
        self.y = x_next[1]
        self.speed = x_next[2]
        self.yaw = x_next[3]

        self.yaw_rate = (self.speed / self.WHEELBASE) * math.tan(self._delta_actual)
        self.acceleration = (self.speed - self.speed_prev) / self.sample_time

        self.publish_transform()
        self.publish_odometry()
        self.publish_acceleration()

        self.rear_x = self.x - (self.WHEELBASE / 2) * math.cos(self.yaw)
        self.rear_y = self.y - (self.WHEELBASE / 2) * math.sin(self.yaw)
        self.speed_prev = self.speed

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _noise(self, std: float) -> float:
        """Return a zero-mean Gaussian sample; returns 0.0 when std is 0."""
        return float(np.random.normal(0.0, std)) if std > 0.0 else 0.0

    def publish_odometry(self):
        # Item 7e: add optional Gaussian noise before publishing
        x_pub = self.x + self._noise(self._noise_std_x)
        y_pub = self.y + self._noise(self._noise_std_y)
        v_pub = self.speed + self._noise(self._noise_std_v)
        psi_pub = self.yaw + self._noise(self._noise_std_psi)

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = self.global_frame
        odom_msg.child_frame_id = self.robot_frame
        odom_msg.pose.pose.position.x = x_pub
        odom_msg.pose.pose.position.y = y_pub
        odom_msg.pose.pose.position.z = 0.0
        q = tf_transformations.quaternion_from_euler(0, 0, psi_pub)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        odom_msg.twist.twist.linear.x = v_pub
        odom_msg.twist.twist.angular.z = self.yaw_rate
        self.odom_pub.publish(odom_msg)

    def publish_acceleration(self):
        accel_msg = AccelWithCovarianceStamped()
        accel_msg.header.stamp = self.get_clock().now().to_msg()
        accel_msg.header.frame_id = self.robot_frame
        accel_msg.accel.accel.linear.x = self.acceleration
        accel_msg.accel.accel.angular.z = self.yaw_rate / self.sample_time
        self.accel_pub.publish(accel_msg)

    def publish_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.global_frame
        t.child_frame_id = self.robot_frame
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.br.sendTransform(t)


def main_spin(node_factory, args=None):
    """Shared spin loop for simulator entry points."""
    rclpy.init(args=args)
    node = node_factory()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()