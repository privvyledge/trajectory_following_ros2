"""
Kinematic bicycle simulator using do-mpc's SUNDIALS CVODES integrator.

Publishes odometry and acceleration feedback at `update_rate` Hz, acting as a
physics plant for hardware-free testing paired with a controller node.
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node

import tf_transformations
from rcl_interfaces.msg import ParameterDescriptor
from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

from trajectory_following_ros2.do_mpc.model import BicycleKinematicModel
from trajectory_following_ros2.simulator.do_mpc.do_mpc_simulator import Simulator


class KinematicDoMPCSimulationNode(Node):

    def __init__(self):
        super(KinematicDoMPCSimulationNode, self).__init__('kinematic_do_mpc_simulator')

        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('global_frame', 'odom')
        self.declare_parameter('model_type', 'continuous')
        self.declare_parameter('update_rate', 20.0)
        self.declare_parameter('wheelbase', 0.256)
        self.declare_parameter('min_steer', -30.0,
                               ParameterDescriptor(description='Minimum steering angle (degrees).'))
        self.declare_parameter('max_steer', 30.0)
        self.declare_parameter('max_steer_rate', 60 / 0.17)
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

        self.robot_frame = self.get_parameter('robot_frame').value
        self.global_frame = self.get_parameter('global_frame').value
        self.update_rate = self.get_parameter('update_rate').value
        self.WHEELBASE = self.get_parameter('wheelbase').value
        self.MAX_STEER_ANGLE = math.radians(self.get_parameter('max_steer').value)
        self.MIN_STEER_ANGLE = math.radians(self.get_parameter('min_steer').value)
        self.MAX_STEER_RATE = math.radians(self.get_parameter('max_steer_rate').value)
        self.MAX_SPEED = self.get_parameter('max_speed').value
        self.MIN_SPEED = self.get_parameter('min_speed').value
        self.MAX_ACCEL = self.get_parameter('max_accel').value
        self.MAX_DECEL = self.get_parameter('max_decel').value
        self.saturate_input = self.get_parameter('saturate_input').value
        self.NX = self.get_parameter('n_states').value
        self.NU = self.get_parameter('n_inputs').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.acceleration_topic = self.get_parameter('acceleration_topic').value
        self.ackermann_cmd_topic = self.get_parameter('ackermann_cmd_topic').value
        self.debug = self.get_parameter('debug').value

        self.dt = self.sample_time = 1.0 / self.update_rate

        self.acc_cmd = self.delta_cmd = self.velocity_cmd = 0.0
        self.zk = np.zeros((self.NX, 1))  # [x, y, v, psi]
        self.uk = np.array([[self.acc_cmd, self.delta_cmd]]).T
        self.u_prev = np.zeros((self.NU, 1))

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.speed_prev = 0.0
        self.yaw_rate = 0.0
        self.acceleration = 0.0
        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))

        self.vehicle_model = BicycleKinematicModel(
            length=self.WHEELBASE, width=0.192, sample_time=self.sample_time)
        # TVP placeholders required by the do-mpc model structure; not used by the simulator.
        self.vehicle_model.wp_id = 0
        self.vehicle_model.reference_path = [[0., 0., 0., 0.]]
        self.simulator = Simulator(self.vehicle_model, sample_time=self.sample_time)
        self.simulator = self.simulator.simulator
        self.simulator.x0 = self.zk

        self.ackermann_cmd_sub = self.create_subscription(
            AckermannDriveStamped, self.ackermann_cmd_topic, self.ackermann_cmd_callback, 1)
        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 1)
        self.accel_pub = self.create_publisher(
            AccelWithCovarianceStamped, self.acceleration_topic, 1)
        self.simulation_timer = self.create_timer(self.sample_time, self.update_timer_callback)

        self.get_logger().info('kinematic_do_mpc_simulator node started.')

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
        self.zk[:, :] = self.simulator.make_step(uk)
        self.x = self.zk[0, 0]
        self.y = self.zk[1, 0]
        self.speed = self.zk[2, 0]
        self.yaw = self.zk[3, 0]

        self.yaw_rate = (self.speed / self.WHEELBASE) * math.tan(uk[1, 0])
        self.acceleration = (self.speed - self.speed_prev) / self.sample_time

        self.publish_odometry()
        self.publish_acceleration()

        self.rear_x = self.x - ((self.WHEELBASE / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((self.WHEELBASE / 2) * math.sin(self.yaw))
        self.speed_prev = self.speed

    def publish_odometry(self):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = self.global_frame
        odom_msg.child_frame_id = self.robot_frame

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        q = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        odom_msg.twist.twist.linear.x = self.speed
        odom_msg.twist.twist.angular.z = self.yaw_rate

        self.odom_pub.publish(odom_msg)

    def publish_acceleration(self):
        accel_msg = AccelWithCovarianceStamped()
        accel_msg.header.stamp = self.get_clock().now().to_msg()
        accel_msg.header.frame_id = self.robot_frame
        accel_msg.accel.accel.linear.x = self.acceleration
        accel_msg.accel.accel.angular.z = self.yaw_rate / self.sample_time
        self.accel_pub.publish(accel_msg)


def main(args=None):
    rclpy.init(args=args)
    kdsim_node = KinematicDoMPCSimulationNode()
    rclpy.spin(kdsim_node)
    kdsim_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()