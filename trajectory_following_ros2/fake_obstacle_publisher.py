"""Publish static spherical obstacles as ``derived_object_msgs/ObjectArray``.

A test/bring-up helper for the obstacle-avoidance path. The controller
(``base_tracker._obstacle_callback``) reads each object's ``pose.position``
(x, y) and, for a ``SPHERE``, ``shape.dimensions[0]`` as the radius. Obstacles
are NOT transformed through TF — the position is compared directly against the
vehicle odometry, so publish in the same frame as the odometry (the controller's
``global_frame``, ``odom`` by default; ``map -> odom`` is an identity TF in the
closed-loop sim, so waypoint/map coordinates can be used directly).

Parameters (all runtime-declared):
- ``obstacle_x`` / ``obstacle_y`` / ``obstacle_radius`` (double arrays, one entry
  per obstacle; must be equal length),
- ``obstacle_topic`` (str, default ``fake_obstacles/object_array``),
- ``frame_id`` (str, default ``odom``),
- ``publish_rate`` (double Hz, default ``10.0``).

Example::

    ros2 run trajectory_following_ros2 fake_obstacle_publisher \
        --ros-args -p obstacle_x:=[3.0] -p obstacle_y:=[0.0] -p obstacle_radius:=[0.3]
"""
import rclpy
from rclpy.node import Node

from derived_object_msgs.msg import Object, ObjectArray


class FakeObstaclePublisher(Node):
    """Publish a fixed set of spherical obstacles at a steady rate."""

    def __init__(self):
        super().__init__('fake_obstacle_publisher')

        self.declare_parameter('obstacle_x', [3.0])
        self.declare_parameter('obstacle_y', [0.0])
        self.declare_parameter('obstacle_radius', [0.3])
        self.declare_parameter('obstacle_topic', 'fake_obstacles/object_array')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('publish_rate', 10.0)

        xs = list(self.get_parameter('obstacle_x').get_parameter_value().double_array_value)
        ys = list(self.get_parameter('obstacle_y').get_parameter_value().double_array_value)
        rs = list(self.get_parameter('obstacle_radius').get_parameter_value().double_array_value)
        if not (len(xs) == len(ys) == len(rs)) or not xs:
            raise ValueError(
                'obstacle_x, obstacle_y and obstacle_radius must be non-empty and equal length '
                f'(got {len(xs)}, {len(ys)}, {len(rs)}).')

        self._obstacles = list(zip(xs, ys, rs))
        self._topic = self.get_parameter('obstacle_topic').value
        self._frame_id = self.get_parameter('frame_id').value
        rate = float(self.get_parameter('publish_rate').value)

        self._pub = self.create_publisher(ObjectArray, self._topic, 1)
        self._timer = self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(
            f'Publishing {len(self._obstacles)} SPHERE obstacle(s) on {self._topic} '
            f'(frame={self._frame_id}) at {rate} Hz: {self._obstacles}')

    def _publish(self):
        msg = ObjectArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id

        for i, (x, y, r) in enumerate(self._obstacles):
            obj = Object()
            obj.header = msg.header
            obj.id = i
            obj.pose.position.x = float(x)
            obj.pose.position.y = float(y)
            obj.pose.position.z = 0.0
            obj.pose.orientation.w = 1.0
            # SPHERE -> dimensions[0] is the radius (see base_tracker._obstacle_callback).
            obj.shape.type = obj.shape.SPHERE
            obj.shape.dimensions = [float(r)]
            msg.objects.append(obj)

        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeObstaclePublisher()
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
