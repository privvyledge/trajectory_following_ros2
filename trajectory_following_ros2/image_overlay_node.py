#!/usr/bin/env python3

import math
import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy

# ROS message types
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PointStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32

# TF2 listener
from tf2_ros import Buffer, TransformListener
import tf_transformations

# CvBridge for OpenCV conversion
from cv_bridge import CvBridge

# Obstacle Array support
try:
    from derived_object_msgs.msg import ObjectArray
    OBSTACLES_AVAILABLE = True
except ImportError:
    OBSTACLES_AVAILABLE = False


class ImageOverlayNode(Node):
    """
    A ROS 2 Node that overlays trajectory paths, goals, and obstacles onto a camera image.
    Uses TF2 to transform points into the camera optical frame, projects them using OpenCV,
    and republishes the annotated image.
    """

    def __init__(self):
        super().__init__('image_overlay_node')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('predicted_path_topic', '/mpc/predicted_path')
        self.declare_parameter('reference_path_topic', '/mpc/reference_path')
        self.declare_parameter('goal_point_topic', '/mpc/goal_point')
        self.declare_parameter('obstacle_topic', '/fake_obstacles/object_array')
        self.declare_parameter('odom_topic', '/odometry/local')
        self.declare_parameter('drive_topic', '/drive')
        self.declare_parameter('solve_time_topic', '/mpc/solve_time')
        self.declare_parameter('output_topic', '/camera/image_overlayed')
        self.declare_parameter('camera_optical_frame_override', '')

        self.declare_parameter('draw_reference_path', True)
        self.declare_parameter('draw_predicted_path', True)
        self.declare_parameter('draw_goal_point', True)
        self.declare_parameter('draw_obstacles', True)
        self.declare_parameter('draw_hud', True)
        self.declare_parameter('color_trajectory_by_speed', True)
        self.declare_parameter('max_visualization_speed', 10.0)

        # Retrieve parameter values
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.predicted_path_topic = self.get_parameter('predicted_path_topic').value
        self.reference_path_topic = self.get_parameter('reference_path_topic').value
        self.goal_point_topic = self.get_parameter('goal_point_topic').value
        self.obstacle_topic = self.get_parameter('obstacle_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.drive_topic = self.get_parameter('drive_topic').value
        self.solve_time_topic = self.get_parameter('solve_time_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.camera_frame_override = self.get_parameter('camera_optical_frame_override').value

        self.draw_reference_path = self.get_parameter('draw_reference_path').value
        self.draw_predicted_path = self.get_parameter('draw_predicted_path').value
        self.draw_goal_point = self.get_parameter('draw_goal_point').value
        self.draw_obstacles = self.get_parameter('draw_obstacles').value
        self.draw_hud = self.get_parameter('draw_hud').value
        self.color_trajectory_by_speed = self.get_parameter('color_trajectory_by_speed').value
        self.max_visualization_speed = self.get_parameter('max_visualization_speed').value

        # Initialize bridge and TF
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Thread-safe caching mutex
        self.mutex = threading.Lock()

        # Cache variables
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.camera_frame_id: Optional[str] = None

        self.latest_predicted_path: Optional[Path] = None
        self.latest_reference_path: Optional[Path] = None
        self.latest_goal_point: Optional[PointStamped] = None
        self.latest_obstacles = []

        self.latest_odom_speed = 0.0
        self.latest_drive_speed = 0.0
        self.latest_drive_steering = 0.0
        self.latest_solve_time = 0.0

        # QoS Settings
        latching_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # Setup subscribers
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, latching_qos)
        self.create_subscription(Path, self.predicted_path_topic, self.predicted_path_callback, 10)
        self.create_subscription(Path, self.reference_path_topic, self.reference_path_callback, 10)
        self.create_subscription(PointStamped, self.goal_point_topic, self.goal_point_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.create_subscription(AckermannDriveStamped, self.drive_topic, self.drive_callback, 10)
        self.create_subscription(Float32, self.solve_time_topic, self.solve_time_callback, 10)

        if OBSTACLES_AVAILABLE:
            self.create_subscription(ObjectArray, self.obstacle_topic, self.obstacles_callback, 10)
            self.get_logger().info(f"Obstacle tracking enabled on: {self.obstacle_topic}")
        else:
            self.get_logger().warn("Obstacle array messages not available (derived_object_msgs missing). Obstacles won't be drawn.")

        # Setup publisher
        self.pub_image = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info("Image Overlay Node initialized successfully.")

    # ------------------------------------------------------------------
    # Cache callbacks
    # ------------------------------------------------------------------

    def camera_info_callback(self, msg: CameraInfo):
        with self.mutex:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.camera_frame_id = msg.header.frame_id

    def predicted_path_callback(self, msg: Path):
        with self.mutex:
            self.latest_predicted_path = msg

    def reference_path_callback(self, msg: Path):
        with self.mutex:
            self.latest_reference_path = msg

    def goal_point_callback(self, msg: PointStamped):
        with self.mutex:
            self.latest_goal_point = msg

    def odom_callback(self, msg: Odometry):
        with self.mutex:
            self.latest_odom_speed = msg.twist.twist.linear.x

    def drive_callback(self, msg: AckermannDriveStamped):
        with self.mutex:
            self.latest_drive_speed = msg.drive.speed
            self.latest_drive_steering = msg.drive.steering_angle

    def solve_time_callback(self, msg: Float32):
        with self.mutex:
            self.latest_solve_time = msg.data

    def obstacles_callback(self, msg: 'ObjectArray'):
        obstacles = []
        for obj in msg.objects:
            pos = [obj.pose.position.x, obj.pose.position.y, obj.pose.position.z]
            orientation = [obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w]

            shape_type = {
                obj.shape.BOX: 'BOX',
                obj.shape.SPHERE: 'SPHERE',
                obj.shape.CYLINDER: 'CYLINDER',
            }.get(obj.shape.type, 'BOX')

            dims = list(obj.shape.dimensions)
            obstacles.append({
                'frame_id': msg.header.frame_id,
                'pos': pos,
                'orientation': orientation,
                'shape_type': shape_type,
                'dims': dims
            })
        with self.mutex:
            self.latest_obstacles = obstacles

    # ------------------------------------------------------------------
    # TF and Project Math
    # ------------------------------------------------------------------

    def transform_points_to_camera(self, points: np.ndarray, source_frame: str, target_frame: str, time_stamp) -> Optional[np.ndarray]:
        """
        Transforms N 3D points from source_frame to target_frame.
        First tries at message time_stamp, and falls back to latest transform.
        """
        try:
            tf_stamped = self.tf_buffer.lookup_transform(target_frame, source_frame, time_stamp)
        except Exception:
            try:
                tf_stamped = self.tf_buffer.lookup_transform(target_frame, source_frame, Time())
            except Exception as e:
                self.get_logger().warn(
                    f"TF lookup failed from '{source_frame}' to '{target_frame}': {e}",
                    throttle_duration_sec=5.0
                )
                return None

        # Apply rotation and translation
        t = tf_stamped.transform.translation
        r = tf_stamped.transform.rotation

        R_4x4 = tf_transformations.quaternion_matrix([r.x, r.y, r.z, r.w])
        R = R_4x4[:3, :3]
        T = np.array([t.x, t.y, t.z])

        # P_cam = R * P_source + T
        return np.dot(points, R.T) + T

    # ------------------------------------------------------------------
    # Image Callback and Drawing logic
    # ------------------------------------------------------------------

    def image_callback(self, msg: Image):
        # Convert image to OpenCV
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Snapshot of cached variables
        with self.mutex:
            camera_matrix = self.camera_matrix
            dist_coeffs = self.dist_coeffs
            camera_frame = self.camera_frame_override if self.camera_frame_override else self.camera_frame_id
            predicted_path = self.latest_predicted_path
            reference_path = self.latest_reference_path
            goal_point = self.latest_goal_point
            obstacles = list(self.latest_obstacles)
            odom_speed = self.latest_odom_speed
            drive_speed = self.latest_drive_speed
            drive_steering = self.latest_drive_steering
            solve_time = self.latest_solve_time

        # If camera calibration is not yet received, pass through raw image
        if camera_matrix is None or dist_coeffs is None or camera_frame is None:
            self.pub_image.publish(msg)
            return

        # Make drawing copy
        annotated_img = cv_img.copy()

        # Draw Reference Path (solid blue-gray)
        if self.draw_reference_path and reference_path is not None:
            self.draw_path(
                annotated_img, reference_path, camera_matrix, dist_coeffs, camera_frame,
                color=(255, 120, 0), thickness=2
            )

        # Draw Predicted Path (dynamic green to red gradient)
        if self.draw_predicted_path and predicted_path is not None:
            self.draw_path(
                annotated_img, predicted_path, camera_matrix, dist_coeffs, camera_frame,
                color=(0, 255, 0), thickness=3, is_predicted=True
            )

        # Draw Goal Point (tracking crosshairs target)
        if self.draw_goal_point and goal_point is not None:
            self.draw_goal(annotated_img, goal_point, camera_matrix, dist_coeffs, camera_frame)

        # Draw Obstacles (3D wireframes)
        if self.draw_obstacles and len(obstacles) > 0:
            self.draw_all_obstacles(annotated_img, obstacles, camera_matrix, dist_coeffs, camera_frame)

        # Draw Telemetry HUD
        if self.draw_hud:
            self.draw_hud_dashboard(annotated_img, odom_speed, drive_speed, drive_steering, solve_time)

        # Convert back and publish
        try:
            output_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
            output_msg.header = msg.header
            self.pub_image.publish(output_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish annotated image: {e}")

    # ------------------------------------------------------------------
    # Rendering Helpers
    # ------------------------------------------------------------------

    def draw_path(self, img: np.ndarray, path_msg: Path, K: np.ndarray, D: np.ndarray, camera_frame: str,
                  color=(255, 0, 0), thickness=2, is_predicted=False):
        source_frame = path_msg.header.frame_id
        if not source_frame or len(path_msg.poses) < 2:
            return

        # Extract path positions
        points_3d = []
        for pose in path_msg.poses:
            p = pose.pose.position
            points_3d.append([p.x, p.y, p.z])

        points_3d = np.array(points_3d, dtype=np.float32)

        # Transform points to camera optical frame
        pts_cam = self.transform_points_to_camera(points_3d, source_frame, camera_frame, path_msg.header.stamp)
        if pts_cam is None:
            return

        # Filter out points behind or too close to the camera (z < 0.1m)
        valid_indices = [i for i in range(len(pts_cam)) if pts_cam[i, 2] > 0.1]
        if len(valid_indices) < 2:
            return

        # Project valid coordinates
        pts_to_project = pts_cam[valid_indices]
        img_pts, _ = cv2.projectPoints(pts_to_project, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), K, D)
        img_pts = img_pts.reshape(-1, 2)

        # Map back to original indices
        proj_map = {valid_indices[i]: img_pts[i] for i in range(len(valid_indices))}

        # Draw lines between sequential valid segments
        n_segments = len(pts_cam) - 1
        for i in range(n_segments):
            if i in proj_map and (i + 1) in proj_map:
                pt1 = proj_map[i]
                pt2 = proj_map[i + 1]

                if not (np.isfinite(pt1).all() and np.isfinite(pt2).all()):
                    continue

                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))

                # If predicted, apply premium color gradient (green near vehicle -> orange/red at horizon)
                if is_predicted:
                    t = i / n_segments
                    r_val = int(255 * t)
                    g_val = int(255 * (1 - 0.5 * t))
                    b_val = int(50 * t)
                    seg_color = (b_val, g_val, r_val)
                else:
                    seg_color = color

                cv2.line(img, p1, p2, seg_color, thickness, lineType=cv2.LINE_AA)

    def draw_goal(self, img: np.ndarray, goal_msg: PointStamped, K: np.ndarray, D: np.ndarray, camera_frame: str):
        source_frame = goal_msg.header.frame_id
        if not source_frame:
            return

        # Assume goal is on the ground plane (z = 0)
        pt_3d = np.array([[goal_msg.point.x, goal_msg.point.y, 0.0]], dtype=np.float32)
        pt_cam = self.transform_points_to_camera(pt_3d, source_frame, camera_frame, goal_msg.header.stamp)

        if pt_cam is None or pt_cam[0, 2] <= 0.1:
            return

        img_pt, _ = cv2.projectPoints(pt_cam, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), K, D)
        img_pt = img_pt.reshape(2)

        if np.isfinite(img_pt).all():
            center = (int(round(img_pt[0])), int(round(img_pt[1])))
            color = (0, 0, 255) # Red target

            # Premium tracking crosshairs
            cv2.circle(img, center, 14, color, 2, lineType=cv2.LINE_AA)
            cv2.circle(img, center, 4, color, -1, lineType=cv2.LINE_AA)
            cv2.line(img, (center[0] - 20, center[1]), (center[0] + 20, center[1]), color, 2, lineType=cv2.LINE_AA)
            cv2.line(img, (center[0], center[1] - 20), (center[0], center[1] + 20), color, 2, lineType=cv2.LINE_AA)

    def draw_all_obstacles(self, img: np.ndarray, obstacles: list, K: np.ndarray, D: np.ndarray, camera_frame: str):
        for obs in obstacles:
            frame_id = obs['frame_id']
            pos = obs['pos']
            orientation = obs['orientation']
            shape_type = obs['shape_type']
            dims = obs['dims']

            # Object orientation matrix
            R_obj = tf_transformations.quaternion_matrix(orientation)[:3, :3]

            if shape_type == 'BOX' and len(dims) >= 3:
                l, w, h = dims[:3]
                # Bounding box corner coordinates relative to shape origin
                x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
                y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
                z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
                corners_local = np.vstack([x_corners, y_corners, z_corners]).T

                # Rotate and translate corners to obstacle frame
                corners_parent = np.dot(corners_local, R_obj.T) + np.array(pos)

                # Transform to camera optical frame
                corners_cam = self.transform_points_to_camera(corners_parent, frame_id, camera_frame, Time())
                if corners_cam is None or not np.any(corners_cam[:, 2] > 0.1):
                    continue

                # Project corners to image
                img_pts, _ = cv2.projectPoints(corners_cam, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), K, D)
                img_pts = img_pts.reshape(-1, 2)

                # Draw wireframe lines (Orange)
                self.draw_wireframe_box(img, img_pts, color=(0, 140, 255))

            elif shape_type == 'CYLINDER' and len(dims) >= 2:
                r, h = dims[:2]
                theta = np.linspace(0, 2 * np.pi, 16)
                top_circle = np.vstack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, h / 2)]).T
                bot_circle = np.vstack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, -h / 2)]).T

                cylinder_local = np.vstack([top_circle, bot_circle])
                cylinder_parent = np.dot(cylinder_local, R_obj.T) + np.array(pos)

                cylinder_cam = self.transform_points_to_camera(cylinder_parent, frame_id, camera_frame, Time())
                if cylinder_cam is None or not np.any(cylinder_cam[:, 2] > 0.1):
                    continue

                img_pts, _ = cv2.projectPoints(cylinder_cam, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), K, D)
                img_pts = img_pts.reshape(-1, 2)

                top_pts = img_pts[:16]
                bot_pts = img_pts[16:]

                # Draw top/bottom circle wireframes (Yellow)
                self.draw_polygon(img, top_pts, color=(0, 220, 220))
                self.draw_polygon(img, bot_pts, color=(0, 220, 220))

                # Connect pillars
                for idx in [0, 4, 8, 12]:
                    pt1 = (int(round(top_pts[idx][0])), int(round(top_pts[idx][1])))
                    pt2 = (int(round(bot_pts[idx][0])), int(round(bot_pts[idx][1])))
                    cv2.line(img, pt1, pt2, (0, 220, 220), 2, lineType=cv2.LINE_AA)

            elif shape_type == 'SPHERE' and len(dims) >= 1:
                r = dims[0]
                theta = np.linspace(0, 2 * np.pi, 16)
                c_xy = np.vstack([r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)]).T
                c_yz = np.vstack([np.zeros_like(theta), r * np.cos(theta), r * np.sin(theta)]).T

                sphere_local = np.vstack([c_xy, c_yz])
                sphere_parent = np.dot(sphere_local, R_obj.T) + np.array(pos)

                sphere_cam = self.transform_points_to_camera(sphere_parent, frame_id, camera_frame, Time())
                if sphere_cam is None or not np.any(sphere_cam[:, 2] > 0.1):
                    continue

                img_pts, _ = cv2.projectPoints(sphere_cam, np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), K, D)
                img_pts = img_pts.reshape(-1, 2)

                xy_pts = img_pts[:16]
                yz_pts = img_pts[16:]

                # Draw circle wireframes (Magenta)
                self.draw_polygon(img, xy_pts, color=(200, 0, 200))
                self.draw_polygon(img, yz_pts, color=(200, 0, 200))

    def draw_wireframe_box(self, img: np.ndarray, img_pts: np.ndarray, color):
        pts = [(int(round(pt[0])), int(round(pt[1]))) for pt in img_pts]

        # Draw top face (0-1-2-3-0)
        for i in range(4):
            cv2.line(img, pts[i], pts[(i + 1) % 4], color, 2, lineType=cv2.LINE_AA)

        # Draw bottom face (4-5-6-7-4)
        for i in range(4):
            cv2.line(img, pts[i + 4], pts[(i + 1) % 4 + 4], color, 2, lineType=cv2.LINE_AA)

        # Draw vertical pillars connecting faces
        for i in range(4):
            cv2.line(img, pts[i], pts[i + 4], color, 2, lineType=cv2.LINE_AA)

    def draw_polygon(self, img: np.ndarray, pts: np.ndarray, color):
        pts_int = [(int(round(pt[0])), int(round(pt[1]))) for pt in pts]
        for i in range(len(pts_int)):
            cv2.line(img, pts_int[i], pts_int[(i + 1) % len(pts_int)], color, 2, lineType=cv2.LINE_AA)

    def draw_hud_dashboard(self, img: np.ndarray, odom_speed: float, drive_speed: float, drive_steering: float, solve_time: float):
        h, w = img.shape[:2]

        # Draw box in top-left corner
        x1, y1 = 15, 15
        x2, y2 = 320, 160

        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)  # Dark solid background
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 140, 255), 1)   # Orange border
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (255, 255, 255)
        label_color = (0, 140, 255)

        # Title
        cv2.putText(img, "MPC TELEMETRY HUD", (x1 + 10, y1 + 25), font, 0.6, label_color, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (x1 + 10, y1 + 32), (x2 - 10, y1 + 32), (80, 80, 80), 1, lineType=cv2.LINE_AA)

        # Rows
        cv2.putText(img, "Odom Speed:", (x1 + 10, y1 + 55), font, font_scale, label_color, thickness, lineType=cv2.LINE_AA)
        cv2.putText(img, f"{odom_speed:.2f} m/s", (x1 + 140, y1 + 55), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

        cv2.putText(img, "Target Speed:", (x1 + 10, y1 + 78), font, font_scale, label_color, thickness, lineType=cv2.LINE_AA)
        cv2.putText(img, f"{drive_speed:.2f} m/s", (x1 + 140, y1 + 78), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

        cv2.putText(img, "Target Steer:", (x1 + 10, y1 + 101), font, font_scale, label_color, thickness, lineType=cv2.LINE_AA)
        cv2.putText(img, f"{math.degrees(drive_steering):.1f} deg", (x1 + 140, y1 + 101), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)

        cv2.putText(img, "Solve Time:", (x1 + 10, y1 + 124), font, font_scale, label_color, thickness, lineType=cv2.LINE_AA)
        cv2.putText(img, f"{solve_time * 1000.0:.1f} ms", (x1 + 140, y1 + 124), font, font_scale, text_color, thickness, lineType=cv2.LINE_AA)


def main(args=None):
    rclpy.init(args=args)
    node = ImageOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
