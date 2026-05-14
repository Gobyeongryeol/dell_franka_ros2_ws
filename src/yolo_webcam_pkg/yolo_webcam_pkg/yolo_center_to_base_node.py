#!/usr/bin/env python3
import json
import math
import re
import time

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

import tf2_geometry_msgs  # noqa: F401  Registers PointStamped transforms.
import tf2_ros


class YoloCenterToBaseNode(Node):
    def __init__(self):
        super().__init__('yolo_center_to_base_node')

        self.declare_parameter('center_topic', '/yolo/center')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('output_topic', '/yolo/target_base')
        self.declare_parameter('source_frame', 'camera_color_optical_frame')
        self.declare_parameter('target_frame', 'fr3_link0')
        self.declare_parameter('window_size', 7)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('debug', True)

        self.center_topic = self.get_parameter('center_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.source_frame = self.get_parameter('source_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.window_size = int(self.get_parameter('window_size').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.debug = bool(self.get_parameter('debug').value)

        self.bridge = CvBridge()
        self.depth_image = None
        self.depth_encoding = ''
        self.depth_stamp = None
        self.camera_info = None
        self.last_publish_time = 0.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.create_subscription(String, self.center_topic, self.center_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.get_logger().info('YOLO center-to-base node started')
        self.get_logger().info(f'center_topic={self.center_topic}')
        self.get_logger().info(f'depth_topic={self.depth_topic}')
        self.get_logger().info(f'camera_info_topic={self.camera_info_topic}')
        self.get_logger().info(f'output_topic={self.output_topic}')
        self.get_logger().info(f'source_frame={self.source_frame}')
        self.get_logger().info(f'target_frame={self.target_frame}')
        self.get_logger().info(f'window_size={self.window_size}')
        self.get_logger().info(f'publish_rate_hz={self.publish_rate_hz}')
        self.get_logger().info(f'debug={self.debug}')

    def depth_callback(self, msg: Image):
        self.depth_encoding = msg.encoding
        self.depth_stamp = msg.header.stamp
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def parse_center(self, data: str):
        text = data.strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, list) and obj:
                obj = obj[0]
            if isinstance(obj, dict):
                for x_key, y_key in (
                    ('u', 'v'),
                    ('pixel_u', 'pixel_v'),
                    ('center_x', 'center_y'),
                    ('cx', 'cy'),
                    ('x', 'y'),
                ):
                    if x_key in obj and y_key in obj:
                        return int(round(float(obj[x_key]))), int(round(float(obj[y_key]))), obj
        except Exception:
            pass

        parts = [p.strip() for p in text.split(',')]
        if len(parts) >= 3:
            try:
                return int(round(float(parts[1]))), int(round(float(parts[2]))), {'raw': text}
            except ValueError:
                pass

        mx = re.search(r'(?:pixel_u|center_x|cx|u|x)\s*[:=]\s*(-?\d+(?:\.\d+)?)', text)
        my = re.search(r'(?:pixel_v|center_y|cy|v|y)\s*[:=]\s*(-?\d+(?:\.\d+)?)', text)
        if mx and my:
            return int(round(float(mx.group(1)))), int(round(float(my.group(1)))), {'raw': text}

        nums = re.findall(r'-?\d+(?:\.\d+)?', text)
        if len(nums) >= 2:
            return int(round(float(nums[-2]))), int(round(float(nums[-1]))), {'raw': text}

        return None, None, {'raw': text}

    def raw_depth_to_m(self, raw):
        if raw is None:
            return None

        value = float(raw)
        if self.depth_encoding in ('16UC1', 'mono16'):
            value *= 0.001

        return value

    @staticmethod
    def valid_depth(value):
        return value is not None and math.isfinite(value) and value > 0.0

    def depth_at_pixel_m(self, u: int, v: int):
        if self.depth_image is None:
            self.get_logger().warn('Waiting for aligned depth image...')
            return None

        h, w = self.depth_image.shape[:2]
        if u < 0 or u >= w or v < 0 or v >= h:
            self.get_logger().warn(f'Pixel out of range: u={u}, v={v}, depth_image={w}x{h}')
            return None

        center_depth = self.raw_depth_to_m(self.depth_image[v, u])
        if self.valid_depth(center_depth):
            return center_depth

        radius = max(1, self.window_size // 2)
        x1 = max(0, u - radius)
        x2 = min(w, u + radius + 1)
        y1 = max(0, v - radius)
        y2 = min(h, v + radius + 1)

        patch = self.depth_image[y1:y2, x1:x2].astype(np.float32)
        valid = patch[np.isfinite(patch)]
        valid = valid[valid > 0]

        if valid.size == 0:
            self.get_logger().warn(
                f'Invalid depth at u={u}, v={v}; no valid values in {self.window_size}x{self.window_size}'
            )
            return None

        median_raw = float(np.median(valid))
        depth_m = self.raw_depth_to_m(median_raw)
        self.get_logger().warn(
            f'Center depth invalid at u={u}, v={v}; using window median depth_m={depth_m:.4f}'
        )
        return depth_m

    def rate_limited(self):
        if self.publish_rate_hz <= 0.0:
            return False

        now = time.monotonic()
        min_period = 1.0 / self.publish_rate_hz
        if now - self.last_publish_time < min_period:
            return True

        self.last_publish_time = now
        return False

    def center_callback(self, msg: String):
        if self.rate_limited():
            return

        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera_info...')
            return

        u, v, raw_obj = self.parse_center(msg.data)
        if u is None or v is None:
            self.get_logger().warn(f'Failed to parse /yolo/center: {msg.data}')
            return

        if self.debug:
            self.get_logger().info(f'/yolo/center received: u={u}, v={v}, raw={msg.data}')

        depth_m = self.depth_at_pixel_m(u, v)
        if not self.valid_depth(depth_m):
            self.get_logger().warn(f'Depth lookup failed for u={u}, v={v}')
            return

        k = self.camera_info.k
        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])
        if fx == 0.0 or fy == 0.0:
            self.get_logger().warn(f'Invalid camera intrinsics: fx={fx}, fy={fy}')
            return

        x_cam = (float(u) - cx) * depth_m / fx
        y_cam = (float(v) - cy) * depth_m / fy
        z_cam = depth_m

        point_camera = PointStamped()
        point_camera.header.stamp = Time().to_msg()
        point_camera.header.frame_id = self.source_frame
        point_camera.point.x = x_cam
        point_camera.point.y = y_cam
        point_camera.point.z = z_cam

        try:
            point_base = self.tf_buffer.transform(
                point_camera,
                self.target_frame,
                timeout=Duration(seconds=0.2)
            )
        except Exception as exc:
            self.get_logger().error(
                f'TF transform failed: {self.source_frame} -> {self.target_frame}: {exc}'
            )
            return

        result = {
            'x_base': point_base.point.x,
            'y_base': point_base.point.y,
            'z_base': point_base.point.z,
            'target_frame': self.target_frame,
            'source_frame': self.source_frame,
            'u': u,
            'v': v,
            'depth_m': depth_m,
            'camera_x_m': x_cam,
            'camera_y_m': y_cam,
            'camera_z_m': z_cam,
            'raw_detection': raw_obj,
        }

        self.pub.publish(String(data=json.dumps(result, ensure_ascii=False)))

        if self.debug:
            self.get_logger().info(
                f'depth_m={depth_m:.4f} camera_xyz=({x_cam:.4f}, {y_cam:.4f}, {z_cam:.4f}) '
                f'base_xyz=({point_base.point.x:.4f}, {point_base.point.y:.4f}, {point_base.point.z:.4f})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = YoloCenterToBaseNode()

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
