#!/usr/bin/env python3
import json
import re
import math
import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class YoloDepthTo3D(Node):
    def __init__(self):
        super().__init__('yolo_depth_to_3d_node')

        self.declare_parameter('center_topic', '/yolo/center')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('patch_size', 7)

        center_topic = self.get_parameter('center_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value

        self.patch_size = int(self.get_parameter('patch_size').value)

        self.bridge = CvBridge()
        self.depth_image = None
        self.depth_encoding = None
        self.camera_info = None

        self.target_pub = self.create_publisher(String, '/yolo/target_3d', 10)

        self.create_subscription(String, center_topic, self.center_callback, 10)
        self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        self.get_logger().info('YOLO depth-to-3D node started')
        self.get_logger().info(f'center_topic={center_topic}')
        self.get_logger().info(f'depth_topic={depth_topic}')
        self.get_logger().info(f'camera_info_topic={camera_info_topic}')

    def depth_callback(self, msg: Image):
        self.depth_encoding = msg.encoding
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def parse_center(self, data: str):
        """
        /yolo/center 문자열에서 중심좌표를 최대한 유연하게 파싱한다.
        지원:
        - JSON: {"center_x": 123, "center_y": 45}
        - JSON: {"x": 123, "y": 45}
        - 문자열 내부 center_x=123 center_y=45
        """
        data = data.strip()

        try:
            obj = json.loads(data)
            if isinstance(obj, list) and len(obj) > 0:
                obj = obj[0]

            if isinstance(obj, dict):
                if 'center_x' in obj and 'center_y' in obj:
                    return int(obj['center_x']), int(obj['center_y']), obj
                if 'cx' in obj and 'cy' in obj:
                    return int(obj['cx']), int(obj['cy']), obj
                if 'x' in obj and 'y' in obj:
                    return int(obj['x']), int(obj['y']), obj
        except Exception:
            pass

        # fallback regex
        mx = re.search(r'(?:center_x|cx|x)\s*[:=]\s*([0-9]+)', data)
        my = re.search(r'(?:center_y|cy|y)\s*[:=]\s*([0-9]+)', data)

        if mx and my:
            return int(mx.group(1)), int(my.group(1)), {'raw': data}

        nums = re.findall(r'\d+', data)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1]), {'raw': data}

        return None, None, {'raw': data}

    def get_depth_median_m(self, u: int, v: int):
        if self.depth_image is None:
            return None

        h, w = self.depth_image.shape[:2]

        if u < 0 or u >= w or v < 0 or v >= h:
            self.get_logger().warn(f'Pixel out of range: u={u}, v={v}, image={w}x{h}')
            return None

        r = max(1, self.patch_size // 2)
        x1 = max(0, u - r)
        x2 = min(w, u + r + 1)
        y1 = max(0, v - r)
        y2 = min(h, v + r + 1)

        patch = self.depth_image[y1:y2, x1:x2].astype(np.float32)
        valid = patch[np.isfinite(patch)]
        valid = valid[valid > 0]

        if valid.size == 0:
            return None

        z_raw = float(np.median(valid))

        # RealSense aligned depth is usually 16UC1 in millimeters.
        if self.depth_encoding in ['16UC1', 'mono16']:
            return z_raw / 1000.0

        # 32FC1 is usually meters.
        return z_raw

    def center_callback(self, msg: String):
        if self.camera_info is None:
            self.get_logger().warn('Waiting for camera_info...')
            return

        if self.depth_image is None:
            self.get_logger().warn('Waiting for aligned depth image...')
            return

        u, v, raw_obj = self.parse_center(msg.data)

        if u is None or v is None:
            self.get_logger().warn(f'Failed to parse center: {msg.data}')
            return

        z = self.get_depth_median_m(u, v)

        if z is None or z <= 0 or math.isnan(z):
            self.get_logger().warn(f'Invalid depth at pixel u={u}, v={v}')
            return

        k = self.camera_info.k
        fx = k[0]
        fy = k[4]
        cx = k[2]
        cy = k[5]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        result = {
            'pixel_u': u,
            'pixel_v': v,
            'x_m': x,
            'y_m': y,
            'z_m': z,
            'frame_id': self.camera_info.header.frame_id,
            'raw_detection': raw_obj,
        }

        self.target_pub.publish(String(data=json.dumps(result, ensure_ascii=False)))

        self.get_logger().info(
            f"target_3d: pixel=({u},{v}) "
            f"camera_xyz=({x:.3f}, {y:.3f}, {z:.3f}) m "
            f"frame={self.camera_info.header.frame_id}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthTo3D()

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
