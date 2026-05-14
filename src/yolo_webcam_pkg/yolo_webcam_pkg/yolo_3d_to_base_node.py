#!/usr/bin/env python3
import json

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

import tf2_ros
import tf2_geometry_msgs


class Yolo3DToBaseNode(Node):
    def __init__(self):
        super().__init__('yolo_3d_to_base_node')

        self.declare_parameter('input_topic', '/yolo/target_3d')
        self.declare_parameter('output_topic', '/yolo/target_base')
        self.declare_parameter('target_frame', 'base')
        self.declare_parameter('source_frame_override', '')

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.source_frame_override = self.get_parameter('source_frame_override').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.sub = self.create_subscription(String, self.input_topic, self.target_callback, 10)

        self.get_logger().info('YOLO 3D to base node started')
        self.get_logger().info(f'input_topic={self.input_topic}')
        self.get_logger().info(f'output_topic={self.output_topic}')
        self.get_logger().info(f'target_frame={self.target_frame}')

    def target_callback(self, msg: String):
        try:
            data = json.loads(msg.data)

            source_frame = self.source_frame_override or data.get('frame_id', 'camera_color_optical_frame')

            p = PointStamped()
            p.header.frame_id = source_frame
            p.header.stamp = Time().to_msg()
            p.point.x = float(data['x_m'])
            p.point.y = float(data['y_m'])
            p.point.z = float(data['z_m'])

            tf = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            out = tf2_geometry_msgs.do_transform_point(p, tf)

            result = {
                'x_base': out.point.x,
                'y_base': out.point.y,
                'z_base': out.point.z,
                'target_frame': self.target_frame,
                'source_frame': source_frame,
                'pixel_u': data.get('pixel_u'),
                'pixel_v': data.get('pixel_v'),
                'camera_x_m': data.get('x_m'),
                'camera_y_m': data.get('y_m'),
                'camera_z_m': data.get('z_m'),
                'raw': data.get('raw_detection', data),
            }

            self.pub.publish(String(data=json.dumps(result, ensure_ascii=False)))

            self.get_logger().info(
                f"target_base=({out.point.x:.3f}, {out.point.y:.3f}, {out.point.z:.3f}) "
                f"from camera=({p.point.x:.3f}, {p.point.y:.3f}, {p.point.z:.3f}) "
                f"frame={source_frame}"
            )

        except Exception as e:
            self.get_logger().warn(f'Failed to transform /yolo/target_3d to base: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = Yolo3DToBaseNode()

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
