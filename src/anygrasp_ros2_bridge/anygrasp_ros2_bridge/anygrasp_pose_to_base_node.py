import json
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
import tf2_ros


def _is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _quat_to_matrix(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError('zero-length quaternion in TF')
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _mat_vec_mul(matrix, vector):
    return [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]


def _mat_mat_mul(a, b):
    return [
        [
            a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


def _validate_rotation_matrix(value):
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError('rotation_matrix must be a 3x3 list')
    matrix = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError('rotation_matrix must be a 3x3 list')
        if not all(_is_finite_number(item) for item in row):
            raise ValueError('rotation_matrix contains non-finite values')
        matrix.append([float(item) for item in row])
    return matrix


class AnyGraspPoseToBaseNode(Node):
    def __init__(self):
        super().__init__('anygrasp_pose_to_base_node')

        self.declare_parameter('input_topic', '/anygrasp/best_grasp_pose')
        self.declare_parameter('output_topic', '/anygrasp/best_grasp_pose_base')
        self.declare_parameter('target_frame', 'base')
        self.declare_parameter('source_frame_override', '')
        self.declare_parameter('min_score', -1.0)
        self.declare_parameter('add_hover_pose', True)
        self.declare_parameter('hover_offset_z', 0.10)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.source_frame_override = self.get_parameter('source_frame_override').value
        self.min_score = float(self.get_parameter('min_score').value)
        self.add_hover_pose = bool(self.get_parameter('add_hover_pose').value)
        self.hover_offset_z = float(self.get_parameter('hover_offset_z').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(String, self.output_topic, 10)
        self.subscription = self.create_subscription(String, self.input_topic, self._callback, 10)

        self.get_logger().info(f'Subscribed grasp JSON: {self.input_topic}')
        self.get_logger().info(f'Publishing base-frame grasp JSON: {self.output_topic}')
        self.get_logger().info(f'Target frame: {self.target_frame}')

    def _callback(self, msg):
        try:
            payload = json.loads(msg.data)
            output = self._transform_payload(payload)
            if output is None:
                return

            out_msg = String()
            out_msg.data = json.dumps(output)
            self.publisher.publish(out_msg)

            translation = output['translation']
            self.get_logger().info(
                f'Published grasp in {self.target_frame}: '
                f'score={output["score"]:.4f}, '
                f't=({translation["x"]:.3f}, {translation["y"]:.3f}, {translation["z"]:.3f})'
            )
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Invalid JSON on {self.input_topic}: {exc}')
        except Exception as exc:
            self.get_logger().warn(f'Failed to transform grasp pose: {repr(exc)}')

    def _transform_payload(self, payload):
        if payload.get('error'):
            self.get_logger().warn(f'Input grasp message contains error={payload.get("error")}; skipping.')
            return None

        header = payload.get('header', {})
        source_frame = self.source_frame_override or header.get('frame_id', 'camera_color_optical_frame')
        if not source_frame:
            source_frame = 'camera_color_optical_frame'

        score = float(payload.get('score', float('nan')))
        if not math.isfinite(score):
            raise ValueError('score is missing or non-finite')
        if score < self.min_score:
            self.get_logger().warn(f'Grasp score {score:.4f} is below min_score {self.min_score:.4f}; skipping.')
            return None

        translation_payload = payload.get('translation', {})
        camera_translation = [
            float(translation_payload.get('x', float('nan'))),
            float(translation_payload.get('y', float('nan'))),
            float(translation_payload.get('z', float('nan'))),
        ]
        if not all(math.isfinite(v) for v in camera_translation):
            raise ValueError('translation contains missing or non-finite values')

        camera_rotation = _validate_rotation_matrix(payload.get('rotation_matrix'))

        transform = self.tf_buffer.lookup_transform(self.target_frame, source_frame, Time())
        base_rotation_tf = _quat_to_matrix(
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        )
        base_translation_tf = [
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
            float(transform.transform.translation.z),
        ]

        rotated_translation = _mat_vec_mul(base_rotation_tf, camera_translation)
        base_translation = [
            base_translation_tf[0] + rotated_translation[0],
            base_translation_tf[1] + rotated_translation[1],
            base_translation_tf[2] + rotated_translation[2],
        ]
        base_rotation = _mat_mat_mul(base_rotation_tf, camera_rotation)

        if not all(math.isfinite(v) for v in base_translation):
            raise ValueError('base translation contains non-finite values')
        for row in base_rotation:
            if not all(math.isfinite(v) for v in row):
                raise ValueError('base rotation contains non-finite values')

        output = {
            'header': {
                'frame_id': self.target_frame,
                'source_frame_id': source_frame,
                'stamp_sec': int(header.get('stamp_sec', 0)),
                'stamp_nanosec': int(header.get('stamp_nanosec', 0)),
            },
            'score': score,
            'translation': {
                'x': base_translation[0],
                'y': base_translation[1],
                'z': base_translation[2],
            },
            'rotation_matrix': base_rotation,
            'width': float(payload.get('width', 0.0)),
            'height': float(payload.get('height', 0.0)),
            'depth': float(payload.get('depth', 0.0)),
            'object_id': int(payload.get('object_id', -1)),
        }

        if self.add_hover_pose:
            output['hover_translation'] = {
                'x': base_translation[0],
                'y': base_translation[1],
                'z': base_translation[2] + self.hover_offset_z,
            }

        return output


def main(args=None):
    rclpy.init(args=args)
    node = AnyGraspPoseToBaseNode()
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
