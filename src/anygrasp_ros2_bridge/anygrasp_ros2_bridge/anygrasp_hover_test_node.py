import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _finite_float(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not a number') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} is not finite')
    return value


def _read_xyz(payload, key):
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'{key} is missing or not an object')
    return {
        'x': _finite_float(value.get('x'), f'{key}.x'),
        'y': _finite_float(value.get('y'), f'{key}.y'),
        'z': _finite_float(value.get('z'), f'{key}.z'),
    }


class AnyGraspHoverTestNode(Node):
    def __init__(self):
        super().__init__('anygrasp_hover_test_node')

        self.declare_parameter('input_topic', '/anygrasp/best_grasp_pose_base')
        self.declare_parameter('dry_run', True)
        self.declare_parameter('plan_only', True)
        self.declare_parameter('once', True)
        self.declare_parameter('min_score', 0.2)
        self.declare_parameter('x_min', 0.20)
        self.declare_parameter('x_max', 0.70)
        self.declare_parameter('y_min', -0.50)
        self.declare_parameter('y_max', 0.50)
        self.declare_parameter('z_min', 0.20)
        self.declare_parameter('z_max', 0.80)
        self.declare_parameter('hover_z_min', 0.20)
        self.declare_parameter('hover_z_max', 0.90)
        self.declare_parameter('max_step', 0.25)

        self.input_topic = self.get_parameter('input_topic').value
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.plan_only = bool(self.get_parameter('plan_only').value)
        self.once = bool(self.get_parameter('once').value)
        self.min_score = float(self.get_parameter('min_score').value)
        self.x_min = float(self.get_parameter('x_min').value)
        self.x_max = float(self.get_parameter('x_max').value)
        self.y_min = float(self.get_parameter('y_min').value)
        self.y_max = float(self.get_parameter('y_max').value)
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)
        self.hover_z_min = float(self.get_parameter('hover_z_min').value)
        self.hover_z_max = float(self.get_parameter('hover_z_max').value)
        self.max_step = float(self.get_parameter('max_step').value)

        self.processed_once = False
        self.subscription = self.create_subscription(String, self.input_topic, self._callback, 10)

        self.get_logger().info(f'Subscribed AnyGrasp base pose: {self.input_topic}')
        self.get_logger().info(f'dry_run={self.dry_run}, plan_only={self.plan_only}, once={self.once}')

    def _callback(self, msg):
        if self.once and self.processed_once:
            return
        self.processed_once = True

        try:
            payload = json.loads(msg.data)
            score, grasp_translation, hover_translation = self._parse_payload(payload)

            self.get_logger().info('received AnyGrasp base pose')
            self.get_logger().info(f'score={score:.4f}')
            self.get_logger().info(
                'grasp_translation='
                f'({grasp_translation["x"]:.3f}, {grasp_translation["y"]:.3f}, {grasp_translation["z"]:.3f})'
            )
            self.get_logger().info(
                'hover_translation='
                f'({hover_translation["x"]:.3f}, {hover_translation["y"]:.3f}, {hover_translation["z"]:.3f})'
            )

            rejected_reason = self._safety_rejected_reason(score, grasp_translation, hover_translation)
            if rejected_reason:
                self.get_logger().warn(f'safety check rejected: {rejected_reason}')
            else:
                self.get_logger().info('safety check passed')

            if self.dry_run:
                self.get_logger().info('dry_run=true, not executing')
            else:
                self.get_logger().warn('dry_run=false requested, but this node has no robot execution code; not executing')

        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Invalid JSON: {exc}')
        except Exception as exc:
            self.get_logger().warn(f'safety check rejected: {repr(exc)}')
            self.get_logger().info('dry_run=true, not executing')

    def _parse_payload(self, payload):
        header = payload.get('header', {})
        frame_id = header.get('frame_id')
        if frame_id != 'base':
            raise ValueError(f'frame_id must be base, got {frame_id!r}')

        score = _finite_float(payload.get('score'), 'score')
        grasp_translation = _read_xyz(payload, 'translation')
        hover_translation = _read_xyz(payload, 'hover_translation')
        return score, grasp_translation, hover_translation

    def _safety_rejected_reason(self, score, grasp, hover):
        if score < self.min_score:
            return f'score {score:.4f} < min_score {self.min_score:.4f}'

        if not (self.x_min <= grasp['x'] <= self.x_max):
            return f'grasp x {grasp["x"]:.3f} outside [{self.x_min:.3f}, {self.x_max:.3f}]'
        if not (self.y_min <= grasp['y'] <= self.y_max):
            return f'grasp y {grasp["y"]:.3f} outside [{self.y_min:.3f}, {self.y_max:.3f}]'
        if not (self.z_min <= grasp['z'] <= self.z_max):
            return f'grasp z {grasp["z"]:.3f} outside [{self.z_min:.3f}, {self.z_max:.3f}]'

        if not (self.x_min <= hover['x'] <= self.x_max):
            return f'hover x {hover["x"]:.3f} outside [{self.x_min:.3f}, {self.x_max:.3f}]'
        if not (self.y_min <= hover['y'] <= self.y_max):
            return f'hover y {hover["y"]:.3f} outside [{self.y_min:.3f}, {self.y_max:.3f}]'
        if not (self.hover_z_min <= hover['z'] <= self.hover_z_max):
            return f'hover z {hover["z"]:.3f} outside [{self.hover_z_min:.3f}, {self.hover_z_max:.3f}]'

        step = math.sqrt(
            (hover['x'] - grasp['x']) ** 2
            + (hover['y'] - grasp['y']) ** 2
            + (hover['z'] - grasp['z']) ** 2
        )
        if step > self.max_step:
            return f'hover step {step:.3f} > max_step {self.max_step:.3f}'

        return ''


def main(args=None):
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = AnyGraspHoverTestNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
