#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from typing import Any

import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
import tf2_ros


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} is not a number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} is not finite')
    return result


def _read_xyz(payload: dict[str, Any], key: str) -> list[float]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f'{key} is missing or not an object')
    return [
        _finite_float(value.get('x'), f'{key}.x'),
        _finite_float(value.get('y'), f'{key}.y'),
        _finite_float(value.get('z'), f'{key}.z'),
    ]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _validate_rotation_matrix(value: Any) -> list[list[float]]:
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


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> list[list[float]]:
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


def _mat_vec_mul(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]


def _mat_mat_mul(
    a: list[list[float]],
    b: list[list[float]],
) -> list[list[float]]:
    return [
        [
            a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col]
            for col in range(3)
        ]
        for row in range(3)
    ]


class LLMAnyGraspPoseFilterNode(Node):
    def __init__(self):
        super().__init__('llm_anygrasp_pose_filter_node')

        self.declare_parameter('require_llm_target', True)
        self.declare_parameter('target_xy_radius', 0.12)
        self.declare_parameter('target_z_radius', 0.35)
        self.declare_parameter('input_grasp_topic', '/anygrasp/grasp_candidates')
        self.declare_parameter('output_grasp_topic', '/anygrasp/best_safe_grasp_pose_base')
        self.declare_parameter('yolo_target_topic', '/yolo/target_base')
        self.declare_parameter('llm_target_topic', '/llm/target_pick')
        self.declare_parameter('target_frame', 'base')
        self.declare_parameter('source_frame_default', 'camera_color_optical_frame')
        self.declare_parameter('hover_offset_z', 0.10)
        self.declare_parameter('tf_timeout_sec', 0.2)

        self.require_llm_target = bool(self.get_parameter('require_llm_target').value)
        self.target_xy_radius = float(self.get_parameter('target_xy_radius').value)
        self.target_z_radius = float(self.get_parameter('target_z_radius').value)
        self.input_grasp_topic = str(self.get_parameter('input_grasp_topic').value)
        self.output_grasp_topic = str(self.get_parameter('output_grasp_topic').value)
        self.yolo_target_topic = str(self.get_parameter('yolo_target_topic').value)
        self.llm_target_topic = str(self.get_parameter('llm_target_topic').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.source_frame_default = str(self.get_parameter('source_frame_default').value)
        self.hover_offset_z = float(self.get_parameter('hover_offset_z').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)

        self.armed = False
        self.target_label: str | None = None
        self.latest_command_id = None
        self.latest_yolo_target_base: tuple[float, float, float] | None = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.grasp_pub = self.create_publisher(String, self.output_grasp_topic, 10)
        self.create_subscription(String, self.llm_target_topic, self.llm_target_callback, 10)
        self.create_subscription(String, self.yolo_target_topic, self.yolo_target_callback, 10)
        self.create_subscription(String, self.input_grasp_topic, self.grasp_candidates_callback, 10)

        self.get_logger().info('LLM AnyGrasp pose filter node ready')
        self.get_logger().info(f'llm_target_topic={self.llm_target_topic}')
        self.get_logger().info(f'yolo_target_topic={self.yolo_target_topic}')
        self.get_logger().info(
            f'input_grasp_topic={self.input_grasp_topic} type=std_msgs/msg/String'
        )
        self.get_logger().info(
            f'output_grasp_topic={self.output_grasp_topic} type=std_msgs/msg/String'
        )
        self.get_logger().info(
            'filter params: '
            f'require_llm_target={self.require_llm_target}, '
            f'target_xy_radius={self.target_xy_radius:.3f}, '
            f'target_z_radius={self.target_z_radius:.3f}, '
            f'target_frame={self.target_frame}, '
            f'source_frame_default={self.source_frame_default}'
        )

    def llm_target_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(
                f'Failed to parse LLM target JSON on {self.llm_target_topic}: {exc}'
            )
            return

        label = payload.get('label')
        if not label:
            labels = payload.get('labels', [])
            if isinstance(labels, list) and labels:
                label = labels[0]

        if not label:
            self.get_logger().warn(f'LLM target message has no label: {msg.data}')
            return

        self.target_label = str(label)
        self.latest_command_id = payload.get('command_id')
        self.latest_yolo_target_base = None
        self.armed = True
        self.get_logger().info(
            f'Received new LLM target command_id={self.latest_command_id}, '
            f'label={self.target_label}. '
            'Cleared stale YOLO target.'
        )

    def yolo_target_callback(self, msg: String):
        if self.require_llm_target and not self.armed:
            return

        try:
            payload = json.loads(msg.data)
            x_base = _finite_float(payload['x_base'], 'x_base')
            y_base = _finite_float(payload['y_base'], 'y_base')
            z_base = _finite_float(payload['z_base'], 'z_base')
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(
                f'Failed to parse YOLO target base on {self.yolo_target_topic}: {exc}'
            )
            return

        yolo_label = self.extract_yolo_target_label(payload)
        if yolo_label and self.target_label and yolo_label != self.target_label:
            self.get_logger().warn(
                f'Ignoring YOLO target label={yolo_label}; current LLM target label={self.target_label}.'
            )
            return

        yolo_command_id = self.extract_yolo_command_id(payload)
        if (
            yolo_command_id is not None
            and self.latest_command_id is not None
            and str(yolo_command_id) != str(self.latest_command_id)
        ):
            self.get_logger().warn(
                f'Ignoring YOLO target command_id={yolo_command_id}; '
                f'current command_id={self.latest_command_id}.'
            )
            return

        self.latest_yolo_target_base = (x_base, y_base, z_base)
        self.get_logger().info(
            f'Updated YOLO target base for label={self.target_label}: '
            f'x={x_base:.3f}, y={y_base:.3f}, z={z_base:.3f}'
        )

    def grasp_candidates_callback(self, msg: String):
        if self.require_llm_target and not self.armed:
            return

        if self.latest_yolo_target_base is None:
            return

        try:
            payload = json.loads(msg.data)
            selected = self.select_candidate(payload)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(
                f'Invalid JSON on {self.input_grasp_topic}: {exc}'
            )
            return
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to select LLM-targeted AnyGrasp candidate: {repr(exc)}'
            )
            return

        if selected is None:
            self.get_logger().warn(
                'No AnyGrasp candidate near LLM target '
                f'label={self.target_label}. Not publishing.'
            )
            return

        output, dist_xy, dz = selected
        self.grasp_pub.publish(String(data=json.dumps(output)))
        self.get_logger().info(
            'Published LLM-selected grasp: '
            f'label={self.target_label} '
            f'index={output["selected_index"]} '
            f'score={output["score"]:.4f} '
            f'dist_xy={dist_xy:.3f} dz={dz:.3f}'
        )

    def select_candidate(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], float, float] | None:
        if not isinstance(payload, dict):
            raise TypeError('candidate payload is not a JSON object')

        header = payload.get('header', {})
        if not isinstance(header, dict):
            header = {}

        source_frame = str(header.get('frame_id') or self.source_frame_default)
        candidates = payload.get('candidates')
        if candidates is None:
            candidates = payload.get('grasps', [])
        if not isinstance(candidates, list):
            raise TypeError('candidate message candidates field is not a list')

        total = len(candidates)
        transform_cache: dict[str, tuple[list[list[float]], list[float]]] = {}
        near_candidates: list[tuple[dict[str, Any], float, float, str]] = []
        target_x, target_y, target_z = self.latest_yolo_target_base

        for index, candidate in enumerate(candidates):
            try:
                candidate_source_frame = self.candidate_source_frame(candidate, source_frame)
                base_rotation_tf, base_translation_tf = self.lookup_base_transform(
                    candidate_source_frame,
                    transform_cache,
                )
                base_candidate = self.candidate_to_base(
                    candidate,
                    index,
                    base_rotation_tf,
                    base_translation_tf,
                )
                grasp = base_candidate['translation']
                dx = grasp['x'] - target_x
                dy = grasp['y'] - target_y
                dz = grasp['z'] - target_z
                dist_xy = math.sqrt(dx * dx + dy * dy)

                if dist_xy <= self.target_xy_radius and abs(dz) <= self.target_z_radius:
                    near_candidates.append(
                        (base_candidate, dist_xy, dz, candidate_source_frame)
                    )
            except Exception as exc:
                self.get_logger().debug(
                    f'candidate {index} rejected before LLM filtering: {repr(exc)}'
                )

        if not near_candidates:
            return None

        selected, dist_xy, dz, selected_source_frame = max(
            near_candidates,
            key=lambda item: (item[0]['score'], -item[1]),
        )
        output = dict(selected)
        output['header'] = {
            'frame_id': self.target_frame,
            'source_frame_id': selected_source_frame,
            'stamp_sec': int(header.get('stamp_sec', 0)),
            'stamp_nanosec': int(header.get('stamp_nanosec', 0)),
        }
        output['num_input_candidates'] = total
        output['num_safe_candidates'] = len(near_candidates)
        output['num_llm_filtered_candidates'] = len(near_candidates)
        return output, dist_xy, dz

    def lookup_base_transform(
        self,
        source_frame: str,
        transform_cache: dict[str, tuple[list[list[float]], list[float]]],
    ) -> tuple[list[list[float]], list[float]]:
        if source_frame in transform_cache:
            return transform_cache[source_frame]

        if source_frame == self.target_frame:
            result = (
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [0.0, 0.0, 0.0],
            )
            transform_cache[source_frame] = result
            return result

        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=self.tf_timeout_sec),
        )
        rotation = transform.transform.rotation
        translation = transform.transform.translation
        result = (
            _quat_to_matrix(rotation.x, rotation.y, rotation.z, rotation.w),
            [
                float(translation.x),
                float(translation.y),
                float(translation.z),
            ],
        )
        transform_cache[source_frame] = result
        return result

    def candidate_to_base(
        self,
        candidate: dict[str, Any],
        index: int,
        base_rotation_tf: list[list[float]],
        base_translation_tf: list[float],
    ) -> dict[str, Any]:
        if not isinstance(candidate, dict):
            raise TypeError('candidate is not a JSON object')

        score = _finite_float(candidate.get('score'), 'score')
        camera_translation = _read_xyz(candidate, 'translation')
        camera_rotation = _validate_rotation_matrix(candidate.get('rotation_matrix'))

        rotated_translation = _mat_vec_mul(base_rotation_tf, camera_translation)
        base_translation = [
            base_translation_tf[0] + rotated_translation[0],
            base_translation_tf[1] + rotated_translation[1],
            base_translation_tf[2] + rotated_translation[2],
        ]
        base_rotation = _mat_mat_mul(base_rotation_tf, camera_rotation)

        if not all(math.isfinite(value) for value in base_translation):
            raise ValueError('base translation contains non-finite values')
        for row in base_rotation:
            if not all(math.isfinite(value) for value in row):
                raise ValueError('base rotation contains non-finite values')

        hover_translation = [
            base_translation[0],
            base_translation[1],
            base_translation[2] + self.hover_offset_z,
        ]

        return {
            'selected_index': index,
            'score': score,
            'translation': {
                'x': base_translation[0],
                'y': base_translation[1],
                'z': base_translation[2],
            },
            'rotation_matrix': base_rotation,
            'width': _finite_float(candidate.get('width', 0.0), 'width'),
            'height': _finite_float(candidate.get('height', 0.0), 'height'),
            'depth': _finite_float(candidate.get('depth', 0.0), 'depth'),
            'object_id': self.safe_int(candidate.get('object_id', -1), -1),
            'hover_translation': {
                'x': hover_translation[0],
                'y': hover_translation[1],
                'z': hover_translation[2],
            },
        }

    def candidate_source_frame(self, candidate: Any, message_source_frame: str) -> str:
        if isinstance(candidate, dict):
            header = candidate.get('header', {})
            if isinstance(header, dict) and header.get('frame_id'):
                return str(header['frame_id'])
        return message_source_frame or self.source_frame_default

    @staticmethod
    def extract_yolo_target_label(payload: dict[str, Any]) -> str | None:
        for key in ('label', 'target_label', 'class_name', 'cls_name'):
            value = payload.get(key)
            if value:
                return str(value)

        raw_detection = payload.get('raw_detection')
        if isinstance(raw_detection, dict):
            for key in ('label', 'target_label', 'class_name', 'cls_name'):
                value = raw_detection.get(key)
                if value:
                    return str(value)
            raw_text = raw_detection.get('raw')
            if isinstance(raw_text, str) and ',' in raw_text:
                return raw_text.split(',', 1)[0].strip()
        elif isinstance(raw_detection, str) and ',' in raw_detection:
            return raw_detection.split(',', 1)[0].strip()

        return None

    @staticmethod
    def extract_yolo_command_id(payload: dict[str, Any]) -> Any:
        if payload.get('command_id') is not None:
            return payload.get('command_id')

        raw_detection = payload.get('raw_detection')
        if isinstance(raw_detection, dict):
            return raw_detection.get('command_id')

        return None

    @staticmethod
    def safe_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


def main(args=None):
    rclpy.init(args=args)
    node = LLMAnyGraspPoseFilterNode()

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
