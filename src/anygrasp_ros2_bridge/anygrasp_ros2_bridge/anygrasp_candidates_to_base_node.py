import json
import math
from collections import Counter
from statistics import median

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
import tf2_ros


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
    return [
        _finite_float(value.get('x'), f'{key}.x'),
        _finite_float(value.get('y'), f'{key}.y'),
        _finite_float(value.get('z'), f'{key}.z'),
    ]


def _is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


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


class AnyGraspCandidatesToBaseNode(Node):
    def __init__(self):
        super().__init__('anygrasp_candidates_to_base_node')

        self.declare_parameter('input_topic', '/anygrasp/grasp_candidates')
        self.declare_parameter('output_topic', '/anygrasp/best_safe_grasp_pose_base')
        self.declare_parameter('target_frame', 'base')
        self.declare_parameter('min_score', 0.2)
        self.declare_parameter('x_min', 0.20)
        self.declare_parameter('x_max', 0.70)
        self.declare_parameter('y_min', -0.50)
        self.declare_parameter('y_max', 0.50)
        self.declare_parameter('z_min', 0.20)
        self.declare_parameter('z_max', 0.80)
        self.declare_parameter('hover_offset_z', 0.10)
        self.declare_parameter('hover_z_min', 0.20)
        self.declare_parameter('hover_z_max', 0.90)
        self.declare_parameter('max_candidates', 20)
        self.declare_parameter('use_candidate_clustering', True)
        self.declare_parameter('cluster_radius_xy', 0.025)
        self.declare_parameter('min_cluster_size', 2)
        self.declare_parameter('selection_method', 'cluster_median')
        self.declare_parameter('width_min', 0.015)
        self.declare_parameter('width_max', 0.080)
        self.declare_parameter('use_width_filter', True)
        self.declare_parameter('prefer_centered_candidate', True)
        self.declare_parameter('center_weight', 0.2)
        self.declare_parameter('score_weight', 1.0)
        self.declare_parameter('z_weight', 0.2)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.min_score = float(self.get_parameter('min_score').value)
        self.x_min = float(self.get_parameter('x_min').value)
        self.x_max = float(self.get_parameter('x_max').value)
        self.y_min = float(self.get_parameter('y_min').value)
        self.y_max = float(self.get_parameter('y_max').value)
        self.z_min = float(self.get_parameter('z_min').value)
        self.z_max = float(self.get_parameter('z_max').value)
        self.hover_offset_z = float(self.get_parameter('hover_offset_z').value)
        self.hover_z_min = float(self.get_parameter('hover_z_min').value)
        self.hover_z_max = float(self.get_parameter('hover_z_max').value)
        self.max_candidates = max(1, int(self.get_parameter('max_candidates').value))
        self.use_candidate_clustering = bool(self.get_parameter('use_candidate_clustering').value)
        self.cluster_radius_xy = float(self.get_parameter('cluster_radius_xy').value)
        self.min_cluster_size = max(1, int(self.get_parameter('min_cluster_size').value))
        self.selection_method = str(self.get_parameter('selection_method').value)
        self.width_min = float(self.get_parameter('width_min').value)
        self.width_max = float(self.get_parameter('width_max').value)
        self.use_width_filter = bool(self.get_parameter('use_width_filter').value)
        self.prefer_centered_candidate = bool(self.get_parameter('prefer_centered_candidate').value)
        self.center_weight = float(self.get_parameter('center_weight').value)
        self.score_weight = float(self.get_parameter('score_weight').value)
        self.z_weight = float(self.get_parameter('z_weight').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(String, self.output_topic, 10)
        self.subscription = self.create_subscription(String, self.input_topic, self._callback, 10)

        self.get_logger().info(f'Subscribed grasp candidates JSON: {self.input_topic}')
        self.get_logger().info(f'Publishing best safe base grasp JSON: {self.output_topic}')
        self.get_logger().info(f'Target frame: {self.target_frame}')
        self.get_logger().info(
            'candidate selection: '
            f'use_candidate_clustering={self.use_candidate_clustering}, '
            f'selection_method={self.selection_method}, '
            f'cluster_radius_xy={self.cluster_radius_xy:.3f}, '
            f'min_cluster_size={self.min_cluster_size}'
        )
        self.get_logger().info(
            f'width filter: enabled={self.use_width_filter}, '
            f'width_min={self.width_min:.3f}, width_max={self.width_max:.3f}'
        )

    def _callback(self, msg):
        try:
            payload = json.loads(msg.data)
            output = self._select_best_safe(payload)
            if output is None:
                return

            out_msg = String()
            out_msg.data = json.dumps(output)
            self.publisher.publish(out_msg)

            translation = output['translation']
            hover = output['hover_translation']
            self.get_logger().info(f'selected candidate index={output["selected_index"]}')
            self.get_logger().info(f'selected score={output["score"]:.4f}')
            self.get_logger().info(
                f'selected translation=({translation["x"]:.3f}, {translation["y"]:.3f}, {translation["z"]:.3f})'
            )
            self.get_logger().info(
                f'selected hover_translation=({hover["x"]:.3f}, {hover["y"]:.3f}, {hover["z"]:.3f})'
            )
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Invalid JSON on {self.input_topic}: {exc}')
        except Exception as exc:
            self.get_logger().warn(f'Failed to select safe grasp candidate: {repr(exc)}')

    def _select_best_safe(self, payload):
        header = payload.get('header', {})
        source_frame = header.get('frame_id')
        if not source_frame:
            self.get_logger().warn('Candidate message has no header.frame_id; skipping.')
            return None

        candidates = payload.get('candidates')
        if candidates is None:
            candidates = payload.get('grasps', [])
        if not isinstance(candidates, list):
            self.get_logger().warn('Candidate message candidates field is not a list; skipping.')
            return None

        total = len(candidates)
        candidates = candidates[:self.max_candidates]
        self.get_logger().info(f'total candidates={total}, evaluating={len(candidates)}')

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

        safe_candidates = []
        rejected = Counter()

        for index, candidate in enumerate(candidates):
            try:
                base_candidate = self._candidate_to_base(
                    candidate,
                    index,
                    base_rotation_tf,
                    base_translation_tf,
                )
                rejected_reason = self._rejected_reason(base_candidate)
                if rejected_reason:
                    rejected[rejected_reason] += 1
                    continue
                safe_candidates.append(base_candidate)
            except Exception as exc:
                rejected[type(exc).__name__] += 1
                self.get_logger().debug(f'candidate {index} rejected: {repr(exc)}')

        rejected_count = sum(rejected.values())
        self.get_logger().info(f'rejected count={rejected_count}, reasons={dict(rejected)}')
        self.get_logger().info(f'safe candidates count={len(safe_candidates)}')

        if not safe_candidates:
            self.get_logger().warn('No safe grasp candidates found; not publishing.')
            return None

        selected = self._select_candidate(safe_candidates)
        selected['header'] = {
            'frame_id': self.target_frame,
            'source_frame_id': source_frame,
            'stamp_sec': int(header.get('stamp_sec', 0)),
            'stamp_nanosec': int(header.get('stamp_nanosec', 0)),
        }
        selected['num_input_candidates'] = total
        selected['num_safe_candidates'] = len(safe_candidates)
        return selected

    def _select_candidate(self, safe_candidates):
        if not self.use_candidate_clustering or self.selection_method == 'best_score':
            selected = max(safe_candidates, key=lambda item: item['score'])
            self.get_logger().info('selected method=best_score')
            self.get_logger().info(
                f'selected score={selected["score"]:.4f}, width={selected["width"]:.4f}'
            )
            return dict(selected)

        clusters = self._cluster_candidates(safe_candidates)
        self.get_logger().info(f'cluster count={len(clusters)}')
        if not clusters:
            self.get_logger().warn('No clusters formed; falling back to best_score.')
            return dict(max(safe_candidates, key=lambda item: item['score']))

        clusters.sort(
            key=lambda cluster: (
                len(cluster),
                max(item['score'] for item in cluster),
            ),
            reverse=True,
        )
        largest = clusters[0]
        self.get_logger().info(f'largest cluster size={len(largest)}')
        if len(largest) < self.min_cluster_size:
            self.get_logger().warn(
                f'largest cluster size {len(largest)} < min_cluster_size {self.min_cluster_size}; '
                'falling back to best_score.'
            )
            selected = max(safe_candidates, key=lambda item: item['score'])
            return dict(selected)

        median_translation = self._median_xyz(largest, 'translation')
        median_hover = self._median_xyz(largest, 'hover_translation')
        median_z = median_translation['z']
        best_rotation_source = max(largest, key=lambda item: item['score'])

        if self.selection_method == 'cluster_best_score':
            if self.prefer_centered_candidate:
                selected = max(
                    largest,
                    key=lambda item: self._cluster_selection_score(item, median_translation, median_z),
                )
            else:
                selected = best_rotation_source
            selected = dict(selected)
            self.get_logger().info('selected method=cluster_best_score')
        elif self.selection_method == 'cluster_median':
            selected = dict(best_rotation_source)
            selected['translation'] = median_translation
            selected['hover_translation'] = median_hover
            selected['width'] = median(item['width'] for item in largest)
            self.get_logger().info('selected method=cluster_median')
        else:
            self.get_logger().warn(
                f'Unsupported selection_method={self.selection_method}; using cluster_median.'
            )
            selected = dict(best_rotation_source)
            selected['translation'] = median_translation
            selected['hover_translation'] = median_hover
            selected['width'] = median(item['width'] for item in largest)

        selected['rotation_matrix'] = best_rotation_source['rotation_matrix']
        selected['score'] = best_rotation_source['score']
        selected['selected_index'] = best_rotation_source['selected_index']

        translation = selected['translation']
        self.get_logger().info(
            'selected median translation='
            f'({median_translation["x"]:.3f}, {median_translation["y"]:.3f}, {median_translation["z"]:.3f})'
        )
        self.get_logger().info(
            f'selected translation=({translation["x"]:.3f}, {translation["y"]:.3f}, {translation["z"]:.3f})'
        )
        self.get_logger().info(f'selected score={selected["score"]:.4f}')
        self.get_logger().info(f'selected width={selected["width"]:.4f}')
        self.get_logger().info(
            f'selected rotation source index={best_rotation_source["selected_index"]}, '
            f'score={best_rotation_source["score"]:.4f}'
        )
        return selected

    def _cluster_candidates(self, candidates):
        clusters = []
        for candidate in candidates:
            placed = False
            cx = candidate['translation']['x']
            cy = candidate['translation']['y']
            for cluster in clusters:
                if any(self._xy_distance(candidate, other) <= self.cluster_radius_xy for other in cluster):
                    cluster.append(candidate)
                    placed = True
                    break
            if not placed:
                clusters.append([candidate])
        return clusters

    @staticmethod
    def _xy_distance(a, b):
        dx = a['translation']['x'] - b['translation']['x']
        dy = a['translation']['y'] - b['translation']['y']
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _median_xyz(items, key):
        return {
            'x': median(item[key]['x'] for item in items),
            'y': median(item[key]['y'] for item in items),
            'z': median(item[key]['z'] for item in items),
        }

    def _cluster_selection_score(self, candidate, median_translation, median_z):
        dx = candidate['translation']['x'] - median_translation['x']
        dy = candidate['translation']['y'] - median_translation['y']
        dist_xy = math.sqrt(dx * dx + dy * dy)
        dz = abs(candidate['translation']['z'] - median_z)
        final_score = (
            self.score_weight * candidate['score']
            - self.center_weight * dist_xy
            - self.z_weight * dz
        )
        self.get_logger().debug(
            f'candidate {candidate["selected_index"]} cluster final_score={final_score:.4f} '
            f'score={candidate["score"]:.4f} dist_xy={dist_xy:.4f} dz={dz:.4f}'
        )
        return final_score

    def _candidate_to_base(self, candidate, index, base_rotation_tf, base_translation_tf):
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
            'object_id': int(candidate.get('object_id', -1)),
            'hover_translation': {
                'x': hover_translation[0],
                'y': hover_translation[1],
                'z': hover_translation[2],
            },
        }

    def _rejected_reason(self, candidate):
        score = candidate['score']
        grasp = candidate['translation']
        hover = candidate['hover_translation']

        if score < self.min_score:
            return 'score_below_min'
        width = candidate['width']
        if self.use_width_filter and not (self.width_min <= width <= self.width_max):
            return 'width_outside'
        if not (self.x_min <= grasp['x'] <= self.x_max):
            return 'grasp_x_outside'
        if not (self.y_min <= grasp['y'] <= self.y_max):
            return 'grasp_y_outside'
        if not (self.z_min <= grasp['z'] <= self.z_max):
            return 'grasp_z_outside'
        if not (self.x_min <= hover['x'] <= self.x_max):
            return 'hover_x_outside'
        if not (self.y_min <= hover['y'] <= self.y_max):
            return 'hover_y_outside'
        if not (self.hover_z_min <= hover['z'] <= self.hover_z_max):
            return 'hover_z_outside'
        return ''


def main(args=None):
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = AnyGraspCandidatesToBaseNode()
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
