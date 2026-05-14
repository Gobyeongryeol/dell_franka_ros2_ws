import json
import math
import os
import sys
import time
from argparse import Namespace
from contextlib import contextmanager


ANYGRASP_CONDA_PYTHON = '/home/jairlab/miniforge3/envs/anygrasp/bin/python'


def _maybe_reexec_with_anygrasp_python():
    """Make ros2 run work even if the generated entry point uses system Python."""
    if os.path.exists(ANYGRASP_CONDA_PYTHON) and os.path.realpath(sys.executable) != os.path.realpath(ANYGRASP_CONDA_PYTHON):
        env = os.environ.copy()
        env.setdefault('PYTHONNOUSERSITE', '1')
        os.execve(ANYGRASP_CONDA_PYTHON, [ANYGRASP_CONDA_PYTHON, '-m', 'anygrasp_ros2_bridge.anygrasp_detection_node', *sys.argv[1:]], env)


@contextmanager
def _pushd(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class AnyGraspDetectionNode:
    def __init__(self):
        import message_filters
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String

        class _Node(Node):
            pass

        self.node = _Node('anygrasp_detection_node')
        self.rclpy = rclpy
        self.Image = Image
        self.CameraInfo = CameraInfo
        self.String = String
        self.message_filters = message_filters
        self.bridge = CvBridge()

        self._declare_parameters()
        self._read_parameters()

        self.best_pub = self.node.create_publisher(String, self.output_topic, 10)
        self.candidates_pub = self.node.create_publisher(String, self.candidates_topic, 10)

        self.camera_info = None
        self.frame_count = 0
        self.last_inference_time = 0.0
        self.has_published_once = False
        self.anygrasp = None

        self._init_anygrasp()
        self._init_ros_io()

    def _declare_parameters(self):
        defaults = {
            'color_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'output_topic': '/anygrasp/best_grasp_pose',
            'candidates_topic': '/anygrasp/grasp_candidates',
            'frame_id': 'camera_color_optical_frame',
            'checkpoint_path': '/home/jairlab/anygrasp_ws/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar',
            'anygrasp_sdk_path': '/home/jairlab/anygrasp_ws/anygrasp_sdk/grasp_detection',
            'max_points': 20000,
            'min_depth': 0.05,
            'max_depth': 1.2,
            'workspace_x_min': -0.5,
            'workspace_x_max': 0.5,
            'workspace_y_min': -0.5,
            'workspace_y_max': 0.5,
            'workspace_z_min': 0.05,
            'workspace_z_max': 1.2,
            'top_down_grasp': True,
            'collision_detection': True,
            'dense_grasp': False,
            'apply_object_mask': True,
            'run_every_n_frames': 15,
            'min_interval_sec': 0.0,
            'once': False,
            'visualize': False,
            'num_candidates': 20,
        }
        for name, value in defaults.items():
            self.node.declare_parameter(name, value)

    def _get_param(self, name):
        return self.node.get_parameter(name).value

    def _read_parameters(self):
        self.color_topic = self._get_param('color_topic')
        self.depth_topic = self._get_param('depth_topic')
        self.camera_info_topic = self._get_param('camera_info_topic')
        self.output_topic = self._get_param('output_topic')
        self.candidates_topic = self._get_param('candidates_topic')
        self.frame_id = self._get_param('frame_id')
        self.checkpoint_path = self._get_param('checkpoint_path')
        self.anygrasp_sdk_path = self._get_param('anygrasp_sdk_path')
        self.max_points = int(self._get_param('max_points'))
        self.min_depth = float(self._get_param('min_depth'))
        self.max_depth = float(self._get_param('max_depth'))
        self.lims = [
            float(self._get_param('workspace_x_min')),
            float(self._get_param('workspace_x_max')),
            float(self._get_param('workspace_y_min')),
            float(self._get_param('workspace_y_max')),
            float(self._get_param('workspace_z_min')),
            float(self._get_param('workspace_z_max')),
        ]
        self.top_down_grasp = bool(self._get_param('top_down_grasp'))
        self.collision_detection = bool(self._get_param('collision_detection'))
        self.dense_grasp = bool(self._get_param('dense_grasp'))
        self.apply_object_mask = bool(self._get_param('apply_object_mask'))
        self.run_every_n_frames = max(1, int(self._get_param('run_every_n_frames')))
        self.min_interval_sec = max(0.0, float(self._get_param('min_interval_sec')))
        self.once = bool(self._get_param('once'))
        self.visualize = bool(self._get_param('visualize'))
        self.num_candidates = max(1, int(self._get_param('num_candidates')))

    def _init_anygrasp(self):
        if not os.path.isdir(self.anygrasp_sdk_path):
            raise RuntimeError(f'anygrasp_sdk_path does not exist: {self.anygrasp_sdk_path}')
        if not os.path.exists(self.checkpoint_path):
            raise RuntimeError(f'checkpoint_path does not exist: {self.checkpoint_path}')

        if self.anygrasp_sdk_path not in sys.path:
            sys.path.insert(0, self.anygrasp_sdk_path)

        cfgs = Namespace(
            checkpoint_path=self.checkpoint_path,
            max_gripper_width=0.1,
            gripper_height=0.03,
            top_down_grasp=self.top_down_grasp,
            debug=False,
        )

        with _pushd(self.anygrasp_sdk_path):
            from gsnet import AnyGrasp
            self.anygrasp = AnyGrasp(cfgs)
            self.anygrasp.load_net()

        self.node.get_logger().info('AnyGrasp initialized once and network loaded.')

    def _init_ros_io(self):
        self.info_sub = self.node.create_subscription(
            self.CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            10,
        )
        self.color_sub = self.message_filters.Subscriber(self.node, self.Image, self.color_topic)
        self.depth_sub = self.message_filters.Subscriber(self.node, self.Image, self.depth_topic)
        self.sync = self.message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=5,
            slop=0.08,
        )
        self.sync.registerCallback(self._rgbd_callback)

        self.node.get_logger().info(f'Subscribed color: {self.color_topic}')
        self.node.get_logger().info(f'Subscribed depth: {self.depth_topic}')
        self.node.get_logger().info(f'Subscribed camera_info: {self.camera_info_topic}')
        self.node.get_logger().info(f'Publishing best grasp JSON: {self.output_topic}')

    def _camera_info_callback(self, msg):
        self.camera_info = msg

    def _rgbd_callback(self, color_msg, depth_msg):
        if self.once and self.has_published_once:
            return

        self.frame_count += 1
        if self.frame_count % self.run_every_n_frames != 0 and not (self.once and self.frame_count == 1):
            return

        now = time.monotonic()
        if self.min_interval_sec > 0.0 and now - self.last_inference_time < self.min_interval_sec:
            return

        if self.camera_info is None:
            self.node.get_logger().warn('No CameraInfo received yet; skipping frame.')
            return

        try:
            points, colors = self._messages_to_anygrasp_inputs(color_msg, depth_msg, self.camera_info)
            if len(points) == 0:
                self.node.get_logger().warn('No valid depth points after filtering; skipping frame.')
                return

            gg, _cloud = self.anygrasp.get_grasp(
                points,
                colors,
                lims=self.lims,
                apply_object_mask=self.apply_object_mask,
                dense_grasp=self.dense_grasp,
                collision_detection=self.collision_detection,
            )

            if len(gg) == 0:
                self.node.get_logger().warn('AnyGrasp returned no grasps.')
                self._publish_empty(color_msg)
                self.has_published_once = True
                return

            gg = gg.nms().sort_by_score()
            best = gg[0]
            self._publish_grasps(color_msg, best, gg)
            self.last_inference_time = now
            self.has_published_once = True
            if self.once:
                self.node.get_logger().info('once=true: published one inference result; staying alive without further inference.')

        except Exception as exc:
            self.node.get_logger().error(f'AnyGrasp inference failed: {repr(exc)}')

    def _messages_to_anygrasp_inputs(self, color_msg, depth_msg, camera_info):
        import cv2
        import numpy as np

        color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='passthrough')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        if color.ndim == 2:
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2RGB)
        elif color.shape[2] == 4:
            if 'bgra' in color_msg.encoding.lower():
                color = cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
            else:
                color = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
        elif 'bgr' in color_msg.encoding.lower():
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        color = color.astype(np.float32) / 255.0

        depth_encoding = depth_msg.encoding.upper()
        if depth_encoding == '16UC1':
            points_z = depth.astype(np.float32) / 1000.0
        elif depth_encoding == '32FC1':
            points_z = depth.astype(np.float32)
        else:
            self.node.get_logger().warn(f'Unexpected depth encoding {depth_msg.encoding}; treating as millimeters if integer, meters otherwise.')
            if np.issubdtype(depth.dtype, np.integer):
                points_z = depth.astype(np.float32) / 1000.0
            else:
                points_z = depth.astype(np.float32)

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            raise RuntimeError(f'Invalid CameraInfo intrinsics fx={fx}, fy={fy}')

        height, width = points_z.shape[:2]
        if color.shape[0] != height or color.shape[1] != width:
            color = cv2.resize(color, (width, height), interpolation=cv2.INTER_LINEAR)

        xmap, ymap = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        points = np.stack([points_x, points_y, points_z], axis=-1)

        mask = (
            np.isfinite(points_z)
            & (points_z > self.min_depth)
            & (points_z < self.max_depth)
            & (points_x >= self.lims[0])
            & (points_x <= self.lims[1])
            & (points_y >= self.lims[2])
            & (points_y <= self.lims[3])
            & (points_z >= self.lims[4])
            & (points_z <= self.lims[5])
        )

        points = points[mask].astype(np.float32)
        colors = color[mask].astype(np.float32)

        if self.max_points > 0 and len(points) > self.max_points:
            idx = np.random.choice(len(points), size=self.max_points, replace=False)
            points = points[idx]
            colors = colors[idx]

        return points, colors

    def _stamp_dict(self, msg):
        stamp = msg.header.stamp
        return {'stamp_sec': int(stamp.sec), 'stamp_nanosec': int(stamp.nanosec)}

    def _grasp_to_dict(self, grasp):
        return {
            'score': float(grasp.score),
            'translation': {
                'x': float(grasp.translation[0]),
                'y': float(grasp.translation[1]),
                'z': float(grasp.translation[2]),
            },
            'rotation_matrix': grasp.rotation_matrix.astype(float).tolist(),
            'width': float(grasp.width),
            'height': float(grasp.height),
            'depth': float(grasp.depth),
            'object_id': int(grasp.object_id),
        }

    def _publish_grasps(self, source_msg, best, gg):
        msg = self.String()
        payload = {
            'header': {
                'frame_id': self.frame_id,
                **self._stamp_dict(source_msg),
            },
            **self._grasp_to_dict(best),
        }
        msg.data = json.dumps(payload)
        self.best_pub.publish(msg)

        candidates_msg = self.String()
        candidates = []
        for i in range(min(self.num_candidates, len(gg))):
            candidates.append(self._grasp_to_dict(gg[i]))
        candidates_msg.data = json.dumps({
            'header': {
                'frame_id': self.frame_id,
                **self._stamp_dict(source_msg),
            },
            'count': len(candidates),
            'candidates': candidates,
            'grasps': candidates,
        })
        self.candidates_pub.publish(candidates_msg)

        self.node.get_logger().info(
            f'Published best grasp in {self.frame_id}: score={payload["score"]:.4f}, '
            f't=({payload["translation"]["x"]:.3f}, {payload["translation"]["y"]:.3f}, {payload["translation"]["z"]:.3f})'
        )

    def _publish_empty(self, source_msg):
        msg = self.String()
        msg.data = json.dumps({
            'header': {
                'frame_id': self.frame_id,
                **self._stamp_dict(source_msg),
            },
            'error': 'no_grasp_detected',
        })
        self.best_pub.publish(msg)

    def spin(self):
        self.rclpy.spin(self.node)

    def shutdown(self):
        self.node.destroy_node()


def main(args=None):
    _maybe_reexec_with_anygrasp_python()

    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = None
    try:
        node = AnyGraspDetectionNode()
        node.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        if node is not None:
            node.node.get_logger().error(f'Fatal startup error: {repr(exc)}')
        else:
            print(f'Fatal startup error: {repr(exc)}', file=sys.stderr)
        raise
    finally:
        if node is not None:
            node.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
