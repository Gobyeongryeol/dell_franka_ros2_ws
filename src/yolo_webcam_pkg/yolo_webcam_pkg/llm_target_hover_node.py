#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.srv import GetCartesianPath
from std_msgs.msg import String

import tf2_ros


class LLMTargetHoverNode(Node):
    def __init__(self):
        super().__init__('llm_target_hover_node')

        self.logger = self.get_logger()

        self.declare_parameter('target_topic', '/yolo/target_base')
        self.declare_parameter('llm_target_topic', '/llm/target_pick')
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('group_name', 'fr3_arm')
        self.declare_parameter('eef_link', 'fr3_link8')
        self.declare_parameter('cartesian_service', '/compute_cartesian_path')
        self.declare_parameter('execute_action', '/execute_trajectory')
        self.declare_parameter('min_cartesian_fraction', 0.90)
        self.declare_parameter('max_step', 0.005)
        self.declare_parameter('avoid_collisions', False)

        self.declare_parameter('hover_z', 0.25)
        self.declare_parameter('x_offset', 0.0)
        self.declare_parameter('y_offset', 0.0)
        self.declare_parameter('once', True)
        self.declare_parameter('dry_run', True)
        self.declare_parameter('min_x', 0.20)
        self.declare_parameter('max_x', 0.70)
        self.declare_parameter('min_y', -0.50)
        self.declare_parameter('max_y', 0.50)
        self.declare_parameter('min_z', 0.10)
        self.declare_parameter('max_z', 0.60)

        self.target_topic = str(self.get_parameter('target_topic').value)
        self.llm_target_topic = str(self.get_parameter('llm_target_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.group_name = str(self.get_parameter('group_name').value)
        self.eef_link = str(self.get_parameter('eef_link').value)
        self.cartesian_service = str(self.get_parameter('cartesian_service').value)
        self.execute_action = str(self.get_parameter('execute_action').value)
        self.min_cartesian_fraction = float(
            self.get_parameter('min_cartesian_fraction').value
        )
        self.max_step = float(self.get_parameter('max_step').value)
        self.avoid_collisions = bool(self.get_parameter('avoid_collisions').value)

        self.hover_z = float(self.get_parameter('hover_z').value)
        self.x_offset = float(self.get_parameter('x_offset').value)
        self.y_offset = float(self.get_parameter('y_offset').value)
        self.once = bool(self.get_parameter('once').value)
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.min_x = float(self.get_parameter('min_x').value)
        self.max_x = float(self.get_parameter('max_x').value)
        self.min_y = float(self.get_parameter('min_y').value)
        self.max_y = float(self.get_parameter('max_y').value)
        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)

        self.move_armed = False
        self.move_in_progress = False
        self.target_label = ''
        self.command_id = 0
        self.state_lock = threading.Lock()

        self.tf_buffer: Optional[tf2_ros.Buffer] = None
        self.tf_listener: Optional[tf2_ros.TransformListener] = None
        self.cartesian_client = None
        self.execute_client = None

        if not self.dry_run:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

            self.cartesian_client = self.create_client(
                GetCartesianPath, self.cartesian_service
            )
            while rclpy.ok() and not self.cartesian_client.wait_for_service(
                timeout_sec=2.0
            ):
                self.logger.info(
                    f'Waiting for {self.cartesian_service} service...'
                )
            self.logger.info('Connected to Cartesian path service')

            self.execute_client = ActionClient(
                self, ExecuteTrajectory, self.execute_action
            )
            while rclpy.ok() and not self.execute_client.wait_for_server(
                timeout_sec=2.0
            ):
                self.logger.info(
                    f'Waiting for {self.execute_action} action server...'
                )
            self.logger.info('Connected to execute trajectory action server')

        self.create_subscription(String, self.target_topic, self.target_callback, 10)
        self.create_subscription(
            String, self.llm_target_topic, self.llm_target_callback, 10
        )

        self.logger.info('LLM target hover node ready')
        self.logger.info(f'Listening target: {self.target_topic}')
        self.logger.info(f'Listening LLM target: {self.llm_target_topic}')
        self.logger.info(
            f'hover_z={self.hover_z:.3f}, x_offset={self.x_offset:.3f}, '
            f'y_offset={self.y_offset:.3f}, once={self.once}, '
            f'dry_run={self.dry_run}'
        )
        self.logger.info(
            f'Safety limits: x=[{self.min_x:.3f}, {self.max_x:.3f}], '
            f'y=[{self.min_y:.3f}, {self.max_y:.3f}], '
            f'z=[{self.min_z:.3f}, {self.max_z:.3f}]'
        )

    def llm_target_callback(self, msg: String):
        try:
            target = json.loads(msg.data)
        except Exception as exc:
            self.logger.warn(f'Failed to parse LLM target JSON: {exc}')
            return

        label = target.get('label')
        if not label:
            labels = target.get('labels', [])
            if isinstance(labels, list) and labels:
                label = labels[0]

        if not label:
            self.logger.warn(f'LLM target message has no label: {msg.data}')
            return

        with self.state_lock:
            self.target_label = str(label)
            self.move_armed = True
            self.command_id += 1

        self.logger.info(
            f'Received LLM pick command: label={self.target_label}, armed hover move'
        )

    def target_callback(self, msg: String):
        with self.state_lock:
            if not self.move_armed:
                return
            if self.move_in_progress:
                self.logger.warn('Move already in progress; ignoring target_base.')
                return
            command_id = self.command_id

        try:
            target = json.loads(msg.data)
        except Exception as exc:
            self.logger.warn(f'Failed to parse target_base JSON: {exc}')
            return

        try:
            raw_x = float(target['x_base'])
            raw_y = float(target['y_base'])
        except Exception as exc:
            self.logger.warn(f'Invalid target_base x_base/y_base: {exc}')
            return

        label_text = f' label={self.target_label}' if self.target_label else ''

        z_base = self.read_optional_float(target.get('z_base'))
        z_base_text = 'nan' if z_base is None else f'{z_base:.3f}'

        x = raw_x + self.x_offset
        y = raw_y + self.y_offset
        z = self.hover_z

        self.logger.info(
            f'Received target_base: x={raw_x:.3f}, y={raw_y:.3f}, '
            f'z_base={z_base_text}{label_text}'
        )
        self.logger.info(f'Hover goal: x={x:.3f}, y={y:.3f}, z={z:.3f}')

        if not self.check_workspace(x, y, z):
            return

        if self.dry_run:
            self.logger.info('dry_run=true, not moving robot')
            self.finish_hover_attempt(command_id)
            return

        with self.state_lock:
            self.move_in_progress = True
        thread = threading.Thread(
            target=self.move_to_hover_worker,
            args=(x, y, z, command_id),
            daemon=True,
        )
        thread.start()

    @staticmethod
    def read_optional_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def check_workspace(self, x: float, y: float, z: float) -> bool:
        ok = (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )

        if not ok:
            self.logger.warn(
                'Hover goal outside safety workspace; not moving. '
                f'goal x={x:.3f}, y={y:.3f}, z={z:.3f}, '
                f'allowed x=[{self.min_x:.3f}, {self.max_x:.3f}], '
                f'y=[{self.min_y:.3f}, {self.max_y:.3f}], '
                f'z=[{self.min_z:.3f}, {self.max_z:.3f}]'
            )

        return ok

    def get_current_orientation(self):
        if self.tf_buffer is None:
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.eef_link,
                Time(),
                timeout=Duration(seconds=2.0),
            )
            return transform.transform.rotation
        except Exception as exc:
            self.logger.error(
                f'Failed to get current orientation from TF '
                f'{self.base_frame}->{self.eef_link}: {exc}'
            )
            return None

    @staticmethod
    def wait_for_future(future):
        while rclpy.ok() and not future.done():
            time.sleep(0.02)
        if not future.done():
            return None
        return future.result()

    def make_hover_pose(self, x: float, y: float, z: float, orientation) -> Pose:
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation = orientation
        return pose

    def compute_cartesian(self, pose: Pose):
        if self.cartesian_client is None:
            return None, 0.0

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.group_name = self.group_name
        req.link_name = self.eef_link
        req.waypoints = [pose]
        req.max_step = self.max_step
        req.jump_threshold = 0.0
        req.avoid_collisions = self.avoid_collisions

        if hasattr(req, 'prismatic_jump_threshold'):
            req.prismatic_jump_threshold = 0.0
        if hasattr(req, 'revolute_jump_threshold'):
            req.revolute_jump_threshold = 0.0

        future = self.cartesian_client.call_async(req)

        result = self.wait_for_future(future)
        if result is None:
            self.logger.error('Cartesian service returned None')
            return None, 0.0

        return result.solution, result.fraction

    def execute_trajectory(self, trajectory) -> bool:
        if self.execute_client is None:
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory

        future = self.execute_client.send_goal_async(goal)

        goal_handle = self.wait_for_future(future)
        if goal_handle is None or not goal_handle.accepted:
            self.logger.error('ExecuteTrajectory rejected')
            return False

        result_future = goal_handle.get_result_async()

        result = self.wait_for_future(result_future)
        if result is None:
            self.logger.error('ExecuteTrajectory result None')
            return False

        code = result.result.error_code.val
        if code == 1:
            return True

        self.logger.error(f'ExecuteTrajectory failed. error_code={code}')
        return False

    def move_to_hover(self, x: float, y: float, z: float) -> bool:
        self.logger.info('Moving to hover pose...')

        orientation = self.get_current_orientation()
        if orientation is None:
            return False

        pose = self.make_hover_pose(x, y, z, orientation)

        trajectory, fraction = self.compute_cartesian(pose)
        self.logger.info(f'Cartesian fraction={fraction * 100.0:.1f}%')

        if trajectory is None or fraction < self.min_cartesian_fraction:
            self.logger.error('Cartesian path failed; not executing hover move.')
            return False

        ok = self.execute_trajectory(trajectory)
        if ok:
            self.logger.info('Hover move complete.')
        else:
            self.logger.error('Hover move failed.')

        return ok

    def finish_hover_attempt(self, command_id: int):
        with self.state_lock:
            if self.once and self.command_id == command_id:
                self.move_armed = False

    def move_to_hover_worker(self, x: float, y: float, z: float, command_id: int):
        try:
            self.move_to_hover(x, y, z)
        except Exception as exc:
            self.logger.error(f'Unhandled hover move exception: {exc}')
        finally:
            with self.state_lock:
                self.move_in_progress = False
                if self.once and self.command_id == command_id:
                    self.move_armed = False


def main(args=None):
    rclpy.init(args=args)
    node = LLMTargetHoverNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
