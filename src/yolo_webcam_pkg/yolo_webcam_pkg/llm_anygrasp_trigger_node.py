#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import threading
import time
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

try:
    from action_msgs.srv import CancelGoal
    from moveit_msgs.action import ExecuteTrajectory
except Exception:
    CancelGoal = None
    ExecuteTrajectory = None


COMMAND_PARAMETER_DEFAULTS = {
    'min_score': -0.1,
    'x_min': 0.15,
    'x_max': 0.85,
    'y_min': -0.60,
    'y_max': 0.60,
    'z_min': -0.10,
    'z_max': 0.85,
    'use_target_smoothing': False,
    'smoothing_window': 1,
    'min_smoothing_samples': 1,
    'smoothing_method': 'median',
    'require_stable_target': False,
    'max_target_std_xy': 0.30,
    'max_target_std_z': 0.30,
    'stable_target_timeout_sec': 3.0,
    'use_anygrasp_orientation': True,
    'anygrasp_orientation_mode': 'planar_yaw',
    'orientation_test_mode': False,
    'do_descent': True,
    'do_preopen_gripper': True,
    'do_gripper': True,
    'do_place': True,
    'use_camera_view_offset': False,
    'camera_view_x_offset': 0.000,
    'camera_view_y_offset': 0.000,
    'target_frame': 'base',
    'gripper_move_action': '/franka_gripper/move',
    'gripper_grasp_action': '/franka_gripper/grasp',
    'gripper_open_width': 0.080,
    'gripper_close_width': 0.045,
    'gripper_speed': 0.020,
    'gripper_force': 20.0,
    'gripper_epsilon_inner': 0.035,
    'gripper_epsilon_outer': 0.035,
    'topdown_orientation_source': 'current_at_start',
    'planar_yaw_axis_index': 0,
    'planar_yaw_offset': 0.0,
    'safe_z': 0.55,
    'lift_z': 0.40,
    'grasp_z': 0.03,
    'place_x': 0.45,
    'place_y': -0.25,
    'place_z': 0.12,
    'place_safe_z': 0.55,
    'velocity_scaling': 0.15,
    'acceleration_scaling': 0.15,
}

PICK_COMPLETION_LOG_PATTERNS = (
    'AnyGrasp pick/place sequence complete',
    'once=true: pick/place node is done',
    'pick/place node is done',
)


class LLMAnyGraspTriggerNode(Node):
    def __init__(self):
        super().__init__('llm_anygrasp_trigger_node')

        self.declare_parameter('llm_target_topic', '/llm/target_pick')
        self.declare_parameter('llm_stop_topic', '/llm/stop')
        self.declare_parameter('yolo_target_topic', '/yolo/target_base')
        self.declare_parameter('input_topic', '/anygrasp/best_safe_grasp_pose_base')
        self.declare_parameter('dry_run', True)
        self.declare_parameter('plan_only', True)
        self.declare_parameter('execute', False)
        self.declare_parameter('once', True)
        self.declare_parameter('pick_process_timeout_sec', 30.0)
        self.declare_parameter('place_z_default', 0.12)
        self.declare_parameter('place_x_min', 0.15)
        self.declare_parameter('place_x_max', 0.85)
        self.declare_parameter('place_y_min', -0.60)
        self.declare_parameter('place_y_max', 0.60)
        self.declare_parameter('place_z_min', 0.05)
        self.declare_parameter('place_z_max', 0.40)
        self.declare_parameter('zone_a_place_x', 0.45)
        self.declare_parameter('zone_a_place_y', 0.25)
        self.declare_parameter('zone_a_place_z', 0.12)
        self.declare_parameter('zone_b_place_x', 0.45)
        self.declare_parameter('zone_b_place_y', -0.25)
        self.declare_parameter('zone_b_place_z', 0.12)
        self.declare_parameter('return_home_on_stop', True)
        self.declare_parameter('return_home_after_pick', False)
        self.declare_parameter('home_safe_z', 0.55)
        self.declare_parameter('home_x', 0.307)
        self.declare_parameter('home_y', 0.000)
        self.declare_parameter('home_z', 0.476)
        self.declare_parameter('home_use_current_orientation', True)
        self.declare_parameter('home_velocity_scaling', 0.15)
        self.declare_parameter('home_acceleration_scaling', 0.15)

        for name, default in COMMAND_PARAMETER_DEFAULTS.items():
            self.declare_parameter(name, default)

        self.llm_target_topic = str(self.get_parameter('llm_target_topic').value)
        self.llm_stop_topic = str(self.get_parameter('llm_stop_topic').value)
        self.yolo_target_topic = str(self.get_parameter('yolo_target_topic').value)
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.plan_only = bool(self.get_parameter('plan_only').value)
        self.execute = bool(self.get_parameter('execute').value)
        self.once = bool(self.get_parameter('once').value)
        self.pick_process_timeout_sec = float(self.get_parameter('pick_process_timeout_sec').value)
        self.place_z_default = float(self.get_parameter('place_z_default').value)
        self.place_x_min = float(self.get_parameter('place_x_min').value)
        self.place_x_max = float(self.get_parameter('place_x_max').value)
        self.place_y_min = float(self.get_parameter('place_y_min').value)
        self.place_y_max = float(self.get_parameter('place_y_max').value)
        self.place_z_min = float(self.get_parameter('place_z_min').value)
        self.place_z_max = float(self.get_parameter('place_z_max').value)
        self.zone_a_place_x = float(self.get_parameter('zone_a_place_x').value)
        self.zone_a_place_y = float(self.get_parameter('zone_a_place_y').value)
        self.zone_a_place_z = float(self.get_parameter('zone_a_place_z').value)
        self.zone_b_place_x = float(self.get_parameter('zone_b_place_x').value)
        self.zone_b_place_y = float(self.get_parameter('zone_b_place_y').value)
        self.zone_b_place_z = float(self.get_parameter('zone_b_place_z').value)
        self.return_home_on_stop = bool(self.get_parameter('return_home_on_stop').value)
        self.return_home_after_pick = bool(self.get_parameter('return_home_after_pick').value)
        self.home_safe_z = float(self.get_parameter('home_safe_z').value)
        self.home_x = float(self.get_parameter('home_x').value)
        self.home_y = float(self.get_parameter('home_y').value)
        self.home_z = float(self.get_parameter('home_z').value)
        self.home_use_current_orientation = bool(
            self.get_parameter('home_use_current_orientation').value
        )
        self.home_velocity_scaling = float(self.get_parameter('home_velocity_scaling').value)
        self.home_acceleration_scaling = float(
            self.get_parameter('home_acceleration_scaling').value
        )

        self.pick_process = None
        self.pick_running = False
        self.pick_process_started_at = None
        self.pick_completed_by_log = False
        self.pick_completion_time = None
        self.pick_stdout_thread = None
        self.home_process = None
        self.home_running = False
        self.home_thread = None
        self.process_lock = threading.Lock()
        self.latest_command_id = None
        self.latest_target_label = None
        self.latest_yolo_target_base = None

        self.subscription = self.create_subscription(
            String,
            self.llm_target_topic,
            self.target_callback,
            10,
        )
        self.stop_subscription = self.create_subscription(
            String,
            self.llm_stop_topic,
            self.stop_callback,
            10,
        )
        self.yolo_target_subscription = self.create_subscription(
            String,
            self.yolo_target_topic,
            self.yolo_target_callback,
            10,
        )
        self.pick_process_timer = self.create_timer(
            0.5,
            self.check_pick_process,
        )

        self.get_logger().info('LLM AnyGrasp trigger node ready')
        self.get_logger().info(f'Listening LLM target: {self.llm_target_topic}')
        self.get_logger().info(f'Listening LLM stop: {self.llm_stop_topic}')
        self.get_logger().info(f'Listening YOLO target base: {self.yolo_target_topic}')
        self.get_logger().info(
            'Trigger mode: '
            f'dry_run={self.dry_run}, plan_only={self.plan_only}, '
            f'execute={self.execute}, once={self.once}'
        )
        self.get_logger().info(
            f'Pick node input_topic={self.input_topic}. '
            'AnyGrasp detection/candidates_to_base must already be running.'
        )
        self.get_logger().info(
            'Safe home recovery: '
            f'return_home_on_stop={self.return_home_on_stop}, '
            f'return_home_after_pick={self.return_home_after_pick}, '
            f'home=({self.home_x:.3f}, {self.home_y:.3f}, {self.home_z:.3f}), '
            f'safe_z={self.home_safe_z:.3f}'
        )
        self.execute_trajectory_action_client = None
        self.execute_trajectory_cancel_client = None
        if ExecuteTrajectory is not None:
            self.execute_trajectory_action_client = ActionClient(
                self,
                ExecuteTrajectory,
                '/execute_trajectory',
            )
        if CancelGoal is not None:
            self.execute_trajectory_cancel_client = self.create_client(
                CancelGoal,
                '/execute_trajectory/_action/cancel_goal',
            )

    def target_callback(self, msg: String):
        with self.process_lock:
            if not self.prepare_for_new_command_locked():
                return

        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Failed to parse /llm/target_pick JSON: {exc}')
            return

        label = payload.get('label')
        if not label:
            labels = payload.get('labels', [])
            if isinstance(labels, list) and labels:
                label = labels[0]

        if not label:
            self.get_logger().warn(f'/llm/target_pick message has no label: {msg.data}')
            return

        self.latest_command_id = payload.get('command_id')
        self.latest_target_label = str(label)
        self.latest_yolo_target_base = None
        self.get_logger().info(
            f'Received LLM pick command_id={self.latest_command_id}, '
            f'label={label}. Cleared stale YOLO target for trigger.'
        )
        self.start_pick_process(str(label), payload)

    def yolo_target_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
            x_base = float(payload['x_base'])
            y_base = float(payload['y_base'])
            z_base = float(payload['z_base'])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(
                f'Failed to parse YOLO target base on {self.yolo_target_topic}: {exc}'
            )
            return

        current_target_label = self.get_current_target_label_hint()
        yolo_label = self.extract_yolo_target_label(payload)
        if yolo_label and current_target_label and yolo_label != current_target_label:
            self.get_logger().warn(
                f'Ignoring YOLO target label={yolo_label}; current LLM target label='
                f'{current_target_label}.'
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

    def start_pick_process(self, label: str, payload: dict[str, Any]):
        with self.process_lock:
            if not self.prepare_for_new_command_locked():
                return

            place_overrides = self.compute_place_overrides(payload)
            if place_overrides is None:
                return

            command = self.build_command(place_overrides)
            self.get_logger().info(f'Starting AnyGrasp pick process for label={label}')
            self.get_logger().info('Command: ' + ' '.join(command))

            env = os.environ.copy()
            try:
                self.pick_completed_by_log = False
                self.pick_completion_time = None
                self.pick_process = subprocess.Popen(
                    command,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                self.pick_running = True
                self.pick_process_started_at = time.monotonic()
                self.pick_stdout_thread = threading.Thread(
                    target=self.read_pick_process_output,
                    args=(self.pick_process,),
                    daemon=True,
                )
                self.pick_stdout_thread.start()
            except Exception as exc:
                self.pick_process = None
                self.pick_running = False
                self.pick_process_started_at = None
                self.pick_completed_by_log = False
                self.pick_completion_time = None
                self.pick_stdout_thread = None
                self.get_logger().error(f'Failed to start pick process: {exc}')
                return
            self.get_logger().info(f'Pick process PID={self.pick_process.pid}')

    def stop_callback(self, msg: String):
        del msg
        self.get_logger().warn('Stop command received. Terminating active pick process.')
        with self.process_lock:
            self.stop_active_pick_process_locked()
        self.try_cancel_execute_trajectory()
        self.get_logger().warn('Pick process stopped by user.')
        with self.process_lock:
            if self.return_home_on_stop:
                self.get_logger().info('Returning robot to safe home after stop.')
                if self.start_safe_home_motion_locked('stop'):
                    return
        self.get_logger().info('Ready for next LLM pick command.')

    def check_pick_process(self):
        with self.process_lock:
            if self.pick_process is None:
                if self.pick_running:
                    self.get_logger().warn(
                        'Inconsistent pick state detected. Resetting pick_running flag.'
                    )
                    self.pick_running = False
                    self.pick_process_started_at = None
                return

            return_code = self.pick_process.poll()
            if return_code is not None:
                self.cleanup_finished_pick_process_locked(return_code)
                return

            if self.pick_completed_by_log:
                completion_time = self.pick_completion_time or time.time()
                if time.time() - completion_time >= 0.5:
                    self.cleanup_completed_log_process_locked()
                return

            if self.pick_process_timed_out_locked():
                self.terminate_timed_out_pick_process_locked()
                return

            self.pick_running = True

    def prepare_for_new_command_locked(self) -> bool:
        if self.home_running:
            self.get_logger().warn('Safe home motion is running. Ignoring new command.')
            return False

        if self.pick_process is None:
            if self.pick_running:
                self.get_logger().warn(
                    'Inconsistent pick state detected. Resetting pick_running flag.'
                )
                self.pick_running = False
                self.pick_process_started_at = None
            return True

        return_code = self.pick_process.poll()
        if return_code is None:
            if self.pick_completed_by_log:
                self.cleanup_completed_log_process_locked()
                return not self.home_running
            self.get_logger().warn('Pick process already running. Ignoring new command.')
            return False

        self.cleanup_finished_pick_process_locked(return_code)
        return not self.home_running

    def pick_process_timed_out_locked(self) -> bool:
        if self.pick_process_started_at is None:
            self.pick_process_started_at = time.monotonic()
            return False
        elapsed = time.monotonic() - self.pick_process_started_at
        return elapsed > self.pick_process_timeout_sec

    def terminate_timed_out_pick_process_locked(self):
        process = self.pick_process
        if process is None:
            self.pick_running = False
            self.pick_process_started_at = None
            return

        self.get_logger().warn('Pick process timeout. Terminating process.')
        self.terminate_process_locked(process, terminate_timeout_sec=3.0)
        self.reset_pick_state_locked()
        self.get_logger().info('Ready for next LLM pick command.')

    def cleanup_completed_log_process_locked(self):
        process = self.pick_process
        if process is None:
            self.reset_pick_state_locked()
            return

        return_code = process.poll()
        if return_code is None:
            self.get_logger().info('Pick completion log detected. Cleaning up pick process.')
            self.terminate_process_locked(process, terminate_timeout_sec=2.0)
        else:
            self.get_logger().info(
                f'Pick process finished with return code {return_code}. Ready for next command.'
            )

        should_return_home = self.return_home_after_pick
        self.reset_pick_state_locked()
        if should_return_home:
            self.get_logger().info('Returning robot to safe home after pick.')
            if self.start_safe_home_motion_locked('pick'):
                return
        self.get_logger().info('Ready for next LLM pick command.')

    def cleanup_finished_pick_process_locked(self, return_code: int | None = None):
        if self.pick_process is None:
            self.reset_pick_state_locked()
            return
        if return_code is None:
            return_code = self.pick_process.poll()
        completed_normally = return_code == 0 or self.pick_completed_by_log
        self.get_logger().info(
            f'Pick process finished with return code {return_code}. '
            'Ready for next command.'
        )
        self.reset_pick_state_locked()
        if completed_normally and self.return_home_after_pick:
            self.get_logger().info('Returning robot to safe home after pick.')
            if self.start_safe_home_motion_locked('pick'):
                return
        self.get_logger().info('Ready for next LLM pick command.')

    def stop_active_pick_process_locked(self):
        process = self.pick_process
        if process is not None and process.poll() is None:
            self.terminate_process_locked(process, terminate_timeout_sec=2.0)
        self.reset_pick_state_locked()

    def reset_pick_state_locked(self):
        self.pick_process = None
        self.pick_running = False
        self.pick_process_started_at = None
        self.pick_completed_by_log = False
        self.pick_completion_time = None
        self.pick_stdout_thread = None

    def start_safe_home_motion_locked(self, reason: str) -> bool:
        if self.home_running:
            self.get_logger().warn('Safe home motion is already running.')
            return True

        command = self.build_safe_home_command()
        self.home_running = True
        self.home_process = None
        self.home_thread = threading.Thread(
            target=self.run_safe_home_motion_thread,
            args=(command, reason),
            daemon=True,
        )
        self.home_thread.start()
        return True

    def build_safe_home_command(self) -> list[str]:
        return [
            'ros2',
            'run',
            'yolo_webcam_pkg',
            'safe_home_node',
            '--ros-args',
            '-p',
            f'safe_z:={self._ros_value(self.home_safe_z)}',
            '-p',
            f'home_x:={self._ros_value(self.home_x)}',
            '-p',
            f'home_y:={self._ros_value(self.home_y)}',
            '-p',
            f'home_z:={self._ros_value(self.home_z)}',
            '-p',
            f'velocity_scaling:={self._ros_value(self.home_velocity_scaling)}',
            '-p',
            f'acceleration_scaling:={self._ros_value(self.home_acceleration_scaling)}',
            '-p',
            f'use_current_orientation:={self._ros_value(self.home_use_current_orientation)}',
            '-p',
            'execute:=true',
        ]

    def run_safe_home_motion_thread(self, command: list[str], reason: str):
        self.get_logger().info('Command: ' + ' '.join(command))
        process = None
        try:
            process = subprocess.Popen(
                command,
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            with self.process_lock:
                self.home_process = process

            if process.stdout is not None:
                for line in process.stdout:
                    text = line.rstrip()
                    if text:
                        self.get_logger().info(f'[home] {text}')

            return_code = process.wait()
            if return_code == 0:
                self.get_logger().info('Safe home motion complete.')
            else:
                self.get_logger().error(
                    f'Safe home motion failed with return code {return_code}.'
                )
        except Exception as exc:
            self.get_logger().error(f'Safe home motion failed after {reason}: {exc}')
        finally:
            with self.process_lock:
                if process is None or self.home_process is process:
                    self.home_process = None
                    self.home_running = False
                    self.home_thread = None
            self.get_logger().info('Ready for next LLM pick command.')

    def terminate_process_locked(
        self,
        process: subprocess.Popen,
        terminate_timeout_sec: float,
    ):
        try:
            process.terminate()
            try:
                return_code = process.wait(timeout=terminate_timeout_sec)
                self.get_logger().warn(
                    f'Pick process finished with return code {return_code} after terminate().'
                )
            except subprocess.TimeoutExpired:
                self.get_logger().warn('Pick process still alive after terminate(). Killing process.')
                process.kill()
                return_code = process.wait(timeout=2.0)
                self.get_logger().warn(f'Pick process killed with return code {return_code}.')
        except Exception as exc:
            self.get_logger().error(f'Failed to terminate pick process: {exc}')

    def read_pick_process_output(self, process: subprocess.Popen):
        stream = process.stdout
        if stream is None:
            return

        try:
            for line in stream:
                text = line.rstrip()
                if text:
                    self.get_logger().info(f'[pick] {text}')
                if any(pattern in text for pattern in PICK_COMPLETION_LOG_PATTERNS):
                    with self.process_lock:
                        if self.pick_process is process and not self.pick_completed_by_log:
                            self.pick_completed_by_log = True
                            self.pick_completion_time = time.time()
                            self.get_logger().info(
                                'Pick completion log detected. Cleaning up pick process.'
                            )
        except Exception as exc:
            self.get_logger().warn(f'Failed to read pick process output: {exc}')

    def compute_place_overrides(self, payload: dict[str, Any]) -> dict[str, float] | None:
        place_mode = payload.get('place_mode')
        if place_mode == 'zone':
            return self.compute_zone_place_overrides(payload)

        if place_mode != 'relative':
            return {}

        try:
            place_dx = float(payload.get('place_dx', 0.0))
            place_dy = float(payload.get('place_dy', 0.0))
            place_dz = float(payload.get('place_dz', 0.0))
        except (TypeError, ValueError) as exc:
            self.get_logger().error(f'Invalid relative place command: {exc}')
            return None

        self.get_logger().info(
            'Received relative place command: '
            f'dx={place_dx:.3f}, dy={place_dy:.3f}, dz={place_dz:.3f}'
        )

        if self.latest_yolo_target_base is None:
            self.get_logger().error(
                'Relative place command has no YOLO target base. Aborting pick trigger.'
            )
            return None

        x_base, y_base, z_base = self.latest_yolo_target_base
        self.get_logger().info(
            'Using YOLO target base for place reference: '
            f'x={x_base:.3f}, y={y_base:.3f}, z={z_base:.3f}'
        )

        place_x = x_base + place_dx
        place_y = y_base + place_dy
        place_z = self.place_z_default + place_dz
        self.get_logger().info(
            'Computed place pose: '
            f'place_x={place_x:.3f}, place_y={place_y:.3f}, place_z={place_z:.3f}'
        )

        if not self.place_pose_is_safe(place_x, place_y, place_z):
            self.get_logger().error(
                'Computed place pose is outside safety limits. Aborting pick trigger. '
                f'place=({place_x:.3f}, {place_y:.3f}, {place_z:.3f}) '
                f'X=[{self.place_x_min:.3f}, {self.place_x_max:.3f}], '
                f'Y=[{self.place_y_min:.3f}, {self.place_y_max:.3f}], '
                f'Z=[{self.place_z_min:.3f}, {self.place_z_max:.3f}]'
            )
            return None

        return {
            'place_x': round(place_x, 6),
            'place_y': round(place_y, 6),
            'place_z': round(place_z, 6),
        }

    def compute_zone_place_overrides(self, payload: dict[str, Any]) -> dict[str, float] | None:
        zone = str(payload.get('zone', '')).strip().upper()
        self.get_logger().info(f'Received zone place command: zone={zone}')

        if zone == 'A':
            place_x = self.zone_a_place_x
            place_y = self.zone_a_place_y
            place_z = self.zone_a_place_z
        elif zone == 'B':
            place_x = self.zone_b_place_x
            place_y = self.zone_b_place_y
            place_z = self.zone_b_place_z
        else:
            self.get_logger().error(
                f'Unsupported zone place command zone={zone!r}. Aborting pick trigger.'
            )
            return None

        self.get_logger().info(
            f'Using zone {zone} place pose: '
            f'place_x={place_x:.3f}, place_y={place_y:.3f}, place_z={place_z:.3f}'
        )

        if not self.place_pose_is_safe(place_x, place_y, place_z):
            self.get_logger().error(
                'Computed zone place pose is outside safety limits. Aborting pick trigger. '
                f'place=({place_x:.3f}, {place_y:.3f}, {place_z:.3f}) '
                f'X=[{self.place_x_min:.3f}, {self.place_x_max:.3f}], '
                f'Y=[{self.place_y_min:.3f}, {self.place_y_max:.3f}], '
                f'Z=[{self.place_z_min:.3f}, {self.place_z_max:.3f}]'
            )
            return None

        return {
            'place_x': round(place_x, 6),
            'place_y': round(place_y, 6),
            'place_z': round(place_z, 6),
        }

    def place_pose_is_safe(self, place_x: float, place_y: float, place_z: float) -> bool:
        return (
            self.place_x_min <= place_x <= self.place_x_max
            and self.place_y_min <= place_y <= self.place_y_max
            and self.place_z_min <= place_z <= self.place_z_max
        )

    def build_command(self, parameter_overrides: dict[str, Any] | None = None) -> list[str]:
        if parameter_overrides is None:
            parameter_overrides = {}

        command = [
            'ros2',
            'run',
            'franka_visual_servo',
            'anygrasp_pick_place_node',
            '--ros-args',
            '-p',
            f'input_topic:={self.input_topic}',
            '-p',
            f'dry_run:={self._ros_value(self.dry_run)}',
            '-p',
            f'plan_only:={self._ros_value(self.plan_only)}',
            '-p',
            f'execute:={self._ros_value(self.execute)}',
            '-p',
            f'once:={self._ros_value(self.once)}',
        ]

        for name in COMMAND_PARAMETER_DEFAULTS:
            value = parameter_overrides.get(name, self.get_parameter(name).value)
            command.extend(['-p', f'{name}:={self._ros_value(value)}'])

        return command

    @staticmethod
    def _ros_value(value: Any) -> str:
        if isinstance(value, bool):
            return 'true' if value else 'false'
        return str(value)

    def get_current_target_label_hint(self) -> str | None:
        return getattr(self, 'latest_target_label', None)

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

    def try_cancel_execute_trajectory(self):
        if self.execute_trajectory_cancel_client is None:
            self.get_logger().warn(
                'MoveIt execute trajectory cancel client is unavailable.'
            )
            return

        try:
            if not self.execute_trajectory_cancel_client.wait_for_service(timeout_sec=0.2):
                self.get_logger().warn(
                    'MoveIt execute trajectory cancel service is unavailable.'
                )
                return

            request = CancelGoal.Request()
            future = self.execute_trajectory_cancel_client.call_async(request)
            future.add_done_callback(self.handle_cancel_execute_trajectory_result)
        except Exception as exc:
            self.get_logger().warn(f'Failed to request MoveIt execute trajectory cancel: {exc}')

    def handle_cancel_execute_trajectory_result(self, future):
        try:
            response = future.result()
            self.get_logger().warn(
                f'MoveIt execute trajectory cancel response return_code={response.return_code}.'
            )
        except Exception as exc:
            self.get_logger().warn(f'MoveIt execute trajectory cancel failed: {exc}')

    def destroy_node(self):
        with self.process_lock:
            process = self.pick_process
            home_process = self.home_process

        if process is not None and process.poll() is None:
            self.get_logger().warn('Trigger node shutting down while pick process is still running.')
            self.terminate_process_on_shutdown(process)

        if home_process is not None and home_process.poll() is None:
            self.get_logger().warn('Trigger node shutting down while safe home process is still running.')
            self.terminate_home_process_on_shutdown(home_process)

        super().destroy_node()

    def terminate_process_on_shutdown(self, process: subprocess.Popen):
        try:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        except Exception as exc:
            self.get_logger().error(f'Failed to terminate pick process on shutdown: {exc}')
        finally:
            with self.process_lock:
                if self.pick_process is process:
                    self.reset_pick_state_locked()

    def terminate_home_process_on_shutdown(self, process: subprocess.Popen):
        try:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        except Exception as exc:
            self.get_logger().error(f'Failed to terminate safe home process on shutdown: {exc}')
        finally:
            with self.process_lock:
                if self.home_process is process:
                    self.home_process = None
                    self.home_running = False
                    self.home_thread = None


def main(args=None):
    rclpy.init(args=args)
    node = LLMAnyGraspTriggerNode()

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
