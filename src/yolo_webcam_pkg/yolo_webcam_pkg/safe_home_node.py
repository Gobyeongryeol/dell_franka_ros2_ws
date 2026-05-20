#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    Constraints,
    MotionPlanRequest,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetMotionPlan
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
import tf2_ros


class SafeHomeNode(Node):
    def __init__(self):
        super().__init__('safe_home_node')

        self.declare_parameter('planning_group', 'fr3_arm')
        self.declare_parameter('target_frame', 'base')
        self.declare_parameter('eef_link', 'fr3_hand_tcp')
        self.declare_parameter('safe_z', 0.55)
        self.declare_parameter('home_x', 0.307)
        self.declare_parameter('home_y', 0.000)
        self.declare_parameter('home_z', 0.476)
        self.declare_parameter('velocity_scaling', 0.15)
        self.declare_parameter('acceleration_scaling', 0.15)
        self.declare_parameter('use_current_orientation', True)
        self.declare_parameter('home_qx', 0.999)
        self.declare_parameter('home_qy', 0.030)
        self.declare_parameter('home_qz', -0.041)
        self.declare_parameter('home_qw', -0.006)
        self.declare_parameter('execute', True)
        self.declare_parameter('planning_time', 5.0)
        self.declare_parameter('position_tolerance', 0.015)
        self.declare_parameter('orientation_tolerance', 0.20)
        self.declare_parameter('tf_timeout_sec', 2.0)
        self.declare_parameter('plan_service', '/plan_kinematic_path')
        self.declare_parameter('execute_action', '/execute_trajectory')

        self.planning_group = str(self.get_parameter('planning_group').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.eef_link = str(self.get_parameter('eef_link').value)
        self.safe_z = float(self.get_parameter('safe_z').value)
        self.home_x = float(self.get_parameter('home_x').value)
        self.home_y = float(self.get_parameter('home_y').value)
        self.home_z = float(self.get_parameter('home_z').value)
        self.velocity_scaling = float(self.get_parameter('velocity_scaling').value)
        self.acceleration_scaling = float(self.get_parameter('acceleration_scaling').value)
        self.use_current_orientation = bool(self.get_parameter('use_current_orientation').value)
        self.home_qx = float(self.get_parameter('home_qx').value)
        self.home_qy = float(self.get_parameter('home_qy').value)
        self.home_qz = float(self.get_parameter('home_qz').value)
        self.home_qw = float(self.get_parameter('home_qw').value)
        self.execute = bool(self.get_parameter('execute').value)
        self.planning_time = float(self.get_parameter('planning_time').value)
        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.orientation_tolerance = float(self.get_parameter('orientation_tolerance').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.plan_service_name = str(self.get_parameter('plan_service').value)
        self.execute_action_name = str(self.get_parameter('execute_action').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
            spin_thread=True,
        )
        self.plan_client = self.create_client(GetMotionPlan, self.plan_service_name)
        self.execute_client = ActionClient(self, ExecuteTrajectory, self.execute_action_name)

        self.get_logger().info(
            'Safe home node ready: '
            f'group={self.planning_group}, target_frame={self.target_frame}, '
            f'eef_link={self.eef_link}, safe_z={self.safe_z:.3f}, '
            f'home=({self.home_x:.3f}, {self.home_y:.3f}, {self.home_z:.3f}), '
            f'execute={self.execute}'
        )

    def run(self) -> bool:
        try:
            current = self.lookup_current_pose()
            orientation = current.rotation if self.use_current_orientation else self.default_orientation()

            safe_z = max(self.safe_z, float(current.translation.z))
            self.get_logger().info(
                f'Moving to safe height: x={current.translation.x:.3f}, '
                f'y={current.translation.y:.3f}, z={safe_z:.3f}'
            )
            if not self.plan_and_execute(
                float(current.translation.x),
                float(current.translation.y),
                safe_z,
                orientation,
                'safe_height',
            ):
                return False

            self.get_logger().info(
                f'Moving to home: x={self.home_x:.3f}, y={self.home_y:.3f}, z={self.home_z:.3f}'
            )
            if not self.plan_and_execute(
                self.home_x,
                self.home_y,
                self.home_z,
                orientation,
                'home',
            ):
                return False

            self.get_logger().info('Safe home motion complete.')
            return True
        except Exception as exc:
            self.get_logger().error(f'Safe home motion failed: {exc}')
            return False

    def lookup_current_pose(self):
        return self.tf_buffer.lookup_transform(
            self.target_frame,
            self.eef_link,
            Time(),
            timeout=Duration(seconds=self.tf_timeout_sec),
        ).transform

    def default_orientation(self):
        from geometry_msgs.msg import Quaternion

        norm = math.sqrt(
            self.home_qx * self.home_qx
            + self.home_qy * self.home_qy
            + self.home_qz * self.home_qz
            + self.home_qw * self.home_qw
        )
        if norm == 0.0:
            raise ValueError('home quaternion has zero length')

        quat = Quaternion()
        quat.x = self.home_qx / norm
        quat.y = self.home_qy / norm
        quat.z = self.home_qz / norm
        quat.w = self.home_qw / norm
        return quat

    def plan_and_execute(self, x, y, z, orientation, name: str) -> bool:
        if not self.plan_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f'MoveIt plan service unavailable: {self.plan_service_name}')
            return False

        request = GetMotionPlan.Request()
        request.motion_plan_request = self.build_motion_plan_request(x, y, z, orientation)

        future = self.plan_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None:
            self.get_logger().error(f'Motion plan request failed for {name}')
            return False

        motion_response = response.motion_plan_response
        if motion_response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'Motion planning failed for {name}: error_code={motion_response.error_code.val}'
            )
            return False

        self.get_logger().info(f'Motion planning succeeded for {name}.')
        if not self.execute:
            self.get_logger().info('execute=false, skipping trajectory execution.')
            return True

        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'MoveIt execute action unavailable: {self.execute_action_name}')
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = motion_response.trajectory

        send_goal_future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f'ExecuteTrajectory goal rejected for {name}')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result is None:
            self.get_logger().error(f'ExecuteTrajectory result missing for {name}')
            return False

        error_code = result.result.error_code
        if error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f'ExecuteTrajectory failed for {name}: error_code={error_code.val}'
            )
            return False

        self.get_logger().info(f'ExecuteTrajectory succeeded for {name}.')
        return True

    def build_motion_plan_request(self, x, y, z, orientation) -> MotionPlanRequest:
        plan_request = MotionPlanRequest()
        plan_request.group_name = self.planning_group
        plan_request.num_planning_attempts = 5
        plan_request.allowed_planning_time = self.planning_time
        plan_request.max_velocity_scaling_factor = self.velocity_scaling
        plan_request.max_acceleration_scaling_factor = self.acceleration_scaling
        plan_request.start_state.is_diff = True

        constraints = Constraints()
        constraints.name = 'safe_home_pose'
        constraints.position_constraints.append(self.position_constraint(x, y, z))
        constraints.orientation_constraints.append(self.orientation_constraint(orientation))
        plan_request.goal_constraints.append(constraints)
        return plan_request

    def position_constraint(self, x, y, z) -> PositionConstraint:
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [self.position_tolerance]

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.w = 1.0

        constraint = PositionConstraint()
        constraint.header.frame_id = self.target_frame
        constraint.link_name = self.eef_link
        constraint.constraint_region.primitives.append(primitive)
        constraint.constraint_region.primitive_poses.append(pose)
        constraint.weight = 1.0
        return constraint

    def orientation_constraint(self, orientation) -> OrientationConstraint:
        constraint = OrientationConstraint()
        constraint.header.frame_id = self.target_frame
        constraint.link_name = self.eef_link
        constraint.orientation = orientation
        constraint.absolute_x_axis_tolerance = self.orientation_tolerance
        constraint.absolute_y_axis_tolerance = self.orientation_tolerance
        constraint.absolute_z_axis_tolerance = self.orientation_tolerance
        constraint.weight = 1.0
        return constraint


def main(args=None):
    rclpy.init(args=args)
    node = SafeHomeNode()
    ok = False
    try:
        ok = node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
