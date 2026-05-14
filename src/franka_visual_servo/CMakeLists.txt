cmake_minimum_required(VERSION 3.8)
project(franka_visual_servo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(gpd_ros2_msgs REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(franka_visual_servo_node src/franka_visual_servo.cpp)

ament_target_dependencies(franka_visual_servo_node
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  tf2_ros
)

add_executable(yolo_hover_linear_node src/yolo_hover_linear_node.cpp)

ament_target_dependencies(yolo_hover_linear_node
  rclcpp
  std_msgs
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
)

add_executable(move_to_ready_pose_node src/move_to_ready_pose_node.cpp)

ament_target_dependencies(move_to_ready_pose_node
  rclcpp
  moveit_ros_planning_interface
)

add_executable(move_left_10cm_node src/move_left_10cm_node.cpp)

ament_target_dependencies(move_left_10cm_node
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  tf2
  tf2_ros
)

add_executable(move_relative_tcp_node src/move_relative_tcp_node.cpp)

ament_target_dependencies(move_relative_tcp_node
  rclcpp
  geometry_msgs
  moveit_core
  moveit_ros_planning_interface
  tf2
  tf2_ros
)

add_executable(move_to_xyz_node src/move_to_xyz_node.cpp)

ament_target_dependencies(move_to_xyz_node
  rclcpp
  geometry_msgs
  moveit_core
  moveit_ros_planning_interface
)

add_executable(gpd_client_node src/gpd_client_node.cpp)

ament_target_dependencies(gpd_client_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  gpd_ros2_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(gpd_pick_execute_node src/gpd_pick_execute_node.cpp)

ament_target_dependencies(gpd_pick_execute_node
  rclcpp
  rclcpp_action
  geometry_msgs
  gpd_ros2_msgs
  franka_msgs
  moveit_msgs
  moveit_ros_planning_interface
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(gpd_grasp_tf_node src/gpd_grasp_tf_node.cpp)

ament_target_dependencies(gpd_grasp_tf_node
  rclcpp
  geometry_msgs
  gpd_ros2_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(fr3_finger_joint_state_publisher_node src/fr3_finger_joint_state_publisher_node.cpp)

ament_target_dependencies(fr3_finger_joint_state_publisher_node
  rclcpp
  sensor_msgs
)

install(TARGETS
  franka_visual_servo_node
  yolo_hover_linear_node
  move_to_ready_pose_node
  move_left_10cm_node
  move_relative_tcp_node
  move_to_xyz_node
  gpd_client_node
  gpd_grasp_tf_node
  gpd_pick_execute_node
  fr3_finger_joint_state_publisher_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

add_executable(anygrasp_hover_move_node src/anygrasp_hover_move_node.cpp)

ament_target_dependencies(anygrasp_hover_move_node
  rclcpp
  std_msgs
  geometry_msgs
  moveit_ros_planning_interface
)

install(TARGETS
  anygrasp_hover_move_node
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(anygrasp_pick_place_node src/anygrasp_pick_place_node.cpp)

ament_target_dependencies(anygrasp_pick_place_node
  rclcpp
  rclcpp_action
  std_msgs
  geometry_msgs
  franka_msgs
  moveit_msgs
  moveit_ros_planning_interface
  tf2_ros
)

target_link_libraries(anygrasp_pick_place_node
  Eigen3::Eigen
)

install(TARGETS
  anygrasp_pick_place_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
