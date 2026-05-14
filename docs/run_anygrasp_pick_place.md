# AnyGrasp Pick-and-Place Pipeline

## Terminal 1 - RealSense D455

```bash
export ROS_DOMAIN_ID=1
unset ROS_LOCALHOST_ONLY

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=true \
  pointcloud.enable:=true \
  rgb_camera.color_profile:=640x480x15 \
  depth_module.depth_profile:=640x480x15
```

## Terminal 2 - MoveIt + Franka FR3

```bash
export ROS_DOMAIN_ID=1
unset ROS_LOCALHOST_ONLY

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=172.16.0.2 \
  robot_type:=fr3 \
  load_gripper:=true
```

## Terminal 3 - Hand-Eye TF

주의: `easy_handeye2 publish.launch.py`는 동시에 켜지 말 것.

```bash
export ROS_DOMAIN_ID=1
unset ROS_LOCALHOST_ONLY

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run tf2_ros static_transform_publisher \
  0.0533 0.0489 -0.0360 \
  0.0008 -0.6889 -0.0088 0.7248 \
  fr3_hand_tcp camera_link
```

## Terminal 4 - AnyGrasp Detection

```bash
export ROS_DOMAIN_ID=1
unset ROS_LOCALHOST_ONLY

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/miniforge3/etc/profile.d/conda.sh

conda activate anygrasp
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=8

ros2 run anygrasp_ros2_bridge anygrasp_detection_node --ros-args \
  -p once:=false \
  -p run_every_n_frames:=30 \
  -p max_points:=20000 \
  -p num_candidates:=20
```

## Terminal 5 - AnyGrasp Candidates to Base

Stable non-clustering version:

```bash
export ROS_DOMAIN_ID=1
unset ROS_LOCALHOST_ONLY

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run anygrasp_ros2_bridge anygrasp_candidates_to_base_node --ros-args \
  -p input_topic:=/anygrasp/grasp_candidates \
  -p output_topic:=/anygrasp/best_safe_grasp_pose_base \
  -p target_frame:=base \
  -p min_score:=-0.1 \
  -p x_min:=0.25 \
  -p x_max:=0.70 \
  -p y_min:=-0.35 \
  -p y_max:=0.35 \
  -p z_min:=-0.05 \
  -p z_max:=0.20 \
  -p hover_z_min:=0.20 \
  -p hover_z_max:=0.45 \
  -p hover_offset_z:=0.20 \
  -p use_candidate_clustering:=false
```

## Terminal 6 - Pick and Place

Tuned version:

```bash
export ROS_DOMAIN_ID=1
unset ROS_LOCALHOST_ONLY

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run franka_visual_servo anygrasp_pick_place_node --ros-args \
  -p input_topic:=/anygrasp/best_safe_grasp_pose_base \
  -p dry_run:=false \
  -p plan_only:=false \
  -p execute:=true \
  -p once:=true \
  -p min_score:=-0.1 \
  -p x_min:=0.20 \
  -p x_max:=0.70 \
  -p y_min:=-0.50 \
  -p y_max:=0.50 \
  -p z_min:=-0.05 \
  -p z_max:=0.80 \
  -p use_target_smoothing:=false \
  -p use_anygrasp_orientation:=true \
  -p anygrasp_orientation_mode:=planar_yaw \
  -p orientation_test_mode:=false \
  -p do_descent:=true \
  -p do_preopen_gripper:=true \
  -p do_gripper:=true \
  -p do_place:=true \
  -p use_camera_view_offset:=true \
  -p camera_view_x_offset:=0.012 \
  -p camera_view_y_offset:=0.006 \
  -p target_frame:=base \
  -p gripper_move_action:=/franka_gripper/move \
  -p gripper_grasp_action:=/franka_gripper/grasp \
  -p gripper_open_width:=0.080 \
  -p gripper_close_width:=0.045 \
  -p gripper_speed:=0.020 \
  -p gripper_force:=20.0 \
  -p gripper_epsilon_inner:=0.035 \
  -p gripper_epsilon_outer:=0.035 \
  -p topdown_orientation_source:=current_at_start \
  -p planar_yaw_axis_index:=0 \
  -p planar_yaw_offset:=0.0 \
  -p safe_z:=0.55 \
  -p lift_z:=0.40 \
  -p grasp_z:=0.03 \
  -p place_x:=0.45 \
  -p place_y:=-0.25 \
  -p place_z:=0.12 \
  -p place_safe_z:=0.55 \
  -p velocity_scaling:=0.05 \
  -p acceleration_scaling:=0.05
```
