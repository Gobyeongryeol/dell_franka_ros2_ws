# Dell Franka ROS2 Workspace

ROS2 Humble 기반 Franka FR3, RealSense D455, AnyGrasp, YOLO pick-and-place 실험용 소스 패키지 모음입니다.

## Included Packages

- `franka_visual_servo`: AnyGrasp hover/pick-place, planar-yaw grasp, Franka gripper action, camera-view offset compensation
- `anygrasp_ros2_bridge`: AnyGrasp detection bridge, candidate filtering, camera/base TF conversion
- `yolo_webcam_pkg`: YOLO detection, depth to 3D, camera 3D to base conversion
- `yolo_detector`: YOLO compressed-image detector node

## Not Included

This repository intentionally does not include:

- ROS2 `build/`, `install/`, `log/`
- AnyGrasp SDK
- AnyGrasp checkpoints
- AnyGrasp license files
- YOLO weights
- datasets
- debug images
- rosbag files

## Build

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Documents

- `docs/run_anygrasp_pick_place.md`
- `docs/run_yolo_pipeline.md`
- `docs/setup_dependencies.md`
