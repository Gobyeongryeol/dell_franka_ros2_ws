# YOLO Pipeline

## YOLO Detection

```bash
export ROS_DOMAIN_ID=1
unset ROS_LOCALHOST_ONLY

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run yolo_webcam_pkg yolo_node
```

## YOLO Depth to 3D

```bash
export ROS_DOMAIN_ID=1
unset ROS_LOCALHOST_ONLY

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run yolo_webcam_pkg yolo_depth_to_3d_node
```

## YOLO 3D to Base

Alias: `yolo3d->base`

```bash
export ROS_DOMAIN_ID=1
unset ROS_LOCALHOST_ONLY

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run yolo_webcam_pkg yolo_3d_to_base_node --ros-args \
  -p source_frame_override:=camera_color_optical_frame \
  -p target_frame:=base
```
