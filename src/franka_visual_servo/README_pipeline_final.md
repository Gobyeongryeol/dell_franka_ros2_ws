# YOLO Hover Linear Pipeline — Final Setup

ROS2 Humble + Franka FR3 + MoveIt2 + RealSense D455 + YOLO

## 보정 현황

- **방식**: 정식 hand-eye calibration이 아닌 `/yolo/target_base` 좌표 기반 경험적 선형 offset calibration
- **TF**: `fr3_link8` → `camera_color_optical_frame` (translation 0, rotation 0)
- **최종 보정 계수** (yolo_hover_linear_final.yaml 값 — 절대 변경 금지):
  ```
  x_offset_kx: -0.8426   x_offset_ky: -1.6070   x_offset_b:  0.2161
  y_offset_kx:  0.6149   y_offset_ky: -0.8378   y_offset_b: -0.0977
  ```
  적용식:
  ```
  final_x_offset = -0.8426 * target_x - 1.6070 * target_y + 0.2161
  final_y_offset =  0.6149 * target_x - 0.8378 * target_y - 0.0977
  ```

---

## TF 트리

```
base ──(identity, 터미널 5-1)──► fr3_link0
                                     │
                               fr3_link1 → ... → fr3_link7
                                                     │
                                               fr3_link8
                                                     │
                               (identity, 터미널 5)  ▼
                               camera_color_optical_frame
```

`base → fr3_link0` transform이 없으면 MoveIt planning frame(`base`)과
카메라 frame이 별개의 TF tree로 분리되어 아래 에러가 발생한다:

```
[WARN] Unable to transform object from frame 'camera_color_optical_frame'
       to planning frame 'base' ... Tf has two or more unconnected trees.
```

---

## 실행 순서 (터미널 8개)

모든 터미널 공통 source 순서 (반드시 이 순서로):
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

> `source ~/franka_ros2_ws/install/setup.bash` 는 이 PC에 없으므로 사용하지 않는다.

---

### 터미널 1 — RealSense

```bash
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

---

### 터미널 2 — MoveIt + Franka

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=172.16.0.2 \
  robot_type:=fr3 \
  load_gripper:=true
```

**MoveIt 준비 확인** (별도 터미널에서 실행 또는 새 탭):
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 node list | grep move_group
ros2 action list | grep execute
```

기대 결과:
- `/move_group` 노드가 목록에 보여야 한다
- `/execute_trajectory` 액션이 보여야 한다
- MoveIt 터미널에 `You can start planning now!` 문구가 떠야 한다

> 터미널 2가 완전히 준비되기 전에 터미널 7을 실행하면 안 된다.

---

### 터미널 3 — YOLO

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run yolo_webcam_pkg yolo_node --ros-args \
  -p image_topic:=/camera/camera/color/image_raw \
  -p model_path:=$HOME/ros2_ws/runs/detect/train/weights/best.pt \
  -p conf_threshold:=0.01
```

---

### 터미널 4 — YOLO depth to 3D

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run yolo_webcam_pkg yolo_depth_to_3d_node
```

---

### 터미널 5 — Static TF (fr3_link8 → camera) + (base → fr3_link0) [통합 launch]

두 static transform을 한 번에 실행 (권장):
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch franka_visual_servo static_tf_camera_to_base.launch.py
```

또는 개별 실행:

**터미널 5** — fr3_link8 → camera_color_optical_frame:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run tf2_ros static_transform_publisher \
  --x 0.00 --y 0.00 --z 0.00 \
  --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id fr3_link8 \
  --child-frame-id camera_color_optical_frame
```

**터미널 5-1** — base → fr3_link0 (TF tree 연결 필수):
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run tf2_ros static_transform_publisher \
  --x 0.00 --y 0.00 --z 0.00 \
  --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id base \
  --child-frame-id fr3_link0
```

> 이 터미널들은 닫으면 안 됩니다. 닫으면 TF tree가 끊어진다.

**TF 연결 확인** (터미널 5 실행 후 반드시 확인):
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run tf2_ros tf2_echo base camera_color_optical_frame
```

정상이라면 transform이 1초 간격으로 계속 출력된다.
에러가 나면 터미널 5, 5-1, 또는 MoveIt(터미널 2)이 준비되지 않은 것이다.
**tf2_echo에서 에러가 나면 터미널 6과 7을 실행하면 안 된다.**

---

### 터미널 6 — YOLO 3D to base

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 run yolo_webcam_pkg yolo_3d_to_base_node --ros-args \
  -p source_frame_override:=camera_color_optical_frame
```

**target_base 토픽 확인**:
```bash
ros2 topic echo /yolo/target_base --once
```

정상이라면 x, y, z 좌표가 출력된다. 출력이 없으면 터미널 3, 4가 정상인지 확인한다.

---

### 터미널 7 — yolo_hover_linear_node (launch 방식)

> **주의: 이 노드는 단독으로 실행하면 안 된다.**
> 반드시 아래 조건을 모두 만족한 후 실행한다:
> 1. 터미널 2(MoveIt)가 `You can start planning now!` 를 출력했을 것
> 2. `ros2 node list | grep move_group` 에서 `/move_group` 이 보일 것
> 3. `ros2 action list | grep execute` 에서 `/execute_trajectory` 가 보일 것
> 4. `ros2 run tf2_ros tf2_echo base camera_color_optical_frame` 이 정상 출력할 것
> 5. `ros2 topic echo /yolo/target_base --once` 에서 좌표가 출력될 것

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch franka_visual_servo yolo_hover_linear_final.launch.py
```

노드가 `/move_group`을 찾지 못하면 아래 메시지가 출력되며 안전하게 종료된다:
```
MoveIt /move_group is not running or not ready after 30 s.
Please run MoveIt in a separate terminal FIRST: ...
```

---

## 최종 실행 전 확인 체크리스트

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# 1. MoveIt 확인
ros2 node list | grep move_group

# 2. MoveIt action 확인
ros2 action list | grep execute

# 3. TF tree 연결 확인
ros2 run tf2_ros tf2_echo base camera_color_optical_frame

# 4. YOLO target 확인
ros2 topic echo /yolo/target_base --once
```

---

## 빌드

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select franka_visual_servo
source ~/ros2_ws/install/setup.bash
```

---

## 주의사항

- `source ~/franka_ros2_ws/install/setup.bash` 는 이 PC에 없으므로 사용하지 않는다
- source 순서는 항상: `source /opt/ros/humble/setup.bash` → `source ~/ros2_ws/install/setup.bash`
- static TF child frame과 `source_frame_override`는 반드시 `camera_color_optical_frame`으로 통일
- `base → fr3_link0` static TF(터미널 5-1)가 없으면 TF tree가 두 조각으로 분리된다
- 터미널 7(yolo_hover_linear_final)은 반드시 터미널 2(MoveIt)가 완전히 뜬 후 실행
- yolo_hover_linear_final.yaml의 계수는 절대 수정하지 않는다
