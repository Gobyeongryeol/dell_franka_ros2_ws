---

# 2026-05-20 추가 작업 — Voice LLM, stop, safe home recovery

## 추가 기능

- `voice_llm_pick_bridge_node.py` 추가
  - `sounddevice` 마이크 입력
  - `webrtcvad` 기반 발화 종료 감지
  - `faster-whisper` STT
  - 기존 `llm_command_parser.py` 재사용
  - `/llm/target_pick` publish

- 음성 stop 명령 추가
  - "멈춰", "정지", "중지", "스톱", "스탑", "stop", "emergency stop"
  - stop 명령은 `/llm/target_pick`이 아니라 `/llm/stop`으로 publish

- `llm_anygrasp_trigger_node.py` 개선
  - `/llm/stop` subscribe
  - active `anygrasp_pick_place_node` subprocess terminate/kill
  - MoveIt `/execute_trajectory` cancel best-effort 시도
  - pick 완료 로그 감지 후 timeout 기다리지 않고 빠른 reset
  - `velocity_scaling=0.15`, `acceleration_scaling=0.15` 기본 적용
  - `return_home_on_stop`, `return_home_after_pick` 파라미터 추가

- `safe_home_node.py` 추가
  - stop 이후 `safe_z`로 먼저 상승
  - 이후 `home_x/y/z`로 복귀
  - MoveIt `/plan_kinematic_path`, `/execute_trajectory` 사용

## 주요 실행 파라미터

Trigger:

```bash
ros2 run yolo_webcam_pkg llm_anygrasp_trigger_node --ros-args \
  -p return_home_on_stop:=true \
  -p return_home_after_pick:=false \
  -p home_safe_z:=0.55 \
  -p home_x:=0.307 \
  -p home_y:=0.000 \
  -p home_z:=0.476 \
  -p velocity_scaling:=0.15 \
  -p acceleration_scaling:=0.15
```
