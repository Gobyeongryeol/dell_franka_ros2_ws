import os
import json
import time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path', os.path.expanduser('~/ros2_ws/runs/detect/train/weights/best.pt'))
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('target_class', '')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = (
            self.get_parameter('conf_threshold').get_parameter_value().double_value
        )
        self.target_class = self.get_parameter('target_class').get_parameter_value().string_value.strip()
        self.target_label = None
        self.target_command_id = None
        self.latest_center = None
        self._last_target_found_log_time = 0.0
        self._last_target_missing_warn_time = 0.0
        self._target_log_period_sec = 1.0

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        self.llm_target_sub = self.create_subscription(
            String,
            '/llm/target_pick',
            self.llm_target_callback,
            10
        )

        self.detection_pub = self.create_publisher(String, '/yolo/detections', 10)
        self.center_pub = self.create_publisher(String, '/yolo/center', 10)

        self.get_logger().info(
            f'YOLO node started. topic={image_topic}, model={model_path}, conf={self.conf_threshold}, '
            f'target_class={self.target_class if self.target_class else "ALL"}'
        )

    def llm_target_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f'Failed to parse /llm/target_pick JSON: {exc}; data={msg.data}')
            return

        label = data.get('label')
        if not label:
            labels = data.get('labels', [])
            if labels:
                label = labels[0]

        if not label:
            self.get_logger().warn(f'/llm/target_pick message has no label: {msg.data}')
            return

        self.target_label = str(label)
        self.target_command_id = data.get('command_id')
        self.latest_center = None
        self._last_target_found_log_time = 0.0
        self._last_target_missing_warn_time = 0.0

        self.get_logger().info(
            f'Received new LLM target command_id={self.target_command_id}, '
            f'label={self.target_label}. Cleared previous target.'
        )

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame, verbose=False)

        annotated = frame.copy()
        published_centers = []
        llm_target_active = self.target_label is not None
        active_target_class = self.target_label if llm_target_active else self.target_class

        for result in results:
            names = result.names

            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, str(cls_id))

                if conf < self.conf_threshold:
                    continue

                if active_target_class and cls_name != active_target_class:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                published_centers.append((cls_name, conf, cx, cy, x1, y1, x2, y2))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
                label = f'{cls_name} {conf:.2f} ({cx},{cy})'
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

        if published_centers:
            if llm_target_active:
                published_centers.sort(key=lambda item: item[1], reverse=True)
                selected = published_centers[0]
                self._log_selected_target(selected[0], selected[1])
            else:
                selected = published_centers[0]

            det_strings = [
                f'{cls_name},{conf:.2f},{cx},{cy},{x1},{y1},{x2},{y2}'
                for cls_name, conf, cx, cy, x1, y1, x2, y2 in published_centers
            ]
            self.detection_pub.publish(String(data=';'.join(det_strings)))

            cls_name, conf, cx, cy, x1, y1, x2, y2 = selected
            self.latest_center = {
                'command_id': self.target_command_id,
                'label': cls_name,
                'center_x': cx,
                'center_y': cy,
            }
            self.center_pub.publish(String(data=f'{cls_name},{cx},{cy}'))
        elif llm_target_active:
            self.latest_center = None
            self._warn_target_not_found()

        cv2.imshow('YOLO Webcam', annotated)
        cv2.waitKey(1)

    def _log_selected_target(self, label: str, conf: float):
        now = time.monotonic()
        if now - self._last_target_found_log_time < self._target_log_period_sec:
            return

        self._last_target_found_log_time = now
        self.get_logger().info(f'Selected target label={label} conf={conf:.3f}')

    def _warn_target_not_found(self):
        now = time.monotonic()
        if now - self._last_target_missing_warn_time < self._target_log_period_sec:
            return

        self._last_target_missing_warn_time = now
        self.get_logger().warn(
            f'Target label={self.target_label} not found in current YOLO detections. '
            'Not publishing center.'
        )


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
