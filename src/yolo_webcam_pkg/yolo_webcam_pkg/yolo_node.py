import os
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

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.detection_pub = self.create_publisher(String, '/yolo/detections', 10)
        self.center_pub = self.create_publisher(String, '/yolo/center', 10)

        self.get_logger().info(
            f'YOLO node started. topic={image_topic}, model={model_path}, conf={self.conf_threshold}, '
            f'target_class={self.target_class if self.target_class else "ALL"}'
        )

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame, verbose=False)

        annotated = frame.copy()
        published_centers = []

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

                if self.target_class and cls_name != self.target_class:
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
            det_strings = [
                f'{cls_name},{conf:.2f},{cx},{cy},{x1},{y1},{x2},{y2}'
                for cls_name, conf, cx, cy, x1, y1, x2, y2 in published_centers
            ]
            self.detection_pub.publish(String(data=';'.join(det_strings)))

            first = published_centers[0]
            cls_name, conf, cx, cy, x1, y1, x2, y2 = first
            self.center_pub.publish(String(data=f'{cls_name},{cx},{cy}'))

        cv2.imshow('YOLO Webcam', annotated)
        cv2.waitKey(1)


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
