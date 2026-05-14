import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

import numpy as np
import cv2
from ultralytics import YOLO


class YoloNode(Node):

    def __init__(self):
        super().__init__('yolo_node')

        self.model = YOLO("yolov8n.pt")

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/front/image/compressed',
            self.callback,
            10
        )

        self.get_logger().info("YOLO node started")

    def callback(self, msg):

        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        results = self.model(frame)

        annotated = results[0].plot()

        cv2.imshow("YOLO Detection", annotated)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = YoloNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
