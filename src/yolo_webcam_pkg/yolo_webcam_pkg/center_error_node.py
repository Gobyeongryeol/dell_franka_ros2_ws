import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point


class CenterErrorNode(Node):
    def __init__(self):
        super().__init__('center_error_node')

        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        self.image_width = self.get_parameter('image_width').get_parameter_value().integer_value
        self.image_height = self.get_parameter('image_height').get_parameter_value().integer_value

        self.cx_ref = self.image_width // 2
        self.cy_ref = self.image_height // 2

        self.center_sub = self.create_subscription(
            String,
            '/yolo/center',
            self.center_callback,
            10
        )

        self.error_pub = self.create_publisher(Point, '/visual_servo/error', 10)
        self.debug_pub = self.create_publisher(String, '/visual_servo/error_text', 10)

        self.get_logger().info(
            f'CenterErrorNode started. image_width={self.image_width}, '
            f'image_height={self.image_height}, ref=({self.cx_ref},{self.cy_ref})'
        )

    def center_callback(self, msg: String):
        data = msg.data.strip()
        if not data:
            return

        parts = data.split(',')
        if len(parts) != 3:
            self.get_logger().warning(f'Invalid /yolo/center format: {data}')
            return

        cls_name = parts[0]

        try:
            cx = int(parts[1])
            cy = int(parts[2])
        except ValueError:
            self.get_logger().warning(f'Invalid center values: {data}')
            return

        error_x = float(cx - self.cx_ref)
        error_y = float(cy - self.cy_ref)

        error_msg = Point()
        error_msg.x = error_x
        error_msg.y = error_y
        error_msg.z = 0.0
        self.error_pub.publish(error_msg)

        debug_msg = String()
        debug_msg.data = (
            f'class={cls_name}, cx={cx}, cy={cy}, '
            f'error_x={error_x}, error_y={error_y}'
        )
        self.debug_pub.publish(debug_msg)

        self.get_logger().info(debug_msg.data)


def main(args=None):
    rclpy.init(args=args)
    node = CenterErrorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
