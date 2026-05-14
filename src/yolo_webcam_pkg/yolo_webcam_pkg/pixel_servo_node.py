import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import String


class PixelServoNode(Node):
    def __init__(self):
        super().__init__('pixel_servo_node')

        self.declare_parameter('deadband_x', 30.0)
        self.declare_parameter('deadband_y', 30.0)
        self.declare_parameter('scale_x', 0.0005)
        self.declare_parameter('scale_y', 0.0005)
        self.declare_parameter('max_step_x', 0.02)
        self.declare_parameter('max_step_y', 0.02)
        self.declare_parameter('invert_x', False)
        self.declare_parameter('invert_y', False)

        self.deadband_x = self.get_parameter('deadband_x').value
        self.deadband_y = self.get_parameter('deadband_y').value
        self.scale_x = self.get_parameter('scale_x').value
        self.scale_y = self.get_parameter('scale_y').value
        self.max_step_x = self.get_parameter('max_step_x').value
        self.max_step_y = self.get_parameter('max_step_y').value
        self.invert_x = self.get_parameter('invert_x').value
        self.invert_y = self.get_parameter('invert_y').value

        self.error_sub = self.create_subscription(
            Point,
            '/visual_servo/error',
            self.error_callback,
            10
        )

        self.step_pub = self.create_publisher(Point, '/visual_servo/step_cmd', 10)
        self.debug_pub = self.create_publisher(String, '/visual_servo/step_text', 10)

        self.get_logger().info(
            f'PixelServoNode started. deadband=({self.deadband_x}, {self.deadband_y}), '
            f'scale=({self.scale_x}, {self.scale_y}), '
            f'max_step=({self.max_step_x}, {self.max_step_y}), '
            f'invert_x={self.invert_x}, invert_y={self.invert_y}'
        )

    @staticmethod
    def clamp(value: float, limit: float) -> float:
        if value > limit:
            return limit
        if value < -limit:
            return -limit
        return value

    def error_callback(self, msg: Point):
        error_x = float(msg.x)
        error_y = float(msg.y)

        step_x = 0.0
        step_y = 0.0

        if abs(error_x) > self.deadband_x:
            step_x = error_x * self.scale_x

        if abs(error_y) > self.deadband_y:
            step_y = error_y * self.scale_y

        if self.invert_x:
            step_x *= -1.0
        if self.invert_y:
            step_y *= -1.0

        step_x = self.clamp(step_x, self.max_step_x)
        step_y = self.clamp(step_y, self.max_step_y)

        step_msg = Point()
        step_msg.x = step_x
        step_msg.y = step_y
        step_msg.z = 0.0
        self.step_pub.publish(step_msg)

        debug = String()
        debug.data = (
            f'error_x={error_x:.1f}, error_y={error_y:.1f}, '
            f'step_x={step_x:.4f}, step_y={step_y:.4f}'
        )
        self.debug_pub.publish(debug)
        self.get_logger().info(debug.data)


def main(args=None):
    rclpy.init(args=args)
    node = PixelServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
