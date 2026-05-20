import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String

from yolo_webcam_pkg.llm_command_parser import llm_chat_or_pick


class LLMPickBridgeNode(Node):
    def __init__(self):
        super().__init__("llm_pick_bridge")

        self.pick_pub = self.create_publisher(String, "/llm/target_pick", 10)
        self.visible_objects = {
            "blue_cube",
            "blue_cylinder",
            "objects-yellow_box",
            "yellow_cylinder",
        }

        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()

        self.get_logger().info("LLM pick bridge node started.")

    def _input_loop(self):
        while rclpy.ok():
            try:
                user_text = input("LLM 명령 입력: ").strip()
            except EOFError:
                if rclpy.ok():
                    rclpy.shutdown()
                break

            if user_text in ["q", "quit", "exit"]:
                if rclpy.ok():
                    rclpy.shutdown()
                break

            result = llm_chat_or_pick(user_text, visible_objects=self.visible_objects)
            print(result)

            if result.get("type") == "pick_queue":
                labels = result.get("labels", [])

                if not labels:
                    self.get_logger().warn("pick_queue 결과에 labels가 없습니다.")
                    continue

                if len(labels) != 1:
                    self.get_logger().warn(
                        f"Expected exactly one label, got {labels}. Not publishing."
                    )
                    print("물체를 하나만 말해줘.")
                    continue

                label = labels[0]
                msg_data = {
                    "command_id": time.time_ns(),
                    "label": label,
                    "labels": labels,
                }
                place_mode = result.get("place_mode")
                if place_mode == "relative":
                    msg_data["place_mode"] = place_mode
                    msg_data["place_dx"] = float(result.get("place_dx", 0.0))
                    msg_data["place_dy"] = float(result.get("place_dy", 0.0))
                    msg_data["place_dz"] = float(result.get("place_dz", 0.0))
                elif place_mode == "zone":
                    msg_data["place_mode"] = place_mode
                    msg_data["zone"] = str(result.get("zone", "")).upper()

                msg = String()
                msg.data = json.dumps(msg_data, ensure_ascii=False)
                self.pick_pub.publish(msg)
                self.get_logger().info(f"Published /llm/target_pick: {msg.data}")

            else:
                self.get_logger().info(result.get("reply", ""))


def main(args=None):
    rclpy.init(args=args)
    node = LLMPickBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, AttributeError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
