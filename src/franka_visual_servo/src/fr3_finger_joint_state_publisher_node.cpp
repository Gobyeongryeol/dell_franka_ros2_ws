#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

class Fr3FingerJointStatePublisherNode : public rclcpp::Node
{
public:
  Fr3FingerJointStatePublisherNode()
  : Node("fr3_finger_joint_state_publisher")
  {
    declare_parameter<std::string>("source_topic", "franka/joint_states");
    declare_parameter<std::string>("output_topic", "joint_states");
    declare_parameter<double>("finger_joint_position", 0.04);
    declare_parameter<double>("finger_joint_velocity", 0.0);
    declare_parameter<double>("finger_joint_effort", 0.0);
    declare_parameter<std::vector<std::string>>(
      "finger_joint_names",
      std::vector<std::string>{"fr3_finger_joint1", "fr3_finger_joint2"});

    source_topic_ = get_parameter("source_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    finger_joint_position_ = get_parameter("finger_joint_position").as_double();
    finger_joint_velocity_ = get_parameter("finger_joint_velocity").as_double();
    finger_joint_effort_ = get_parameter("finger_joint_effort").as_double();
    finger_joint_names_ = get_parameter("finger_joint_names").as_string_array();

    pub_ = create_publisher<sensor_msgs::msg::JointState>(output_topic_, rclcpp::QoS(20));
    sub_ = create_subscription<sensor_msgs::msg::JointState>(
      source_topic_,
      rclcpp::QoS(50),
      std::bind(&Fr3FingerJointStatePublisherNode::jointStateCallback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "[fr3_finger_joint_state_publisher] source=%s output=%s finger_position=%.3f",
      source_topic_.c_str(), output_topic_.c_str(), finger_joint_position_);
  }

private:
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    sensor_msgs::msg::JointState merged;
    merged.header = msg->header;
    if (merged.header.stamp.sec == 0 && merged.header.stamp.nanosec == 0) {
      merged.header.stamp = now();
    }

    const auto joint_count = msg->name.size();
    const bool has_position = msg->position.size() == joint_count;
    const bool has_velocity = msg->velocity.size() == joint_count;
    const bool has_effort = msg->effort.size() == joint_count;

    merged.name = msg->name;
    if (has_position) {
      merged.position = msg->position;
    }
    if (has_velocity) {
      merged.velocity = msg->velocity;
    }
    if (has_effort) {
      merged.effort = msg->effort;
    }

    for (const auto & finger_name : finger_joint_names_) {
      const auto it = std::find(merged.name.begin(), merged.name.end(), finger_name);
      if (it != merged.name.end()) {
        continue;
      }

      merged.name.push_back(finger_name);
      if (has_position) {
        merged.position.push_back(finger_joint_position_);
      }
      if (has_velocity) {
        merged.velocity.push_back(finger_joint_velocity_);
      }
      if (has_effort) {
        merged.effort.push_back(finger_joint_effort_);
      }
    }

    pub_->publish(merged);
  }

  std::string source_topic_;
  std::string output_topic_;
  double finger_joint_position_{0.04};
  double finger_joint_velocity_{0.0};
  double finger_joint_effort_{0.0};
  std::vector<std::string> finger_joint_names_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Fr3FingerJointStatePublisherNode>());
  rclcpp::shutdown();
  return 0;
}
