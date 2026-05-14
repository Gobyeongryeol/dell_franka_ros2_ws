#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "tf2/exceptions.h"
#include "tf2/time.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

class MoveLeft10cmNode : public rclcpp::Node
{
public:
  MoveLeft10cmNode()
  : Node("move_left_10cm_node"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter<std::string>("planning_group", "fr3_arm");
    declare_parameter<std::string>("base_frame", "base");
    declare_parameter<std::string>("tcp_frame", "fr3_hand_tcp");
    declare_parameter<double>("left_offset_m", 0.10);
    declare_parameter<double>("planning_time", 10.0);
    declare_parameter<double>("velocity_scaling", 0.10);
    declare_parameter<double>("acceleration_scaling", 0.10);
    declare_parameter<int>("robot_description_wait_timeout_sec", 30);
    declare_parameter<double>("tf_lookup_timeout_sec", 5.0);
    declare_parameter<bool>("dry_run", false);

    planning_group_ = get_parameter("planning_group").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    tcp_frame_ = get_parameter("tcp_frame").as_string();
    left_offset_m_ = get_parameter("left_offset_m").as_double();
    planning_time_ = get_parameter("planning_time").as_double();
    velocity_scaling_ = get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = get_parameter("acceleration_scaling").as_double();
    robot_description_wait_timeout_sec_ =
      get_parameter("robot_description_wait_timeout_sec").as_int();
    tf_lookup_timeout_sec_ = get_parameter("tf_lookup_timeout_sec").as_double();
    dry_run_ = get_parameter("dry_run").as_bool();

    RCLCPP_INFO(
      get_logger(),
      "move_left_10cm_node params: group=%s base=%s tcp=%s left_offset_m=%.4f "
      "planning_time=%.1f vel=%.2f acc=%.2f dry_run=%s",
      planning_group_.c_str(), base_frame_.c_str(), tcp_frame_.c_str(), left_offset_m_,
      planning_time_, velocity_scaling_, acceleration_scaling_,
      dry_run_ ? "true" : "false");
  }

  void run()
  {
    if (!fetchRobotDescription()) {
      return;
    }

    try {
      move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), planning_group_);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "MoveGroupInterface construction failed: %s", e.what());
      return;
    }

    move_group_->setEndEffectorLink(tcp_frame_);
    move_group_->setPoseReferenceFrame(base_frame_);
    move_group_->setPlanningTime(planning_time_);
    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);

    RCLCPP_INFO(
      get_logger(), "MoveGroup ready. planning_frame=%s eef=%s pose_reference_frame=%s",
      move_group_->getPlanningFrame().c_str(),
      move_group_->getEndEffectorLink().c_str(),
      base_frame_.c_str());

    RCLCPP_INFO(get_logger(), "Waiting 2s for CurrentStateMonitor and TF...");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    TransformStamped current_tf;
    if (!lookupCurrentTcpTransform(current_tf)) {
      return;
    }

    geometry_msgs::msg::PoseStamped target_pose;
    target_pose.header.frame_id = base_frame_;
    target_pose.header.stamp = now();
    target_pose.pose.position.x = current_tf.transform.translation.x;
    target_pose.pose.position.y = current_tf.transform.translation.y + left_offset_m_;
    target_pose.pose.position.z = current_tf.transform.translation.z;
    target_pose.pose.orientation = current_tf.transform.rotation;

    RCLCPP_INFO(
      get_logger(),
      "Current %s in %s: position=(%.6f, %.6f, %.6f) "
      "orientation=(x=%.6f, y=%.6f, z=%.6f, w=%.6f)",
      tcp_frame_.c_str(), base_frame_.c_str(),
      current_tf.transform.translation.x,
      current_tf.transform.translation.y,
      current_tf.transform.translation.z,
      current_tf.transform.rotation.x,
      current_tf.transform.rotation.y,
      current_tf.transform.rotation.z,
      current_tf.transform.rotation.w);

    RCLCPP_INFO(
      get_logger(),
      "Target pose in %s: position=(%.6f, %.6f, %.6f) "
      "orientation=(x=%.6f, y=%.6f, z=%.6f, w=%.6f)",
      base_frame_.c_str(),
      target_pose.pose.position.x,
      target_pose.pose.position.y,
      target_pose.pose.position.z,
      target_pose.pose.orientation.x,
      target_pose.pose.orientation.y,
      target_pose.pose.orientation.z,
      target_pose.pose.orientation.w);

    move_group_->setPoseTarget(target_pose, tcp_frame_);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool planned = static_cast<bool>(move_group_->plan(plan));
    if (!planned) {
      RCLCPP_ERROR(get_logger(), "Planning failed. Robot will not move.");
      move_group_->clearPoseTargets();
      return;
    }

    RCLCPP_INFO(get_logger(), "Planning succeeded.");

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "dry_run=true, skipping execute.");
      move_group_->clearPoseTargets();
      return;
    }

    const auto result = move_group_->execute(plan);
    move_group_->clearPoseTargets();

    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Execution failed (code=%d).", result.val);
      return;
    }

    RCLCPP_INFO(
      get_logger(), "Move complete: %s y += %.4f m in %s frame.",
      tcp_frame_.c_str(), left_offset_m_, base_frame_.c_str());
  }

private:
  using TransformStamped = geometry_msgs::msg::TransformStamped;

  bool fetchRobotDescription()
  {
    RCLCPP_INFO(
      get_logger(), "Fetching robot_description from /move_group (timeout=%d s)...",
      robot_description_wait_timeout_sec_);

    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(
      shared_from_this(), "/move_group");

    bool ready = false;
    for (int i = 0; i < robot_description_wait_timeout_sec_ * 2 && rclcpp::ok(); ++i) {
      if (param_client->service_is_ready()) {
        ready = true;
        break;
      }
      if (i % 2 == 0) {
        RCLCPP_INFO(
          get_logger(), "Waiting for /move_group parameter service... (%d/%d s)",
          i / 2, robot_description_wait_timeout_sec_);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    if (!ready) {
      RCLCPP_ERROR(
        get_logger(),
        "/move_group parameter service not ready after %d s. Start MoveIt first.",
        robot_description_wait_timeout_sec_);
      return false;
    }

    auto future = param_client->get_parameters(
      {"robot_description", "robot_description_semantic"});

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
      if (!rclcpp::ok() || std::chrono::steady_clock::now() > deadline) {
        RCLCPP_ERROR(get_logger(), "Timed out waiting for robot descriptions.");
        return false;
      }
    }

    bool ok = true;
    for (const auto & param : future.get()) {
      if (param.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
        RCLCPP_ERROR(get_logger(), "'%s' is not set in /move_group.", param.get_name().c_str());
        ok = false;
        continue;
      }

      try {
        declare_parameter(param.get_name(), param.get_parameter_value());
      } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
        set_parameter(param);
      }

      RCLCPP_INFO(
        get_logger(), "Loaded '%s' (%zu chars) from /move_group.",
        param.get_name().c_str(), param.as_string().size());
    }

    return ok;
  }

  bool lookupCurrentTcpTransform(TransformStamped & transform)
  {
    try {
      transform = tf_buffer_.lookupTransform(
        base_frame_, tcp_frame_, tf2::TimePointZero,
        tf2::durationFromSec(tf_lookup_timeout_sec_));
      return true;
    } catch (const tf2::TransformException & e) {
      RCLCPP_ERROR(
        get_logger(), "Failed to lookup TF %s -> %s: %s",
        base_frame_.c_str(), tcp_frame_.c_str(), e.what());
      return false;
    }
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::string planning_group_;
  std::string base_frame_;
  std::string tcp_frame_;
  double left_offset_m_;
  double planning_time_;
  double velocity_scaling_;
  double acceleration_scaling_;
  int robot_description_wait_timeout_sec_;
  double tf_lookup_timeout_sec_;
  bool dry_run_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveLeft10cmNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread spin_thread([&executor]() { executor.spin(); });

  node->run();

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
