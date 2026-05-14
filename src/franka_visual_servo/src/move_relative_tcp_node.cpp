#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "moveit/robot_state/robot_state.h"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/exceptions.h"
#include "tf2/time.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

class MoveRelativeTcpNode : public rclcpp::Node
{
public:
  MoveRelativeTcpNode()
  : Node("move_relative_tcp_node"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter<std::string>("planning_group", "fr3_arm");
    declare_parameter<std::string>("base_frame", "fr3_link0");
    declare_parameter<std::string>("tcp_frame", "fr3_hand_tcp");
    declare_parameter<double>("dx", 0.0);
    declare_parameter<double>("dy", 0.0);
    declare_parameter<double>("dz", 0.0);
    declare_parameter<bool>("dry_run", false);
    declare_parameter<double>("planning_time", 10.0);
    declare_parameter<int>("robot_description_wait_timeout_sec", 30);
    declare_parameter<double>("tf_warmup_sec", 5.0);
    declare_parameter<double>("tf_lookup_timeout_sec", 5.0);
    declare_parameter<double>("moveit_state_wait_sec", 5.0);

    planning_group_ = get_parameter("planning_group").as_string();
    base_frame_ = get_parameter("base_frame").as_string();
    tcp_frame_ = get_parameter("tcp_frame").as_string();
    dx_ = get_parameter("dx").as_double();
    dy_ = get_parameter("dy").as_double();
    dz_ = get_parameter("dz").as_double();
    dry_run_ = get_parameter("dry_run").as_bool();
    planning_time_ = get_parameter("planning_time").as_double();
    robot_description_wait_timeout_sec_ =
      get_parameter("robot_description_wait_timeout_sec").as_int();
    tf_warmup_sec_ = get_parameter("tf_warmup_sec").as_double();
    tf_lookup_timeout_sec_ = get_parameter("tf_lookup_timeout_sec").as_double();
    moveit_state_wait_sec_ = get_parameter("moveit_state_wait_sec").as_double();

    RCLCPP_INFO(
      get_logger(),
      "move_relative_tcp_node params: group=%s base=%s tcp=%s "
      "dx=%.6f dy=%.6f dz=%.6f dry_run=%s tf_warmup=%.1f tf_timeout=%.1f",
      planning_group_.c_str(), base_frame_.c_str(), tcp_frame_.c_str(),
      dx_, dy_, dz_, dry_run_ ? "true" : "false",
      tf_warmup_sec_, tf_lookup_timeout_sec_);
  }

  void run()
  {
    warmupTfListener();

    TransformStamped current_tf;
    if (!lookupCurrentTcpTransform(current_tf)) {
      return;
    }

    geometry_msgs::msg::PoseStamped target_pose;
    target_pose.header.frame_id = base_frame_;
    target_pose.header.stamp = now();
    target_pose.pose.position.x = current_tf.transform.translation.x + dx_;
    target_pose.pose.position.y = current_tf.transform.translation.y + dy_;
    target_pose.pose.position.z = current_tf.transform.translation.z + dz_;
    target_pose.pose.orientation = current_tf.transform.rotation;

    logCurrentAndTarget(current_tf, target_pose);

    if (!initMoveGroup()) {
      return;
    }

    const bool state_ready = checkMoveItStateReady();
    if (!state_ready && !dry_run_) {
      RCLCPP_ERROR(get_logger(), "MoveIt robot state is not ready. Robot will not move.");
      return;
    }

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "dry_run=true, skipping MoveIt planning and execution.");
      return;
    }

    move_group_->setPoseTarget(target_pose, tcp_frame_);

    RCLCPP_INFO(
      get_logger(),
      "Planning relative TCP move with group=%s planning_frame=%s pose_reference_frame=%s eef=%s",
      planning_group_.c_str(),
      move_group_->getPlanningFrame().c_str(),
      move_group_->getPoseReferenceFrame().c_str(),
      move_group_->getEndEffectorLink().c_str());

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool planned = static_cast<bool>(move_group_->plan(plan));
    if (!planned) {
      RCLCPP_ERROR(
        get_logger(),
        "Planning failed. Robot will not move. target_frame=%s target_xyz=(%.6f, %.6f, %.6f) "
        "group=%s planning_frame=%s pose_reference_frame=%s eef=%s",
        target_pose.header.frame_id.c_str(),
        target_pose.pose.position.x,
        target_pose.pose.position.y,
        target_pose.pose.position.z,
        planning_group_.c_str(),
        move_group_->getPlanningFrame().c_str(),
        move_group_->getPoseReferenceFrame().c_str(),
        move_group_->getEndEffectorLink().c_str());
      move_group_->clearPoseTargets();
      return;
    }

    RCLCPP_INFO(get_logger(), "Planning succeeded. Executing trajectory...");
    const auto result = move_group_->execute(plan);
    move_group_->clearPoseTargets();

    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Execution failed (code=%d).", result.val);
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Relative TCP move complete: dx=%.6f dy=%.6f dz=%.6f in %s frame.",
      dx_, dy_, dz_, base_frame_.c_str());
  }

private:
  using TransformStamped = geometry_msgs::msg::TransformStamped;

  void warmupTfListener()
  {
    const double warmup_sec = std::max(3.0, tf_warmup_sec_);
    RCLCPP_INFO(
      get_logger(),
      "Warming up TF listener for %.1f s. Executor must be spinning during this time.",
      warmup_sec);

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::duration<double>(warmup_sec);
    auto next_log = start;
    bool transform_seen = false;
    std::string last_error;

    while (rclcpp::ok() && std::chrono::steady_clock::now() < deadline) {
      std::string err;
      if (tf_buffer_.canTransform(
          base_frame_, tcp_frame_, tf2::TimePointZero,
          tf2::durationFromSec(0.1), &err))
      {
        transform_seen = true;
      } else {
        last_error = err;
      }

      const auto now = std::chrono::steady_clock::now();
      if (now >= next_log) {
        const auto elapsed = std::chrono::duration<double>(now - start).count();
        RCLCPP_INFO(
          get_logger(),
          "TF warmup %.1f/%.1f s: canTransform(%s <- %s)=%s%s%s",
          elapsed, warmup_sec,
          base_frame_.c_str(), tcp_frame_.c_str(),
          transform_seen ? "true" : "false",
          last_error.empty() ? "" : " last_error=",
          last_error.empty() ? "" : last_error.c_str());
        next_log = now + std::chrono::seconds(1);
      }
    }

    RCLCPP_INFO(
      get_logger(), "TF warmup complete. Transform observed during warmup: %s",
      transform_seen ? "yes" : "no");

    if (!transform_seen) {
      dumpTfFrames("TF warmup completed without seeing requested transform");
    }
  }

  bool lookupCurrentTcpTransform(TransformStamped & transform)
  {
    std::string err;
    if (!tf_buffer_.canTransform(
        base_frame_, tcp_frame_, tf2::TimePointZero,
        tf2::durationFromSec(tf_lookup_timeout_sec_), &err))
    {
      RCLCPP_ERROR(
        get_logger(),
        "canTransform failed for %s <- %s after %.1f s: %s",
        base_frame_.c_str(), tcp_frame_.c_str(), tf_lookup_timeout_sec_, err.c_str());
      dumpTfFrames("canTransform failure");
      return false;
    }

    try {
      transform = tf_buffer_.lookupTransform(
        base_frame_, tcp_frame_, tf2::TimePointZero,
        tf2::durationFromSec(tf_lookup_timeout_sec_));
      return true;
    } catch (const tf2::TransformException & e) {
      RCLCPP_ERROR(
        get_logger(), "Failed to lookup TF %s -> %s: %s",
        base_frame_.c_str(), tcp_frame_.c_str(), e.what());
      dumpTfFrames("lookupTransform exception");
      return false;
    }
  }

  void dumpTfFrames(const std::string & reason) const
  {
    const std::string yaml = tf_buffer_.allFramesAsYAML();
    const std::string frames = tf_buffer_.allFramesAsString();

    RCLCPP_ERROR(get_logger(), "TF frame dump reason: %s", reason.c_str());
    RCLCPP_ERROR(
      get_logger(),
      "Frames visible to move_relative_tcp_node as YAML:\n%s",
      yaml.empty() ? "<no frames visible yet>" : yaml.c_str());
    RCLCPP_ERROR(
      get_logger(),
      "Frames visible to move_relative_tcp_node as string:\n%s",
      frames.empty() ? "<no frames visible yet>" : frames.c_str());
  }

  void logCurrentAndTarget(
    const TransformStamped & current_tf,
    const geometry_msgs::msg::PoseStamped & target_pose) const
  {
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
      "Target %s in %s: position=(%.6f, %.6f, %.6f) "
      "orientation=(x=%.6f, y=%.6f, z=%.6f, w=%.6f)",
      tcp_frame_.c_str(), base_frame_.c_str(),
      target_pose.pose.position.x,
      target_pose.pose.position.y,
      target_pose.pose.position.z,
      target_pose.pose.orientation.x,
      target_pose.pose.orientation.y,
      target_pose.pose.orientation.z,
      target_pose.pose.orientation.w);
  }

  bool initMoveGroup()
  {
    if (!fetchRobotDescription()) {
      return false;
    }

    try {
      move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), planning_group_);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "MoveGroupInterface construction failed: %s", e.what());
      return false;
    }

    move_group_->setEndEffectorLink(tcp_frame_);
    move_group_->setPoseReferenceFrame(base_frame_);
    move_group_->setPlanningTime(planning_time_);
    move_group_->setMaxVelocityScalingFactor(0.1);
    move_group_->setMaxAccelerationScalingFactor(0.1);
    move_group_->setStartStateToCurrentState();

    RCLCPP_INFO(
      get_logger(),
      "MoveGroup ready. planning_frame=%s end_effector_link=%s pose_reference_frame=%s "
      "planning_time=%.1f velocity_scale=0.10 acceleration_scale=0.10",
      move_group_->getPlanningFrame().c_str(),
      move_group_->getEndEffectorLink().c_str(),
      move_group_->getPoseReferenceFrame().c_str(),
      planning_time_);

    return true;
  }

  bool checkMoveItStateReady()
  {
    RCLCPP_INFO(
      get_logger(),
      "Checking MoveIt CurrentStateMonitor / robot state readiness (timeout=%.1f s)...",
      moveit_state_wait_sec_);

    const auto current_state = move_group_->getCurrentState(moveit_state_wait_sec_);
    if (!current_state) {
      RCLCPP_ERROR(
        get_logger(),
        "MoveIt current robot state is not available. Check /joint_states, "
        "joint_state_broadcaster, fr3_arm_controller, and /move_group logs.");
      return false;
    }

    const auto * joint_model_group = current_state->getJointModelGroup(planning_group_);
    if (joint_model_group == nullptr) {
      RCLCPP_WARN(
        get_logger(),
        "MoveIt current state exists, but planning group '%s' was not found in robot model.",
        planning_group_.c_str());
    } else {
      RCLCPP_INFO(
        get_logger(),
        "MoveIt current robot state ready. group=%s variable_count=%u",
        planning_group_.c_str(),
        joint_model_group->getVariableCount());
    }

    const auto moveit_current_pose = move_group_->getCurrentPose(tcp_frame_);
    RCLCPP_INFO(
      get_logger(),
      "MoveIt current pose for %s in %s: position=(%.6f, %.6f, %.6f) "
      "orientation=(x=%.6f, y=%.6f, z=%.6f, w=%.6f)",
      tcp_frame_.c_str(),
      moveit_current_pose.header.frame_id.c_str(),
      moveit_current_pose.pose.position.x,
      moveit_current_pose.pose.position.y,
      moveit_current_pose.pose.position.z,
      moveit_current_pose.pose.orientation.x,
      moveit_current_pose.pose.orientation.y,
      moveit_current_pose.pose.orientation.z,
      moveit_current_pose.pose.orientation.w);

    return true;
  }

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

    const std::vector<std::string> description_params = {
      "robot_description",
      "robot_description_semantic",
      "robot_description_kinematics"
    };
    auto future = param_client->get_parameters(description_params);

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
        if (param.get_name() == "robot_description_kinematics") {
          RCLCPP_WARN(
            get_logger(),
            "'%s' is not set in /move_group. Continuing, but pose-goal planning may need "
            "kinematics.yaml if IK is required.",
            param.get_name().c_str());
          continue;
        }
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

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::string planning_group_;
  std::string base_frame_;
  std::string tcp_frame_;
  double dx_;
  double dy_;
  double dz_;
  bool dry_run_;
  double planning_time_;
  int robot_description_wait_timeout_sec_;
  double tf_warmup_sec_;
  double tf_lookup_timeout_sec_;
  double moveit_state_wait_sec_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveRelativeTcpNode>();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  std::thread spin_thread([&executor]() { executor.spin(); });
  RCLCPP_INFO(node->get_logger(), "SingleThreadedExecutor is spinning for TF and MoveIt callbacks.");

  node->run();

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
