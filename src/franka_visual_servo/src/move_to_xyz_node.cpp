#include <chrono>
#include <cmath>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "moveit/robot_state/robot_state.h"
#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/rclcpp.hpp"

class MoveToXyzNode : public rclcpp::Node
{
public:
  MoveToXyzNode()
  : Node("move_to_xyz_node")
  {
    declare_parameter<std::string>("planning_group", "fr3_arm");
    declare_parameter<std::string>("pose_reference_frame", "fr3_link0");
    declare_parameter<std::string>("end_effector_link", "fr3_hand_tcp");
    declare_parameter<double>("target_x", 0.666);
    declare_parameter<double>("target_y", 0.039);
    declare_parameter<double>("target_z", 0.634);
    declare_parameter<double>("qx", 0.999);
    declare_parameter<double>("qy", 0.030);
    declare_parameter<double>("qz", -0.041);
    declare_parameter<double>("qw", -0.006);
    declare_parameter<bool>("dry_run", false);
    declare_parameter<double>("planning_time", 10.0);
    declare_parameter<int>("robot_description_wait_timeout_sec", 30);
    declare_parameter<double>("moveit_state_wait_sec", 5.0);

    planning_group_ = get_parameter("planning_group").as_string();
    pose_reference_frame_ = get_parameter("pose_reference_frame").as_string();
    end_effector_link_ = get_parameter("end_effector_link").as_string();
    target_x_ = get_parameter("target_x").as_double();
    target_y_ = get_parameter("target_y").as_double();
    target_z_ = get_parameter("target_z").as_double();
    qx_ = get_parameter("qx").as_double();
    qy_ = get_parameter("qy").as_double();
    qz_ = get_parameter("qz").as_double();
    qw_ = get_parameter("qw").as_double();
    dry_run_ = get_parameter("dry_run").as_bool();
    planning_time_ = get_parameter("planning_time").as_double();
    robot_description_wait_timeout_sec_ =
      get_parameter("robot_description_wait_timeout_sec").as_int();
    moveit_state_wait_sec_ = get_parameter("moveit_state_wait_sec").as_double();

    RCLCPP_INFO(
      get_logger(),
      "move_to_xyz_node params: group=%s reference_frame=%s eef=%s "
      "target=(%.6f, %.6f, %.6f) q=(%.6f, %.6f, %.6f, %.6f) dry_run=%s",
      planning_group_.c_str(),
      pose_reference_frame_.c_str(),
      end_effector_link_.c_str(),
      target_x_, target_y_, target_z_,
      qx_, qy_, qz_, qw_,
      dry_run_ ? "true" : "false");
  }

  void run()
  {
    const auto target_pose = makeTargetPose();
    logTargetPose(target_pose);

    if (!initMoveGroup()) {
      return;
    }

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "dry_run=true, skipping planning and execution.");
      return;
    }

    if (!checkMoveItStateReady()) {
      RCLCPP_ERROR(get_logger(), "MoveIt robot state is not ready. Robot will not move.");
      return;
    }

    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(target_pose, end_effector_link_);

    RCLCPP_INFO(
      get_logger(),
      "Planning absolute pose move. group=%s planning_frame=%s pose_reference_frame=%s eef=%s",
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
        "target_q=(%.6f, %.6f, %.6f, %.6f)",
        target_pose.header.frame_id.c_str(),
        target_pose.pose.position.x,
        target_pose.pose.position.y,
        target_pose.pose.position.z,
        target_pose.pose.orientation.x,
        target_pose.pose.orientation.y,
        target_pose.pose.orientation.z,
        target_pose.pose.orientation.w);
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

    RCLCPP_INFO(get_logger(), "Move to absolute XYZ pose complete.");
  }

private:
  geometry_msgs::msg::PoseStamped makeTargetPose()
  {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.frame_id = pose_reference_frame_;
    pose.header.stamp = now();
    pose.pose.position.x = target_x_;
    pose.pose.position.y = target_y_;
    pose.pose.position.z = target_z_;
    pose.pose.orientation.x = qx_;
    pose.pose.orientation.y = qy_;
    pose.pose.orientation.z = qz_;
    pose.pose.orientation.w = qw_;
    return pose;
  }

  void logTargetPose(const geometry_msgs::msg::PoseStamped & pose) const
  {
    const double q_norm =
      std::sqrt(qx_ * qx_ + qy_ * qy_ + qz_ * qz_ + qw_ * qw_);

    RCLCPP_INFO(
      get_logger(),
      "Target pose in %s: position=(%.6f, %.6f, %.6f) "
      "orientation=(x=%.6f, y=%.6f, z=%.6f, w=%.6f), quaternion_norm=%.6f",
      pose.header.frame_id.c_str(),
      pose.pose.position.x,
      pose.pose.position.y,
      pose.pose.position.z,
      pose.pose.orientation.x,
      pose.pose.orientation.y,
      pose.pose.orientation.z,
      pose.pose.orientation.w,
      q_norm);
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

    move_group_->setPoseReferenceFrame(pose_reference_frame_);
    move_group_->setEndEffectorLink(end_effector_link_);
    move_group_->setPlanningTime(planning_time_);
    move_group_->setMaxVelocityScalingFactor(0.1);
    move_group_->setMaxAccelerationScalingFactor(0.1);

    RCLCPP_INFO(
      get_logger(),
      "MoveGroup ready. planning_frame=%s pose_reference_frame=%s end_effector_link=%s "
      "planning_time=%.1f velocity_scale=0.10 acceleration_scale=0.10",
      move_group_->getPlanningFrame().c_str(),
      move_group_->getPoseReferenceFrame().c_str(),
      move_group_->getEndEffectorLink().c_str(),
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
        "MoveIt current robot state is not available. Check /joint_states and /move_group.");
      return false;
    }

    const auto * joint_model_group = current_state->getJointModelGroup(planning_group_);
    if (joint_model_group == nullptr) {
      RCLCPP_WARN(
        get_logger(),
        "Current state exists, but planning group '%s' was not found in robot model.",
        planning_group_.c_str());
    } else {
      RCLCPP_INFO(
        get_logger(),
        "MoveIt current robot state ready. group=%s variable_count=%u",
        planning_group_.c_str(),
        joint_model_group->getVariableCount());
    }

    return true;
  }

  bool fetchRobotDescription()
  {
    RCLCPP_INFO(
      get_logger(),
      "Fetching robot_description parameters from /move_group (timeout=%d s)...",
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
          get_logger(),
          "Waiting for /move_group parameter service... (%d/%d s)",
          i / 2,
          robot_description_wait_timeout_sec_);
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
        RCLCPP_ERROR(get_logger(), "Timed out waiting for robot description parameters.");
        return false;
      }
    }

    bool ok = true;
    for (const auto & param : future.get()) {
      if (param.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
        if (param.get_name() == "robot_description_kinematics") {
          RCLCPP_WARN(
            get_logger(),
            "'%s' is not set in /move_group. Continuing, but pose planning may need IK config.",
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
        get_logger(),
        "Loaded '%s' (%zu chars) from /move_group.",
        param.get_name().c_str(),
        param.value_to_string().size());
    }

    return ok;
  }

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::string planning_group_;
  std::string pose_reference_frame_;
  std::string end_effector_link_;
  double target_x_;
  double target_y_;
  double target_z_;
  double qx_;
  double qy_;
  double qz_;
  double qw_;
  bool dry_run_;
  double planning_time_;
  int robot_description_wait_timeout_sec_;
  double moveit_state_wait_sec_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveToXyzNode>();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->run();

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
