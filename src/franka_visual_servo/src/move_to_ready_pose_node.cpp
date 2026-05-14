#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "moveit/move_group_interface/move_group_interface.h"

class MoveToReadyPoseNode : public rclcpp::Node
{
public:
  MoveToReadyPoseNode()
  : Node("move_to_ready_pose_node")
  {
    declare_parameter<std::string>("planning_group",      "fr3_arm");
    declare_parameter<std::string>("eef_link",            "fr3_hand_tcp");
    declare_parameter<bool>("dry_run",                    false);
    declare_parameter<double>("velocity_scaling",         0.1);
    declare_parameter<double>("acceleration_scaling",     0.1);
    declare_parameter<double>("ready_joint1",             0.0);
    declare_parameter<double>("ready_joint2",            -0.785);
    declare_parameter<double>("ready_joint3",             0.0);
    declare_parameter<double>("ready_joint4",            -2.356);
    declare_parameter<double>("ready_joint5",             0.0);
    declare_parameter<double>("ready_joint6",             1.571);
    declare_parameter<double>("ready_joint7",             0.785);

    planning_group_       = get_parameter("planning_group").as_string();
    eef_link_             = get_parameter("eef_link").as_string();
    dry_run_              = get_parameter("dry_run").as_bool();
    velocity_scaling_     = get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = get_parameter("acceleration_scaling").as_double();
    ready_joint1_         = get_parameter("ready_joint1").as_double();
    ready_joint2_         = get_parameter("ready_joint2").as_double();
    ready_joint3_         = get_parameter("ready_joint3").as_double();
    ready_joint4_         = get_parameter("ready_joint4").as_double();
    ready_joint5_         = get_parameter("ready_joint5").as_double();
    ready_joint6_         = get_parameter("ready_joint6").as_double();
    ready_joint7_         = get_parameter("ready_joint7").as_double();

    RCLCPP_INFO(get_logger(),
      "Params: group=%s eef=%s dry_run=%s vel=%.2f acc=%.2f",
      planning_group_.c_str(), eef_link_.c_str(),
      dry_run_ ? "true" : "false",
      velocity_scaling_, acceleration_scaling_);
    RCLCPP_INFO(get_logger(),
      "Ready joints: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
      ready_joint1_, ready_joint2_, ready_joint3_, ready_joint4_,
      ready_joint5_, ready_joint6_, ready_joint7_);
  }

  bool fetchRobotDescription()
  {
    RCLCPP_INFO(get_logger(), "Fetching robot_description from /move_group ...");
    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(
      shared_from_this(), "/move_group");

    for (int i = 0; i < 20 && rclcpp::ok(); ++i) {
      if (param_client->service_is_ready()) break;
      if (i == 19) {
        RCLCPP_ERROR(get_logger(),
          "/move_group parameter service not ready after 10 s. Aborting.");
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    auto future = param_client->get_parameters(
      {"robot_description", "robot_description_semantic"});

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
      if (!rclcpp::ok() || std::chrono::steady_clock::now() > deadline) {
        RCLCPP_ERROR(get_logger(),
          "Timed out waiting for robot description from /move_group. Aborting.");
        return false;
      }
    }

    bool all_ok = true;
    for (auto & p : future.get()) {
      if (p.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
        RCLCPP_ERROR(get_logger(), "'%s' not set in /move_group.", p.get_name().c_str());
        all_ok = false;
        continue;
      }
      try {
        declare_parameter(p.get_name(), p.get_parameter_value());
      } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
        set_parameter(p);
      }
      RCLCPP_INFO(get_logger(), "Loaded '%s' (%zu chars) from /move_group.",
        p.get_name().c_str(), p.as_string().size());
    }
    return all_ok;
  }

  void initAndMove()
  {
    if (!fetchRobotDescription()) return;

    try {
      move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), planning_group_);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(),
        "MoveGroupInterface construction failed: %s. Aborting.", e.what());
      return;
    }

    move_group_->setEndEffectorLink(eef_link_);
    move_group_->setPlanningTime(10.0);
    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);

    RCLCPP_INFO(get_logger(), "MoveGroup created. planning_frame=%s  eef=%s",
      move_group_->getPlanningFrame().c_str(),
      move_group_->getEndEffectorLink().c_str());

    RCLCPP_INFO(get_logger(), "Waiting 2s for CurrentStateMonitor...");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::map<std::string, double> joint_values;
    joint_values["fr3_joint1"] = ready_joint1_;
    joint_values["fr3_joint2"] = ready_joint2_;
    joint_values["fr3_joint3"] = ready_joint3_;
    joint_values["fr3_joint4"] = ready_joint4_;
    joint_values["fr3_joint5"] = ready_joint5_;
    joint_values["fr3_joint6"] = ready_joint6_;
    joint_values["fr3_joint7"] = ready_joint7_;

    move_group_->setJointValueTarget(joint_values);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_->plan(plan));

    if (!success) {
      RCLCPP_ERROR(get_logger(), "Planning to ready pose failed.");
      return;
    }

    RCLCPP_INFO(get_logger(), "Planning succeeded.");

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "[dry_run] Skipping execute.");
      return;
    }

    auto result = move_group_->execute(plan);
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Execution failed (code=%d).", result.val);
      return;
    }

    RCLCPP_INFO(get_logger(), "Moved to ready pose.");
  }

private:
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::string planning_group_;
  std::string eef_link_;
  bool        dry_run_;
  double      velocity_scaling_;
  double      acceleration_scaling_;
  double      ready_joint1_, ready_joint2_, ready_joint3_, ready_joint4_;
  double      ready_joint5_, ready_joint6_, ready_joint7_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveToReadyPoseNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread spin_thread([&executor]() { executor.spin(); });

  node->initAndMove();

  rclcpp::shutdown();
  spin_thread.join();
  return 0;
}
