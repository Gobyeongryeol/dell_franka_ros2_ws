#include <memory>
#include <string>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "moveit/move_group_interface/move_group_interface.h"

class FrankaVisualServoNode : public rclcpp::Node
{
public:
  FrankaVisualServoNode()
  : Node("franka_visual_servo_node")
  {
    this->declare_parameter<std::string>("planning_group", "fr3_arm");
    this->declare_parameter<std::string>("ee_link", "");
    this->declare_parameter<double>("workspace_min_x", 0.15);
    this->declare_parameter<double>("workspace_max_x", 0.80);
    this->declare_parameter<double>("workspace_min_y", -0.50);
    this->declare_parameter<double>("workspace_max_y", 0.50);
    this->declare_parameter<double>("workspace_min_z", 0.05);
    this->declare_parameter<double>("workspace_max_z", 0.80);
    this->declare_parameter<double>("fixed_z", 0.30);
    this->declare_parameter<bool>("invert_x", false);
    this->declare_parameter<bool>("invert_y", false);
    this->declare_parameter<double>("execute_threshold", 0.0005);
    this->declare_parameter<bool>("dry_run", true);

    planning_group_ = this->get_parameter("planning_group").as_string();
    ee_link_ = this->get_parameter("ee_link").as_string();

    workspace_min_x_ = this->get_parameter("workspace_min_x").as_double();
    workspace_max_x_ = this->get_parameter("workspace_max_x").as_double();
    workspace_min_y_ = this->get_parameter("workspace_min_y").as_double();
    workspace_max_y_ = this->get_parameter("workspace_max_y").as_double();
    workspace_min_z_ = this->get_parameter("workspace_min_z").as_double();
    workspace_max_z_ = this->get_parameter("workspace_max_z").as_double();

    fixed_z_ = this->get_parameter("fixed_z").as_double();
    invert_x_ = this->get_parameter("invert_x").as_bool();
    invert_y_ = this->get_parameter("invert_y").as_bool();
    execute_threshold_ = this->get_parameter("execute_threshold").as_double();
    dry_run_ = this->get_parameter("dry_run").as_bool();

    step_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
      "/visual_servo/step_cmd", 10,
      std::bind(&FrankaVisualServoNode::stepCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
      "FrankaVisualServoNode started. planning_group=%s dry_run=%s",
      planning_group_.c_str(), dry_run_ ? "true" : "false");
  }

  void initMoveGroup()
  {
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), planning_group_);

    if (!ee_link_.empty()) {
      move_group_->setEndEffectorLink(ee_link_);
    }

    move_group_->setPlanningTime(2.0);
    move_group_->setNumPlanningAttempts(5);
    move_group_->setMaxVelocityScalingFactor(0.10);
    move_group_->setMaxAccelerationScalingFactor(0.10);

    RCLCPP_INFO(this->get_logger(), "MoveGroup initialized.");
    RCLCPP_INFO(this->get_logger(), "Planning frame: %s", move_group_->getPlanningFrame().c_str());
    RCLCPP_INFO(this->get_logger(), "End effector link: %s", move_group_->getEndEffectorLink().c_str());
  }

private:
  double clamp(double v, double lo, double hi)
  {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
  }

  void stepCallback(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    if (!move_group_) {
      RCLCPP_WARN(this->get_logger(), "MoveGroup not initialized yet.");
      return;
    }

    double step_x = msg->x;
    double step_y = msg->y;

    if (invert_x_) step_x *= -1.0;
    if (invert_y_) step_y *= -1.0;

    if (std::abs(step_x) < execute_threshold_ && std::abs(step_y) < execute_threshold_) {
      return;
    }

    geometry_msgs::msg::Pose current_pose = move_group_->getCurrentPose().pose;
    geometry_msgs::msg::Pose target_pose = current_pose;

    target_pose.position.x = clamp(current_pose.position.x + step_x, workspace_min_x_, workspace_max_x_);
    target_pose.position.y = clamp(current_pose.position.y + step_y, workspace_min_y_, workspace_max_y_);
    target_pose.position.z = clamp(fixed_z_, workspace_min_z_, workspace_max_z_);

    RCLCPP_INFO(this->get_logger(),
      "Current pose: x=%.4f y=%.4f z=%.4f | step_x=%.4f step_y=%.4f | Target pose: x=%.4f y=%.4f z=%.4f",
      current_pose.position.x, current_pose.position.y, current_pose.position.z,
      step_x, step_y,
      target_pose.position.x, target_pose.position.y, target_pose.position.z);

    if (dry_run_) {
      return;
    }

    move_group_->setPoseTarget(target_pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_->plan(plan));

    if (!success) {
      RCLCPP_WARN(this->get_logger(), "Planning failed.");
      move_group_->clearPoseTargets();
      return;
    }

    auto result = move_group_->execute(plan);
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_WARN(this->get_logger(), "Execution failed.");
    } else {
      RCLCPP_INFO(this->get_logger(), "Execution success.");
    }

    move_group_->clearPoseTargets();
  }

  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr step_sub_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::string planning_group_;
  std::string ee_link_;

  double workspace_min_x_;
  double workspace_max_x_;
  double workspace_min_y_;
  double workspace_max_y_;
  double workspace_min_z_;
  double workspace_max_z_;
  double fixed_z_;

  bool invert_x_;
  bool invert_y_;
  double execute_threshold_;
  bool dry_run_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<FrankaVisualServoNode>();
  node->initMoveGroup();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
