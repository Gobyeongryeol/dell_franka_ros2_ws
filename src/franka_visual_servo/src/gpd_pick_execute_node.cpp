#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "gpd_ros2_msgs/msg/grasp_config.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "franka_msgs/action/move.hpp"
#include "franka_msgs/action/grasp.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "moveit_msgs/msg/robot_trajectory.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

using GraspConfig  = gpd_ros2_msgs::msg::GraspConfig;
using FrankaMove   = franka_msgs::action::Move;
using FrankaGrasp  = franka_msgs::action::Grasp;

class GpdPickExecuteNode : public rclcpp::Node
{
public:
  GpdPickExecuteNode()
  : Node("gpd_pick_execute_node"),
    has_grasp_(false), processing_(false), done_(false)
  {
    // ── params ─────────────────────────────────────────────────────────────
    declare_parameter<bool>("dry_run",                    true);
    declare_parameter<bool>("once",                       true);
    declare_parameter<bool>("hover_only",                 true);
    declare_parameter<double>("pregrasp_z_offset",        0.10);
    declare_parameter<double>("approach_z_offset",        0.03);
    declare_parameter<double>("lift_z_offset",            0.10);
    declare_parameter<double>("min_safe_z",               0.05);
    declare_parameter<double>("safe_z",                   0.45);
    declare_parameter<bool>("force_vertical_pregrasp",    true);
    declare_parameter<bool>("force_vertical_approach",    true);
    declare_parameter<bool>("use_gpd_approach_vector",    false);
    declare_parameter<bool>("use_gpd_orientation",        false);
    declare_parameter<std::string>("planning_group",      "fr3_arm");
    declare_parameter<std::string>("end_effector_link",   "fr3_hand_tcp");
    declare_parameter<bool>("use_fixed_orientation",      false);
    declare_parameter<double>("fixed_qx",                 1.0);
    declare_parameter<double>("fixed_qy",                 0.0);
    declare_parameter<double>("fixed_qz",                 0.0);
    declare_parameter<double>("fixed_qw",                 0.0);
    declare_parameter<double>("gripper_width_open",       0.08);
    declare_parameter<double>("gripper_width_close",      0.025);
    declare_parameter<double>("gripper_force",            20.0);
    declare_parameter<double>("velocity_scaling",         0.10);
    declare_parameter<double>("acceleration_scaling",     0.10);
    declare_parameter<double>("eef_step",                 0.005);
    declare_parameter<double>("jump_threshold",           0.0);
    declare_parameter<double>("min_fraction",             0.95);
    declare_parameter<std::string>("grasp_topic",         "/best_gpd_grasp_fr3");
    declare_parameter<std::string>("gripper_action_ns",   "/fr3/franka_gripper");
    declare_parameter<int>("robot_description_wait_timeout_sec", 30);
    declare_parameter<double>("wait_for_grasp_timeout_sec", 30.0);
    declare_parameter<bool>("use_transient_local_qos", false);
    declare_parameter<bool>("print_wait_hint", true);
    declare_parameter<bool>("fail_if_no_grasp", true);
    declare_parameter<double>("min_x", 0.10);
    declare_parameter<double>("max_x", 0.75);
    declare_parameter<double>("min_y", -0.50);
    declare_parameter<double>("max_y", 0.50);
    declare_parameter<double>("min_z", 0.05);
    declare_parameter<double>("max_z", 0.60);

    dry_run_              = get_parameter("dry_run").as_bool();
    once_                 = get_parameter("once").as_bool();
    hover_only_           = get_parameter("hover_only").as_bool();
    pregrasp_z_offset_    = get_parameter("pregrasp_z_offset").as_double();
    approach_z_offset_    = get_parameter("approach_z_offset").as_double();
    lift_z_offset_        = get_parameter("lift_z_offset").as_double();
    min_safe_z_           = get_parameter("min_safe_z").as_double();
    safe_z_               = get_parameter("safe_z").as_double();
    force_vertical_pregrasp_ = get_parameter("force_vertical_pregrasp").as_bool();
    force_vertical_approach_ = get_parameter("force_vertical_approach").as_bool();
    use_gpd_approach_vector_ = get_parameter("use_gpd_approach_vector").as_bool();
    use_gpd_orientation_     = get_parameter("use_gpd_orientation").as_bool();
    planning_group_       = get_parameter("planning_group").as_string();
    eef_link_             = get_parameter("end_effector_link").as_string();
    use_fixed_orientation_= get_parameter("use_fixed_orientation").as_bool();
    fixed_qx_             = get_parameter("fixed_qx").as_double();
    fixed_qy_             = get_parameter("fixed_qy").as_double();
    fixed_qz_             = get_parameter("fixed_qz").as_double();
    fixed_qw_             = get_parameter("fixed_qw").as_double();
    gripper_width_open_   = get_parameter("gripper_width_open").as_double();
    gripper_width_close_  = get_parameter("gripper_width_close").as_double();
    gripper_force_        = get_parameter("gripper_force").as_double();
    velocity_scaling_     = get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = get_parameter("acceleration_scaling").as_double();
    eef_step_             = get_parameter("eef_step").as_double();
    jump_threshold_       = get_parameter("jump_threshold").as_double();
    min_fraction_         = get_parameter("min_fraction").as_double();
    grasp_topic_          = get_parameter("grasp_topic").as_string();
    gripper_action_ns_    = get_parameter("gripper_action_ns").as_string();
    robot_desc_wait_sec_  = get_parameter("robot_description_wait_timeout_sec").as_int();
    wait_for_grasp_timeout_sec_ = get_parameter("wait_for_grasp_timeout_sec").as_double();
    use_transient_local_qos_    = get_parameter("use_transient_local_qos").as_bool();
    print_wait_hint_            = get_parameter("print_wait_hint").as_bool();
    fail_if_no_grasp_           = get_parameter("fail_if_no_grasp").as_bool();
    min_x_ = get_parameter("min_x").as_double();
    max_x_ = get_parameter("max_x").as_double();
    min_y_ = get_parameter("min_y").as_double();
    max_y_ = get_parameter("max_y").as_double();
    min_z_ = get_parameter("min_z").as_double();
    max_z_ = get_parameter("max_z").as_double();

    RCLCPP_INFO(get_logger(),
      "[GPD pick] dry_run=%s  once=%s  hover_only=%s\n"
      "  pregrasp_z=%.3f  approach_z=%.3f  lift_z=%.3f  min_safe_z=%.3f  safe_z=%.3f\n"
      "  force_vertical_pregrasp=%s  force_vertical_approach=%s  use_gpd_approach_vector=%s  use_gpd_orientation=%s\n"
      "  group=%s  eef=%s  vel=%.2f  acc=%.2f\n"
      "  gripper_open=%.3f  gripper_close=%.3f  force=%.1f\n"
      "  workspace x=[%.2f, %.2f] y=[%.2f, %.2f] z=[%.2f, %.2f]\n"
      "  grasp_topic=%s  gripper_ns=%s\n"
      "  wait_for_grasp_timeout=%.1f  transient_local_qos=%s  print_wait_hint=%s  fail_if_no_grasp=%s",
      dry_run_ ? "true" : "false",
      once_    ? "true" : "false",
      hover_only_ ? "true" : "false",
      pregrasp_z_offset_, approach_z_offset_, lift_z_offset_, min_safe_z_, safe_z_,
      force_vertical_pregrasp_ ? "true" : "false",
      force_vertical_approach_ ? "true" : "false",
      use_gpd_approach_vector_ ? "true" : "false",
      use_gpd_orientation_ ? "true" : "false",
      planning_group_.c_str(), eef_link_.c_str(),
      velocity_scaling_, acceleration_scaling_,
      gripper_width_open_, gripper_width_close_, gripper_force_,
      min_x_, max_x_, min_y_, max_y_, min_z_, max_z_,
      grasp_topic_.c_str(), gripper_action_ns_.c_str(),
      wait_for_grasp_timeout_sec_,
      use_transient_local_qos_ ? "true" : "false",
      print_wait_hint_ ? "true" : "false",
      fail_if_no_grasp_ ? "true" : "false");
  }

  // Called after executor is spinning in a separate thread.
  void initMoveGroup()
  {
    if (!fetchRobotDescription()) return;

    try {
      move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), planning_group_);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(),
        "[GPD pick] MoveGroupInterface failed: %s. Aborting.", e.what());
      return;
    }

    move_group_->setEndEffectorLink(eef_link_);
    move_group_->setPoseReferenceFrame("fr3_link0");
    move_group_->setPlanningTime(10.0);
    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);

    RCLCPP_INFO(get_logger(),
      "[GPD pick] MoveGroup ready. frame=%s  eef=%s",
      move_group_->getPlanningFrame().c_str(),
      move_group_->getEndEffectorLink().c_str());

    // Create gripper action clients
    gripper_move_client_ = rclcpp_action::create_client<FrankaMove>(
      shared_from_this(), gripper_action_ns_ + "/move");
    gripper_grasp_client_ = rclcpp_action::create_client<FrankaGrasp>(
      shared_from_this(), gripper_action_ns_ + "/grasp");

    RCLCPP_INFO(get_logger(), "Waiting 2s for CurrentStateMonitor...");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    const auto sub_qos = makeGraspQoS();
    sub_ = create_subscription<GraspConfig>(
      grasp_topic_, sub_qos,
      std::bind(&GpdPickExecuteNode::graspCallback, this, std::placeholders::_1));
    grasp_subscription_ready_ = true;
    grasp_wait_started_ = std::chrono::steady_clock::now();
    grasp_wait_timeout_reported_ = false;

    RCLCPP_INFO(get_logger(),
      "[GPD pick] Subscribed to %s. Waiting for grasp...", grasp_topic_.c_str());
    if (print_wait_hint_) {
      RCLCPP_INFO(
        get_logger(),
        "[GPD pick] If no grasp arrives, run gpd_client_node again to publish a new %s.",
        grasp_topic_.c_str());
    }
  }

  void startWorker()
  {
    worker_thread_ = std::thread(&GpdPickExecuteNode::workerLoop, this);
  }

  ~GpdPickExecuteNode()
  {
    if (worker_thread_.joinable()) worker_thread_.join();
  }

private:
  // ── Subscription callback ────────────────────────────────────────────────
  void graspCallback(const GraspConfig::SharedPtr msg)
  {
    if (once_ && done_) return;
    if (processing_) {
      RCLCPP_WARN(get_logger(), "[GPD pick] Already processing a grasp. Ignoring.");
      return;
    }
    {
      std::lock_guard<std::mutex> lock(grasp_mutex_);
      latest_grasp_ = *msg;
      has_grasp_    = true;
    }
    grasp_wait_timeout_reported_ = false;
    RCLCPP_INFO(get_logger(),
      "[GPD pick] received grasp pose  pos=(%.3f, %.3f, %.3f)  score=%.4f",
      msg->position.x, msg->position.y, msg->position.z, msg->score.data);
  }

  // ── Worker thread ────────────────────────────────────────────────────────
  void workerLoop()
  {
    while (rclcpp::ok()) {
      if (once_ && done_) break;

      if (!checkGraspWaitTimeout()) {
        break;
      }

      GraspConfig g;
      bool got = false;
      {
        std::lock_guard<std::mutex> lock(grasp_mutex_);
        if (has_grasp_ && !processing_) {
          g           = latest_grasp_;
          has_grasp_  = false;
          processing_ = true;
          got         = true;
        }
      }

      if (got) {
        executePick(g);
        processing_ = false;
        grasp_wait_started_ = std::chrono::steady_clock::now();
        grasp_wait_timeout_reported_ = false;
        if (once_) {
          done_ = true;
          RCLCPP_INFO(get_logger(), "[GPD pick] once=true — will not pick again.");
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  rclcpp::QoS makeGraspQoS() const
  {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    if (use_transient_local_qos_) {
      qos.transient_local();
    }
    return qos;
  }

  bool checkGraspWaitTimeout()
  {
    if (!grasp_subscription_ready_) {
      return true;
    }

    bool has_pending_grasp = false;
    {
      std::lock_guard<std::mutex> lock(grasp_mutex_);
      has_pending_grasp = has_grasp_;
    }

    if (has_pending_grasp || processing_ || wait_for_grasp_timeout_sec_ <= 0.0) {
      return true;
    }

    const auto now_tp = std::chrono::steady_clock::now();
    const double elapsed_sec =
      std::chrono::duration<double>(now_tp - grasp_wait_started_).count();
    if (elapsed_sec < wait_for_grasp_timeout_sec_) {
      return true;
    }

    if (!grasp_wait_timeout_reported_) {
      grasp_wait_timeout_reported_ = true;
      RCLCPP_ERROR(
        get_logger(),
        "[GPD pick] ERROR: No grasp received within %.1f sec.",
        wait_for_grasp_timeout_sec_);
      RCLCPP_ERROR(get_logger(), "[GPD pick] Check:");
      RCLCPP_ERROR(get_logger(), "  - gpd_grasp_tf_node is running");
      RCLCPP_ERROR(get_logger(), "  - gpd_client_node published /best_gpd_grasp");
      RCLCPP_ERROR(get_logger(), "  - %s is being published", grasp_topic_.c_str());
      RCLCPP_ERROR(get_logger(), "  - run: ros2 topic echo %s", grasp_topic_.c_str());
      RCLCPP_ERROR(get_logger(), "  - then rerun gpd_client_node");
    }

    if (fail_if_no_grasp_) {
      RCLCPP_ERROR(
        get_logger(),
        "[GPD pick] fail_if_no_grasp=true. Shutting down without motion.");
      done_ = true;
      rclcpp::shutdown();
      return false;
    }

    RCLCPP_WARN(
      get_logger(),
      "[GPD pick] fail_if_no_grasp=false. Continuing to wait for grasp...");
    grasp_wait_started_ = now_tp;
    grasp_wait_timeout_reported_ = false;
    return true;
  }

  // ── Quaternion from GPD orientation vectors ──────────────────────────────
  // GPD convention: axis=x (finger closing), binormal=y, approach=z (into object)
  geometry_msgs::msg::Quaternion graspQuaternion(const GraspConfig & g)
  {
    if (use_fixed_orientation_) {
      geometry_msgs::msg::Quaternion q;
      q.x = fixed_qx_; q.y = fixed_qy_; q.z = fixed_qz_; q.w = fixed_qw_;
      return q;
    }

    tf2::Vector3 ax(g.axis.x,     g.axis.y,     g.axis.z);
    tf2::Vector3 bi(g.binormal.x, g.binormal.y, g.binormal.z);
    tf2::Vector3 ap(g.approach.x, g.approach.y, g.approach.z);
    ax.normalize(); bi.normalize(); ap.normalize();

    // Rotation matrix: columns are axis (x), binormal (y), approach (z)
    tf2::Matrix3x3 R(
      ax.x(), bi.x(), ap.x(),
      ax.y(), bi.y(), ap.y(),
      ax.z(), bi.z(), ap.z()
    );
    tf2::Quaternion q;
    R.getRotation(q);
    q.normalize();

    geometry_msgs::msg::Quaternion qm;
    qm.x = q.x(); qm.y = q.y(); qm.z = q.z(); qm.w = q.w();
    return qm;
  }

  // ── Build a geometry_msgs::Pose ──────────────────────────────────────────
  geometry_msgs::msg::Pose makePose(double x, double y, double z,
                                    const geometry_msgs::msg::Quaternion & q)
  {
    geometry_msgs::msg::Pose p;
    p.position.x = x; p.position.y = y; p.position.z = z;
    p.orientation = q;
    return p;
  }

  geometry_msgs::msg::Quaternion selectGoalOrientation(
    const geometry_msgs::msg::Pose & current_pose,
    const GraspConfig & g)
  {
    (void)g;
    if (use_fixed_orientation_) {
      geometry_msgs::msg::Quaternion q;
      q.x = fixed_qx_;
      q.y = fixed_qy_;
      q.z = fixed_qz_;
      q.w = fixed_qw_;
      return q;
    }

    return current_pose.orientation;
  }

  bool poseInWorkspace(
    const geometry_msgs::msg::Pose & pose,
    const std::string & pose_name) const
  {
    const bool in_x = pose.position.x >= min_x_ && pose.position.x <= max_x_;
    const bool in_y = pose.position.y >= min_y_ && pose.position.y <= max_y_;
    const bool in_z = pose.position.z >= min_z_ && pose.position.z <= max_z_;
    if (in_x && in_y && in_z) {
      return true;
    }

    RCLCPP_ERROR(
      get_logger(),
      "[GPD pick] %s pose outside workspace: pos=(%.3f, %.3f, %.3f) "
      "workspace x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f]",
      pose_name.c_str(),
      pose.position.x, pose.position.y, pose.position.z,
      min_x_, max_x_, min_y_, max_y_, min_z_, max_z_);
    return false;
  }

  // ── Plan and execute a PTP pose goal ────────────────────────────────────
  bool moveTopose(const geometry_msgs::msg::Pose & pose, const std::string & phase)
  {
    move_group_->setPoseTarget(pose, eef_link_);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool ok = static_cast<bool>(move_group_->plan(plan));

    if (!ok) {
      RCLCPP_ERROR(get_logger(), "[GPD pick] [%s] Planning failed.", phase.c_str());
      return false;
    }

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "[GPD pick] [%s] dry_run=true, no motion.", phase.c_str());
      return true;
    }

    auto result = move_group_->execute(plan);
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(),
        "[GPD pick] [%s] Execute failed (code=%d).", phase.c_str(), result.val);
      return false;
    }
    RCLCPP_INFO(get_logger(), "[GPD pick] [%s] done.", phase.c_str());
    return true;
  }

  // ── Cartesian path motion ────────────────────────────────────────────────
  bool cartesianMove(const std::vector<geometry_msgs::msg::Pose> & waypoints,
                     const std::string & phase)
  {
    moveit_msgs::msg::RobotTrajectory traj;
    double fraction = move_group_->computeCartesianPath(
      waypoints, eef_step_, jump_threshold_, traj);

    RCLCPP_INFO(get_logger(),
      "[GPD pick] [%s] Cartesian fraction: %.3f", phase.c_str(), fraction);

    if (fraction < min_fraction_) {
      RCLCPP_ERROR(get_logger(),
        "[GPD pick] [%s] fraction %.3f < %.2f — aborting.",
        phase.c_str(), fraction, min_fraction_);
      return false;
    }

    if (dry_run_) {
      RCLCPP_INFO(get_logger(),
        "[GPD pick] [%s] dry_run=true, no motion.", phase.c_str());
      return true;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = traj;
    auto result = move_group_->execute(plan);
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(),
        "[GPD pick] [%s] Execute failed (code=%d).", phase.c_str(), result.val);
      return false;
    }
    RCLCPP_INFO(get_logger(), "[GPD pick] [%s] done.", phase.c_str());
    return true;
  }

  // ── Gripper open (Move action) ───────────────────────────────────────────
  bool gripperOpen()
  {
    if (dry_run_) {
      RCLCPP_INFO(get_logger(),
        "[GPD pick] dry_run=true — skipping gripper open (width=%.3f).",
        gripper_width_open_);
      return true;
    }

    if (!gripper_move_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(),
        "[GPD pick] Gripper move action server not available.");
      return false;
    }

    FrankaMove::Goal goal;
    goal.width = gripper_width_open_;
    goal.speed = 0.1;

    auto send_goal_opts = rclcpp_action::Client<FrankaMove>::SendGoalOptions();
    auto future_handle  = gripper_move_client_->async_send_goal(goal, send_goal_opts);

    if (future_handle.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "[GPD pick] Gripper open: goal send timeout.");
      return false;
    }
    auto handle = future_handle.get();
    if (!handle) {
      RCLCPP_ERROR(get_logger(), "[GPD pick] Gripper open: goal rejected.");
      return false;
    }

    auto future_result = gripper_move_client_->async_get_result(handle);
    if (future_result.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "[GPD pick] Gripper open: result timeout.");
      return false;
    }

    bool success = future_result.get().result->success;
    if (!success)
      RCLCPP_WARN(get_logger(), "[GPD pick] Gripper open reported failure.");
    else
      RCLCPP_INFO(get_logger(), "[GPD pick] Gripper opened (width=%.3f).", gripper_width_open_);
    return success;
  }

  // ── Gripper close (Grasp action) ─────────────────────────────────────────
  bool gripperClose()
  {
    if (dry_run_) {
      RCLCPP_INFO(get_logger(),
        "[GPD pick] dry_run=true — skipping gripper close (width=%.3f force=%.1f).",
        gripper_width_close_, gripper_force_);
      return true;
    }

    if (!gripper_grasp_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(get_logger(),
        "[GPD pick] Gripper grasp action server not available.");
      return false;
    }

    FrankaGrasp::Goal goal;
    goal.width       = gripper_width_close_;
    goal.speed       = 0.05;
    goal.force       = gripper_force_;
    goal.epsilon.inner = 0.005;
    goal.epsilon.outer = 0.005;

    auto send_goal_opts = rclcpp_action::Client<FrankaGrasp>::SendGoalOptions();
    auto future_handle  = gripper_grasp_client_->async_send_goal(goal, send_goal_opts);

    if (future_handle.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "[GPD pick] Gripper close: goal send timeout.");
      return false;
    }
    auto handle = future_handle.get();
    if (!handle) {
      RCLCPP_ERROR(get_logger(), "[GPD pick] Gripper close: goal rejected.");
      return false;
    }

    auto future_result = gripper_grasp_client_->async_get_result(handle);
    if (future_result.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "[GPD pick] Gripper close: result timeout.");
      return false;
    }

    bool success = future_result.get().result->success;
    if (!success)
      RCLCPP_WARN(get_logger(), "[GPD pick] Gripper grasp reported failure (object may be missing).");
    else
      RCLCPP_INFO(get_logger(), "[GPD pick] Gripper closed (width=%.3f).", gripper_width_close_);
    return true;  // continue lift even if grasp reports failure (object may be held)
  }

  // ── Main pick sequence ───────────────────────────────────────────────────
  void executePick(const GraspConfig & g)
  {
    // Safety: check EEF pose is initialized
    geometry_msgs::msg::Pose current = move_group_->getCurrentPose(eef_link_).pose;
    if (std::abs(current.position.x) < 1e-6 &&
        std::abs(current.position.y) < 1e-6 &&
        std::abs(current.position.z) < 1e-6)
    {
      RCLCPP_ERROR(get_logger(),
        "[GPD pick] EEF pose is (0,0,0) — joint_states not received yet. Aborting.");
      return;
    }

    // Grasp position
    double gx = g.position.x;
    double gy = g.position.y;
    double gz = g.position.z;

    RCLCPP_INFO(
      get_logger(),
      "[GPD pick] force_vertical_pregrasp=%s",
      force_vertical_pregrasp_ ? "true" : "false");
    RCLCPP_INFO(
      get_logger(),
      "[GPD pick] force_vertical_approach=%s",
      force_vertical_approach_ ? "true" : "false");
    if (!use_gpd_approach_vector_) {
      RCLCPP_INFO(get_logger(), "[GPD pick] ignoring GPD approach vector");
    }
    RCLCPP_INFO(
      get_logger(),
      "[GPD pick] raw grasp pose       (%.3f, %.3f, %.3f)",
      gx, gy, gz);

    // Safety: clamp z
    if (gz < min_safe_z_) {
      RCLCPP_WARN(get_logger(),
        "[GPD pick] Grasp z=%.4f < min_safe_z=%.4f -> clamped.", gz, min_safe_z_);
      gz = min_safe_z_;
    }

    double prx = gx;
    double pry = gy;
    double prz = gz + pregrasp_z_offset_;
    if (prz > safe_z_) {
      RCLCPP_WARN(
        get_logger(),
        "[GPD pick] pregrasp z=%.4f > safe_z=%.4f -> clamped.",
        prz, safe_z_);
      prz = safe_z_;
    }

    double grx = gx;
    double gry = gy;
    double grz = gz + approach_z_offset_;
    if (grz < min_safe_z_) {
      RCLCPP_WARN(
        get_logger(),
        "[GPD pick] grasp_goal z=%.4f < min_safe_z=%.4f -> clamped.",
        grz, min_safe_z_);
      grz = min_safe_z_;
    }

    // lift: straight up from grasp position
    double lx = grx;
    double ly = gry;
    double lz = grz + lift_z_offset_;

    const auto q = selectGoalOrientation(current, g);

    geometry_msgs::msg::Pose pregrasp_pose = makePose(prx, pry, prz, q);
    geometry_msgs::msg::Pose grasp_pose    = makePose(grx, gry, grz, q);
    geometry_msgs::msg::Pose lift_pose     = makePose(lx,  ly,  lz,  q);

    if (!poseInWorkspace(pregrasp_pose, "pregrasp")) return;
    if (!poseInWorkspace(grasp_pose, "grasp_goal")) return;
    if (!poseInWorkspace(lift_pose, "lift")) return;

    RCLCPP_INFO(get_logger(),
      "[GPD pick] vertical pregrasp pose      (%.3f, %.3f, %.3f)",
      prx, pry, prz);
    RCLCPP_INFO(get_logger(),
      "[GPD pick] vertical approach/grasp pose (%.3f, %.3f, %.3f)",
      grx, gry, grz);
    RCLCPP_INFO(get_logger(),
      "[GPD pick] lift pose                   (%.3f, %.3f, %.3f)",
      lx, ly, lz);

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "[GPD pick] dry_run=true, no motion");
      return;
    }

    // Phase 1: pregrasp
    RCLCPP_INFO(get_logger(), "[GPD pick] moving to pregrasp");
    if (!moveTopose(pregrasp_pose, "pregrasp")) return;

    if (hover_only_) {
      RCLCPP_INFO(get_logger(), "[GPD pick] hover_only=true, stopping at pregrasp");
      return;
    }

    // Phase 2: open gripper
    RCLCPP_INFO(get_logger(), "[GPD pick] opening gripper");
    if (!gripperOpen()) return;

    // Phase 3: approach (Cartesian)
    RCLCPP_INFO(get_logger(), "[GPD pick] approaching grasp pose");
    if (!cartesianMove({grasp_pose}, "approach")) return;

    // Phase 4: close gripper
    RCLCPP_INFO(get_logger(), "[GPD pick] closing gripper");
    gripperClose();

    // Phase 5: lift (Cartesian)
    RCLCPP_INFO(get_logger(), "[GPD pick] lifting object");
    cartesianMove({lift_pose}, "lift");

    RCLCPP_INFO(get_logger(), "[GPD pick] Pick sequence complete.");
  }

  // ── Robot description fetcher (same as yolo_hover_linear_node) ──────────
  bool fetchRobotDescription()
  {
    RCLCPP_INFO(get_logger(),
      "[GPD pick] Fetching robot_description from /move_group (timeout=%d s)...",
      robot_desc_wait_sec_);

    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(
      shared_from_this(), "/move_group");

    bool ready = false;
    for (int i = 0; i < robot_desc_wait_sec_ * 2 && rclcpp::ok(); ++i) {
      if (param_client->service_is_ready()) { ready = true; break; }
      if (i % 2 == 0) {
        RCLCPP_INFO(get_logger(),
          "[GPD pick] Waiting for /move_group... (%d/%d s)", i / 2, robot_desc_wait_sec_);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (!ready) {
      RCLCPP_ERROR(get_logger(),
        "[GPD pick] /move_group not ready after %d s.\n"
        "  Run MoveIt first:\n"
        "    ros2 launch franka_fr3_moveit_config moveit.launch.py "
        "robot_ip:=172.16.0.2 robot_type:=fr3 load_gripper:=true",
        robot_desc_wait_sec_);
      return false;
    }

    auto future = param_client->get_parameters(
      {"robot_description", "robot_description_semantic"});

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
      if (!rclcpp::ok() || std::chrono::steady_clock::now() > deadline) {
        RCLCPP_ERROR(get_logger(),
          "[GPD pick] Timed out waiting for robot description. Aborting.");
        return false;
      }
    }

    bool all_ok = true;
    for (auto & p : future.get()) {
      if (p.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
        RCLCPP_ERROR(get_logger(), "[GPD pick] '%s' not set in /move_group.", p.get_name().c_str());
        all_ok = false;
        continue;
      }
      try {
        declare_parameter(p.get_name(), p.get_parameter_value());
      } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
        set_parameter(p);
      }
    }
    return all_ok;
  }

  // ── state ────────────────────────────────────────────────────────────────
  std::mutex    grasp_mutex_;
  GraspConfig   latest_grasp_;
  bool          has_grasp_;
  std::atomic<bool> processing_;
  bool          done_;
  std::thread   worker_thread_;

  // ── params ───────────────────────────────────────────────────────────────
  bool        dry_run_, once_, hover_only_, use_fixed_orientation_;
  bool        force_vertical_pregrasp_, force_vertical_approach_;
  bool        use_gpd_approach_vector_, use_gpd_orientation_;
  bool        use_transient_local_qos_, print_wait_hint_, fail_if_no_grasp_;
  double      pregrasp_z_offset_, approach_z_offset_, lift_z_offset_, min_safe_z_, safe_z_;
  double      min_x_, max_x_, min_y_, max_y_, min_z_, max_z_;
  double      fixed_qx_, fixed_qy_, fixed_qz_, fixed_qw_;
  double      gripper_width_open_, gripper_width_close_, gripper_force_;
  double      velocity_scaling_, acceleration_scaling_;
  double      eef_step_, jump_threshold_, min_fraction_, wait_for_grasp_timeout_sec_;
  std::string planning_group_, eef_link_, grasp_topic_, gripper_action_ns_;
  int         robot_desc_wait_sec_;
  std::chrono::steady_clock::time_point grasp_wait_started_{std::chrono::steady_clock::now()};
  bool        grasp_wait_timeout_reported_{false};
  bool        grasp_subscription_ready_{false};

  // ── ROS ──────────────────────────────────────────────────────────────────
  rclcpp::Subscription<GraspConfig>::SharedPtr sub_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp_action::Client<FrankaMove>::SharedPtr  gripper_move_client_;
  rclcpp_action::Client<FrankaGrasp>::SharedPtr gripper_grasp_client_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GpdPickExecuteNode>();

  // Spin in a separate thread so MoveGroupInterface can use callbacks
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->initMoveGroup();
  node->startWorker();

  spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
