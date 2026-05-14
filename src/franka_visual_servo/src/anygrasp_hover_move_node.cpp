#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>

#include <cmath>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <thread>
#include <vector>

struct ParsedPose
{
  double score;
  double x;
  double y;
  double z;
};

class AnyGraspHoverMove : public rclcpp::Node
{
public:
  AnyGraspHoverMove()
  : Node("anygrasp_hover_move_node")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/anygrasp/best_safe_grasp_pose_base");
    planning_group_ = declare_parameter<std::string>("planning_group", "fr3_arm");

    dry_run_   = declare_parameter<bool>("dry_run",   true);
    plan_only_ = declare_parameter<bool>("plan_only", true);
    execute_   = declare_parameter<bool>("execute",   false);
    once_      = declare_parameter<bool>("once",      true);
    auto_shutdown_after_once_ = declare_parameter<bool>("auto_shutdown_after_once", true);

    min_score_ = declare_parameter<double>("min_score", 0.2);
    x_min_     = declare_parameter<double>("x_min",  0.20);
    x_max_     = declare_parameter<double>("x_max",  0.70);
    y_min_     = declare_parameter<double>("y_min", -0.50);
    y_max_     = declare_parameter<double>("y_max",  0.50);
    z_min_     = declare_parameter<double>("z_min",  0.20);
    z_max_     = declare_parameter<double>("z_max",  0.80);

    max_step_              = declare_parameter<double>("max_step",              0.35);
    velocity_scaling_      = declare_parameter<double>("velocity_scaling",      0.05);
    acceleration_scaling_  = declare_parameter<double>("acceleration_scaling",  0.05);
    robot_desc_timeout_sec_ = declare_parameter<int>("robot_desc_timeout_sec", 30);

    use_cartesian_path_ = declare_parameter<bool>("use_cartesian_path", true);
    safe_z_ = declare_parameter<double>("safe_z", 0.55);
    approach_mode_ = declare_parameter<std::string>("approach_mode", "xy_then_vertical");
    cartesian_eef_step_ = declare_parameter<double>("cartesian_eef_step", 0.01);
    cartesian_jump_threshold_ = declare_parameter<double>("cartesian_jump_threshold", 0.0);
    min_cartesian_fraction_ = declare_parameter<double>("min_cartesian_fraction", 0.95);
    return_after_execute_ = declare_parameter<bool>("return_after_execute", false);
    return_z_ = declare_parameter<double>("return_z", 0.55);

    loop_mode_ = declare_parameter<bool>("loop_mode", false);
    loop_period_sec_ = declare_parameter<double>("loop_period_sec", 2.0);
    go_initial_before_target_ = declare_parameter<bool>("go_initial_before_target", true);
    return_initial_after_target_ = declare_parameter<bool>("return_initial_after_target", true);
    initial_move_mode_ = declare_parameter<std::string>("initial_move_mode", "joint");

    initial_x_ = declare_parameter<double>("initial_x", 0.55);
    initial_y_ = declare_parameter<double>("initial_y", 0.00);
    initial_z_ = declare_parameter<double>("initial_z", 0.55);
    initial_qx_ = declare_parameter<double>("initial_qx", 0.0108);
    initial_qy_ = declare_parameter<double>("initial_qy", -0.0128);
    initial_qz_ = declare_parameter<double>("initial_qz", -0.0037);
    initial_qw_ = declare_parameter<double>("initial_qw", 0.9999);

    initial_j1_ = declare_parameter<double>("initial_j1", 0.0);
    initial_j2_ = declare_parameter<double>("initial_j2", -0.785);
    initial_j3_ = declare_parameter<double>("initial_j3", 0.0);
    initial_j4_ = declare_parameter<double>("initial_j4", -2.356);
    initial_j5_ = declare_parameter<double>("initial_j5", 0.0);
    initial_j6_ = declare_parameter<double>("initial_j6", 1.571);
    initial_j7_ = declare_parameter<double>("initial_j7", 0.785);

    use_fixed_orientation_ = declare_parameter<bool>("use_fixed_orientation", true);
    fixed_qx_ = declare_parameter<double>("fixed_qx", 0.0108);
    fixed_qy_ = declare_parameter<double>("fixed_qy", -0.0128);
    fixed_qz_ = declare_parameter<double>("fixed_qz", -0.0037);
    fixed_qw_ = declare_parameter<double>("fixed_qw", 0.9999);
  }

  ~AnyGraspHoverMove() override
  {
    stop_loop_.store(true);
    if (loop_thread_.joinable()) {
      loop_thread_.join();
    }
  }

  // Must be called AFTER executor.spin() is running in a separate thread.
  void initMoveGroup()
  {
    if (!fetchRobotDescription()) {
      RCLCPP_FATAL(get_logger(), "Failed to load robot description. Shutting down.");
      rclcpp::shutdown();
      return;
    }

    RCLCPP_INFO(get_logger(), "Creating MoveGroupInterface (group=%s)...", planning_group_.c_str());
    try {
      move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), planning_group_);
    } catch (const std::exception & e) {
      RCLCPP_FATAL(get_logger(), "MoveGroupInterface construction failed: %s", e.what());
      rclcpp::shutdown();
      return;
    }

    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);
    move_group_->setPlanningTime(5.0);

    RCLCPP_INFO(get_logger(), "MoveGroupInterface ready. planning_frame=%s",
      move_group_->getPlanningFrame().c_str());

    // Explicitly start CurrentStateMonitor and wait for first valid joint state.
    RCLCPP_INFO(get_logger(), "Starting CurrentStateMonitor (wait up to 5 s)...");
    move_group_->startStateMonitor(5.0);
    auto initial_state = move_group_->getCurrentState(5.0);
    if (!initial_state) {
      RCLCPP_WARN(get_logger(),
        "Could not get initial robot state within 5 s. "
        "Will retry per-callback — ensure /joint_states is publishing.");
    } else {
      RCLCPP_INFO(get_logger(), "Current robot state is ready.");
    }

    RCLCPP_INFO(get_logger(),
      "dry_run=%s  plan_only=%s  execute=%s  once=%s  auto_shutdown_after_once=%s",
      dry_run_ ? "true" : "false",
      plan_only_ ? "true" : "false",
      execute_ ? "true" : "false",
      once_ ? "true" : "false",
      auto_shutdown_after_once_ ? "true" : "false");
    RCLCPP_INFO(get_logger(),
      "use_cartesian_path=%s  safe_z=%.3f  approach_mode=%s  "
      "eef_step=%.3f  jump_threshold=%.3f  min_fraction=%.2f  "
      "return_after_execute=%s  return_z=%.3f",
      use_cartesian_path_ ? "true" : "false",
      safe_z_,
      approach_mode_.c_str(),
      cartesian_eef_step_,
      cartesian_jump_threshold_,
      min_cartesian_fraction_,
      return_after_execute_ ? "true" : "false",
      return_z_);
    RCLCPP_INFO(get_logger(),
      "loop_mode=%s  loop_period_sec=%.3f  go_initial_before_target=%s  "
      "return_initial_after_target=%s  initial_move_mode=%s",
      loop_mode_ ? "true" : "false",
      loop_period_sec_,
      go_initial_before_target_ ? "true" : "false",
      return_initial_after_target_ ? "true" : "false",
      initial_move_mode_.c_str());
    RCLCPP_INFO(get_logger(),
      "initial pose=(%.3f, %.3f, %.3f) q=(%.4f, %.4f, %.4f, %.4f)",
      initial_x_, initial_y_, initial_z_,
      initial_qx_, initial_qy_, initial_qz_, initial_qw_);
    RCLCPP_INFO(get_logger(),
      "initial joints=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
      initial_j1_, initial_j2_, initial_j3_, initial_j4_,
      initial_j5_, initial_j6_, initial_j7_);
    RCLCPP_INFO(get_logger(),
      "use_fixed_orientation=%s fixed_q=(%.4f, %.4f, %.4f, %.4f)",
      use_fixed_orientation_ ? "true" : "false",
      fixed_qx_, fixed_qy_, fixed_qz_, fixed_qw_);

    // Separate callback group so getCurrentState() blocking calls inside the
    // callback do not starve MoveIt's own joint_states subscriber.
    anygrasp_cb_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = anygrasp_cb_group_;

    sub_ = create_subscription<std_msgs::msg::String>(
      input_topic_, rclcpp::QoS(10),
      std::bind(&AnyGraspHoverMove::callback, this, std::placeholders::_1),
      sub_opts);

    RCLCPP_INFO(get_logger(), "Subscribed to %s", input_topic_.c_str());

    if (loop_mode_) {
      RCLCPP_INFO(get_logger(),
        "Persistent loop_mode=true: MoveGroupInterface will stay alive and reuse latest target.");
      loop_thread_ = std::thread([this]() { loopMain(); });
    }
  }

private:
  // ── robot description loader (copies from /move_group parameter server) ── //

  bool fetchRobotDescription()
  {
    RCLCPP_INFO(get_logger(),
      "Fetching robot_description from /move_group (timeout=%d s)...",
      robot_desc_timeout_sec_);

    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(
      shared_from_this(), "/move_group");

    bool ready = false;
    for (int i = 0; i < robot_desc_timeout_sec_ * 2 && rclcpp::ok(); ++i) {
      if (param_client->service_is_ready()) { ready = true; break; }
      if (i % 4 == 0) {
        RCLCPP_INFO(get_logger(),
          "Waiting for /move_group parameter service... (%d/%d s)",
          i / 2, robot_desc_timeout_sec_);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    if (!ready) {
      RCLCPP_ERROR(get_logger(),
        "/move_group parameter service not ready after %d s.\n"
        "Make sure MoveIt is running:\n"
        "  ros2 launch franka_fr3_moveit_config moveit.launch.py "
        "robot_ip:=172.16.0.2 robot_type:=fr3 load_gripper:=true",
        robot_desc_timeout_sec_);
      return false;
    }

    auto future = param_client->get_parameters(
      {"robot_description", "robot_description_semantic"});

    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
      if (!rclcpp::ok() || std::chrono::steady_clock::now() > deadline) {
        RCLCPP_ERROR(get_logger(), "Timed out waiting for robot description. Aborting.");
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

  // ── JSON parser (regex-based, no extra deps) ─────────────────────────── //

  static std::optional<double> extract_number_after_key(
    const std::string & text, const std::string & key)
  {
    const std::string pattern =
      "\"" + key + "\"\\s*:\\s*(-?[0-9]+\\.?[0-9]*(?:[eE][-+]?[0-9]+)?)";
    std::regex re(pattern);
    std::smatch m;
    if (std::regex_search(text, m, re)) {
      return std::stod(m[1].str());
    }
    return std::nullopt;
  }

  static std::optional<ParsedPose> parse_json(const std::string & text)
  {
    auto score = extract_number_after_key(text, "score");
    if (!score) return std::nullopt;

    const auto pos = text.find("\"hover_translation\"");
    if (pos == std::string::npos) return std::nullopt;

    const std::string tail = text.substr(pos);
    auto x = extract_number_after_key(tail, "x");
    auto y = extract_number_after_key(tail, "y");
    auto z = extract_number_after_key(tail, "z");
    if (!x || !y || !z) return std::nullopt;

    return ParsedPose{*score, *x, *y, *z};
  }

  // ── callback ─────────────────────────────────────────────────────────── //

  void callback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!loop_mode_ && processed_ && once_) return;

    const auto parsed = parse_json(msg->data);
    if (!parsed) {
      RCLCPP_WARN(get_logger(), "Failed to parse AnyGrasp JSON.");
      if (!loop_mode_) {
        processed_ = true;
      }
      return;
    }

    if (loop_mode_) {
      {
        std::lock_guard<std::mutex> lock(latest_target_mutex_);
        latest_target_ = *parsed;
        has_latest_target_ = true;
      }
      RCLCPP_INFO(get_logger(),
        "latest AnyGrasp target updated: score=%.4f hover=(%.3f, %.3f, %.3f)%s",
        parsed->score, parsed->x, parsed->y, parsed->z,
        executing_cycle_.load() ? " while cycle is running" : "");
      return;
    }

    executeParsedTarget(*parsed, true);
  }

  bool executeParsedTarget(const ParsedPose & target, bool finish_after)
  {
    const double score = target.score;
    const double x = target.x;
    const double y = target.y;
    const double z = target.z;

    RCLCPP_INFO(get_logger(), "received best safe AnyGrasp pose");
    RCLCPP_INFO(get_logger(), "score=%.4f", score);
    RCLCPP_INFO(get_logger(), "target hover=(%.3f, %.3f, %.3f)", x, y, z);

    if (score < min_score_) {
      RCLCPP_WARN(get_logger(), "REJECTED: score %.4f < min_score %.4f", score, min_score_);
      if (finish_after) finishOnceIfNeeded();
      return false;
    }
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      RCLCPP_WARN(get_logger(), "REJECTED: NaN/inf in target");
      if (finish_after) finishOnceIfNeeded();
      return false;
    }
    if (x < x_min_ || x > x_max_) {
      RCLCPP_WARN(get_logger(), "REJECTED: x %.3f outside [%.3f, %.3f]", x, x_min_, x_max_);
      if (finish_after) finishOnceIfNeeded();
      return false;
    }
    if (y < y_min_ || y > y_max_) {
      RCLCPP_WARN(get_logger(), "REJECTED: y %.3f outside [%.3f, %.3f]", y, y_min_, y_max_);
      if (finish_after) finishOnceIfNeeded();
      return false;
    }
    if (z < z_min_ || z > z_max_) {
      RCLCPP_WARN(get_logger(), "REJECTED: z %.3f outside [%.3f, %.3f]", z, z_min_, z_max_);
      if (finish_after) finishOnceIfNeeded();
      return false;
    }

    // Guard: require a fresh robot state before reading EE pose.
    // Do NOT mark processed_ here so the next message retries automatically.
    {
      auto cs = move_group_->getCurrentState(3.0);
      if (!cs) {
        RCLCPP_WARN(get_logger(),
          "Current robot state is not available. Skip this target.");
        return false;
      }
    }

    geometry_msgs::msg::PoseStamped current_pose;
    try {
      current_pose = move_group_->getCurrentPose();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to get current EE pose: %s", e.what());
      return false;  // retry — do not mark processed
    }

    const double cx = current_pose.pose.position.x;
    const double cy = current_pose.pose.position.y;
    const double cz = current_pose.pose.position.z;

    // Reject zero-origin pose: impossible for FR3, means monitor not ready yet.
    if (std::sqrt(cx * cx + cy * cy + cz * cz) < 0.01) {
      RCLCPP_WARN(get_logger(),
        "REJECTED: current EE=(%.3f, %.3f, %.3f) is near origin "
        "— state monitor not ready yet. Retrying next message.", cx, cy, cz);
      return false;
    }

    const geometry_msgs::msg::Pose reference_pose =
      (loop_mode_ && go_initial_before_target_ && initial_move_mode_ == "cartesian_pose") ?
      makeInitialPose() : current_pose.pose;
    const double rx = reference_pose.position.x;
    const double ry = reference_pose.position.y;
    const double rz = reference_pose.position.z;
    const double dist = std::sqrt(
      (x - rx) * (x - rx) + (y - ry) * (y - ry) + (z - rz) * (z - rz));

    RCLCPP_INFO(get_logger(), "current EE=(%.3f, %.3f, %.3f)", cx, cy, cz);
    RCLCPP_INFO(get_logger(),
      "distance to hover target=%.3f m from reference=(%.3f, %.3f, %.3f)",
      dist, rx, ry, rz);

    if (dist > max_step_) {
      RCLCPP_WARN(get_logger(), "REJECTED: dist %.3f > max_step %.3f", dist, max_step_);
      if (finish_after) finishOnceIfNeeded();
      return false;
    }

    RCLCPP_INFO(get_logger(), "safety check passed");

    if (dry_run_) {
      logCartesianPreview(current_pose.pose, x, y, z);
      RCLCPP_INFO(get_logger(), "dry_run=true → not planning or executing");
      if (finish_after) finishOnceIfNeeded();
      return true;
    }

    geometry_msgs::msg::Pose target_pose = current_pose.pose;
    target_pose.position.x = x;
    target_pose.position.y = y;
    target_pose.position.z = z;
    target_pose.orientation = targetOrientation(current_pose.pose);

    RCLCPP_INFO(get_logger(), "orientation: %s",
      use_fixed_orientation_ ? "using fixed orientation" : "keep current EE orientation");
    RCLCPP_INFO(get_logger(), "target position=(%.3f, %.3f, %.3f)", x, y, z);

    if (use_cartesian_path_) {
      const bool ok = executeCartesianHover(current_pose.pose, target_pose);
      (void)ok;
      if (finish_after) finishOnceIfNeeded();
      return ok;
    }

    move_group_->setPoseTarget(target_pose);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool plan_ok = static_cast<bool>(move_group_->plan(plan));

    if (!plan_ok) {
      RCLCPP_ERROR(get_logger(), "planning failed");
      if (finish_after) finishOnceIfNeeded();
      return false;
    }
    RCLCPP_INFO(get_logger(), "planning success");

    if (plan_only_ || !execute_) {
      RCLCPP_INFO(get_logger(), "plan_only=true or execute=false → not executing");
      if (finish_after) finishOnceIfNeeded();
      return true;
    }

    RCLCPP_WARN(get_logger(), "EXECUTING hover move only. No gripper/descent/lift.");
    const auto result = move_group_->execute(plan);
    if (result == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(get_logger(), "hover execute SUCCESS");
    } else {
      RCLCPP_ERROR(get_logger(), "hover execute FAILED");
    }

    if (finish_after) finishOnceIfNeeded();
    return result == moveit::core::MoveItErrorCode::SUCCESS;
  }

  void requestShutdownIfOnce()
  {
    if (!once_ || !auto_shutdown_after_once_) {
      return;
    }

    RCLCPP_INFO(get_logger(), "once=true: auto shutdown requested.");
    std::thread([]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      if (rclcpp::ok()) {
        rclcpp::shutdown();
      }
    }).detach();
  }

  void finishOnceIfNeeded()
  {
    processed_ = true;
    requestShutdownIfOnce();
  }

  geometry_msgs::msg::Quaternion normalizedQuaternion(
    double x, double y, double z, double w)
  {
    geometry_msgs::msg::Quaternion q;
    const double norm = std::sqrt(x * x + y * y + z * z + w * w);
    if (norm < 1e-9) {
      q.w = 1.0;
      return q;
    }
    q.x = x / norm;
    q.y = y / norm;
    q.z = z / norm;
    q.w = w / norm;
    return q;
  }

  geometry_msgs::msg::Quaternion fixedOrientation()
  {
    return normalizedQuaternion(fixed_qx_, fixed_qy_, fixed_qz_, fixed_qw_);
  }

  geometry_msgs::msg::Quaternion initialOrientation()
  {
    return normalizedQuaternion(initial_qx_, initial_qy_, initial_qz_, initial_qw_);
  }

  geometry_msgs::msg::Quaternion targetOrientation(
    const geometry_msgs::msg::Pose & current_pose)
  {
    if (use_fixed_orientation_) {
      return fixedOrientation();
    }
    return current_pose.orientation;
  }

  geometry_msgs::msg::Pose makeInitialPose()
  {
    geometry_msgs::msg::Pose pose;
    pose.position.x = initial_x_;
    pose.position.y = initial_y_;
    pose.position.z = initial_z_;
    pose.orientation = initialOrientation();
    return pose;
  }

  void logPose(const std::string & label, const geometry_msgs::msg::Pose & pose)
  {
    RCLCPP_INFO(get_logger(),
      "%s position=(%.3f, %.3f, %.3f) q=(%.4f, %.4f, %.4f, %.4f)",
      label.c_str(),
      pose.position.x, pose.position.y, pose.position.z,
      pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
  }

  std::vector<geometry_msgs::msg::Pose> makeHoverWaypoints(
    const geometry_msgs::msg::Pose & current_pose,
    const geometry_msgs::msg::Pose & target_pose)
  {
    std::vector<geometry_msgs::msg::Pose> waypoints;

    if (approach_mode_ != "xy_then_vertical") {
      RCLCPP_WARN(get_logger(),
        "Unsupported approach_mode='%s'; using xy_then_vertical.",
        approach_mode_.c_str());
    }

    geometry_msgs::msg::Pose p1 = current_pose;
    p1.position.z = safe_z_;
    p1.orientation = targetOrientation(current_pose);

    geometry_msgs::msg::Pose p2 = current_pose;
    p2.position.x = target_pose.position.x;
    p2.position.y = target_pose.position.y;
    p2.position.z = safe_z_;
    p2.orientation = targetOrientation(current_pose);

    geometry_msgs::msg::Pose p3 = current_pose;
    p3.position.x = target_pose.position.x;
    p3.position.y = target_pose.position.y;
    p3.position.z = target_pose.position.z;
    p3.orientation = targetOrientation(current_pose);

    waypoints.push_back(p1);
    waypoints.push_back(p2);
    waypoints.push_back(p3);
    return waypoints;
  }

  void logCartesianPreview(
    const geometry_msgs::msg::Pose & current_pose,
    double target_x,
    double target_y,
    double target_z)
  {
    geometry_msgs::msg::Pose target_pose = current_pose;
    target_pose.position.x = target_x;
    target_pose.position.y = target_y;
    target_pose.position.z = target_z;
    target_pose.orientation = targetOrientation(current_pose);

    RCLCPP_INFO(get_logger(), "Cartesian hover preview only.");
    logPose("current_pose", current_pose);
    logPose("target_hover_pose", target_pose);

    const auto waypoints = makeHoverWaypoints(current_pose, target_pose);
    for (size_t i = 0; i < waypoints.size(); ++i) {
      logPose("waypoint " + std::to_string(i + 1), waypoints[i]);
    }
  }

  bool computeCartesianPlan(
    const std::string & phase,
    const std::vector<geometry_msgs::msg::Pose> & waypoints,
    moveit_msgs::msg::RobotTrajectory & trajectory)
  {
    RCLCPP_INFO(get_logger(),
      "[%s] computeCartesianPath: waypoints=%zu eef_step=%.3f jump_threshold=%.3f",
      phase.c_str(),
      waypoints.size(),
      cartesian_eef_step_,
      cartesian_jump_threshold_);

    const double fraction = move_group_->computeCartesianPath(
      waypoints,
      cartesian_eef_step_,
      cartesian_jump_threshold_,
      trajectory);

    RCLCPP_INFO(get_logger(), "[%s] Cartesian fraction=%.4f", phase.c_str(), fraction);
    if (fraction < min_cartesian_fraction_) {
      RCLCPP_WARN(get_logger(),
        "[%s] REJECTED: Cartesian fraction %.4f < min_cartesian_fraction %.4f. Not executing.",
        phase.c_str(), fraction, min_cartesian_fraction_);
      return false;
    }

    return true;
  }

  bool executeTrajectory(
    const std::string & phase,
    const moveit_msgs::msg::RobotTrajectory & trajectory)
  {
    if (plan_only_ || !execute_) {
      RCLCPP_INFO(get_logger(),
        "[%s] plan_only=true or execute=false → not executing",
        phase.c_str());
      return true;
    }

    RCLCPP_WARN(get_logger(), "[%s] EXECUTING Cartesian hover path only. No gripper/descent/grasp/lift.", phase.c_str());
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = trajectory;
    const auto result = move_group_->execute(plan);
    if (result == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(get_logger(), "[%s] execute SUCCESS", phase.c_str());
      return true;
    }

    RCLCPP_ERROR(get_logger(), "[%s] execute FAILED", phase.c_str());
    return false;
  }

  bool executeCartesianHover(
    const geometry_msgs::msg::Pose & current_pose,
    const geometry_msgs::msg::Pose & target_pose)
  {
    logCartesianPreview(
      current_pose,
      target_pose.position.x,
      target_pose.position.y,
      target_pose.position.z);

    const auto waypoints = makeHoverWaypoints(current_pose, target_pose);
    moveit_msgs::msg::RobotTrajectory trajectory;
    if (!computeCartesianPlan("hover_approach", waypoints, trajectory)) {
      return false;
    }

    if (!executeTrajectory("hover_approach", trajectory)) {
      return false;
    }

    if (!return_after_execute_ || plan_only_ || !execute_) {
      return true;
    }

    geometry_msgs::msg::PoseStamped after_pose;
    try {
      after_pose = move_group_->getCurrentPose();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to get current EE pose before return lift: %s", e.what());
      return false;
    }

    geometry_msgs::msg::Pose return_pose = after_pose.pose;
    return_pose.position.z = return_z_;
    return_pose.orientation = use_fixed_orientation_ ? fixedOrientation() : after_pose.pose.orientation;

    RCLCPP_INFO(get_logger(),
      "return_after_execute=true: Cartesian lift-only return to z=%.3f at current x,y.",
      return_z_);
    logPose("return current_pose", after_pose.pose);
    logPose("return waypoint 1", return_pose);

    moveit_msgs::msg::RobotTrajectory return_trajectory;
    std::vector<geometry_msgs::msg::Pose> return_waypoints{ return_pose };
    if (!computeCartesianPlan("return_lift_only", return_waypoints, return_trajectory)) {
      return false;
    }

    return executeTrajectory("return_lift_only", return_trajectory);
  }

  bool moveToInitialPoseCartesian(const std::string & phase)
  {
    geometry_msgs::msg::PoseStamped current_pose;
    try {
      current_pose = move_group_->getCurrentPose();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "[%s] Failed to get current EE pose: %s", phase.c_str(), e.what());
      return false;
    }

    const auto initial_pose = makeInitialPose();
    RCLCPP_INFO(get_logger(), "[%s] moving to initial pose", phase.c_str());
    logPose("initial current_pose", current_pose.pose);
    logPose("initial target_pose", initial_pose);

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "[%s] dry_run=true, not planning or executing initial move.", phase.c_str());
      return true;
    }

    moveit_msgs::msg::RobotTrajectory trajectory;
    std::vector<geometry_msgs::msg::Pose> waypoints{ initial_pose };
    if (!computeCartesianPlan(phase, waypoints, trajectory)) {
      return false;
    }
    return executeTrajectory(phase, trajectory);
  }

  std::map<std::string, double> makeInitialJointTarget()
  {
    std::map<std::string, double> joint_target;
    joint_target["fr3_joint1"] = initial_j1_;
    joint_target["fr3_joint2"] = initial_j2_;
    joint_target["fr3_joint3"] = initial_j3_;
    joint_target["fr3_joint4"] = initial_j4_;
    joint_target["fr3_joint5"] = initial_j5_;
    joint_target["fr3_joint6"] = initial_j6_;
    joint_target["fr3_joint7"] = initial_j7_;
    return joint_target;
  }

  bool moveToInitialJointPose(const std::string & phase)
  {
    RCLCPP_INFO(get_logger(),
      "[%s] moving to initial joint pose: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
      phase.c_str(),
      initial_j1_, initial_j2_, initial_j3_, initial_j4_,
      initial_j5_, initial_j6_, initial_j7_);

    const auto joint_target = makeInitialJointTarget();

    if (dry_run_) {
      RCLCPP_INFO(get_logger(),
        "[%s] dry_run=true, not planning or executing initial joint move.",
        phase.c_str());
      return true;
    }

    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);
    move_group_->setJointValueTarget(joint_target);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool plan_ok = static_cast<bool>(move_group_->plan(plan));
    if (!plan_ok) {
      RCLCPP_ERROR(get_logger(),
        "[%s] initial joint move planning FAILED",
        phase.c_str());
      return false;
    }

    RCLCPP_INFO(get_logger(),
      "[%s] initial joint move planning SUCCESS",
      phase.c_str());

    if (plan_only_ || !execute_) {
      RCLCPP_INFO(get_logger(),
        "[%s] plan_only=true or execute=false → not executing initial joint move",
        phase.c_str());
      return true;
    }

    RCLCPP_WARN(get_logger(),
      "[%s] EXECUTING initial joint move only. No gripper/descent/grasp/lift.",
      phase.c_str());
    const auto result = move_group_->execute(plan);
    if (result == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(get_logger(),
        "[%s] initial joint move execute SUCCESS",
        phase.c_str());
      return true;
    }

    RCLCPP_ERROR(get_logger(),
      "[%s] initial joint move execute FAILED",
      phase.c_str());
    return false;
  }

  bool moveToInitial(const std::string & phase)
  {
    RCLCPP_INFO(get_logger(), "[%s] initial_move_mode=%s", phase.c_str(), initial_move_mode_.c_str());
    if (initial_move_mode_ == "joint") {
      return moveToInitialJointPose(phase);
    }
    if (initial_move_mode_ == "cartesian_pose") {
      return moveToInitialPoseCartesian(phase);
    }

    RCLCPP_ERROR(get_logger(),
      "[%s] Unsupported initial_move_mode='%s'. Use 'joint' or 'cartesian_pose'.",
      phase.c_str(),
      initial_move_mode_.c_str());
    return false;
  }

  bool returnToInitialPoseCartesian()
  {
    geometry_msgs::msg::PoseStamped current_pose;
    try {
      current_pose = move_group_->getCurrentPose();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "[return_initial] Failed to get current EE pose: %s", e.what());
      return false;
    }

    const auto initial_pose = makeInitialPose();
    geometry_msgs::msg::Pose p1 = current_pose.pose;
    p1.position.z = safe_z_;
    p1.orientation = initialOrientation();

    geometry_msgs::msg::Pose p2 = initial_pose;
    p2.position.z = safe_z_;
    p2.orientation = initialOrientation();

    geometry_msgs::msg::Pose p3 = initial_pose;
    p3.orientation = initialOrientation();

    RCLCPP_INFO(get_logger(), "[return_initial] returning to initial pose");
    logPose("return current_pose", current_pose.pose);
    logPose("return waypoint 1", p1);
    logPose("return waypoint 2", p2);
    logPose("return waypoint 3", p3);

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "[return_initial] dry_run=true, not planning or executing return.");
      return true;
    }

    moveit_msgs::msg::RobotTrajectory trajectory;
    std::vector<geometry_msgs::msg::Pose> waypoints{ p1, p2, p3 };
    if (!computeCartesianPlan("return_initial", waypoints, trajectory)) {
      return false;
    }
    return executeTrajectory("return_initial", trajectory);
  }

  bool returnToInitial()
  {
    RCLCPP_INFO(get_logger(), "[return_initial] initial_move_mode=%s", initial_move_mode_.c_str());
    if (initial_move_mode_ == "joint") {
      return moveToInitialJointPose("return_initial_joint");
    }
    if (initial_move_mode_ == "cartesian_pose") {
      return returnToInitialPoseCartesian();
    }

    RCLCPP_ERROR(get_logger(),
      "[return_initial] Unsupported initial_move_mode='%s'. Use 'joint' or 'cartesian_pose'.",
      initial_move_mode_.c_str());
    return false;
  }


  void sleepLoopPeriod()
  {
    const auto total_ms = static_cast<int>(std::max(0.0, loop_period_sec_) * 1000.0);
    const auto step = std::chrono::milliseconds(100);
    int slept_ms = 0;
    while (rclcpp::ok() && !stop_loop_.load() && slept_ms < total_ms) {
      std::this_thread::sleep_for(step);
      slept_ms += 100;
    }
  }

  std::optional<ParsedPose> latestTargetCopy()
  {
    std::lock_guard<std::mutex> lock(latest_target_mutex_);
    if (!has_latest_target_) {
      return std::nullopt;
    }
    return latest_target_;
  }

  void loopMain()
  {
    size_t cycle = 0;
    while (rclcpp::ok() && !stop_loop_.load()) {
      ++cycle;
      executing_cycle_.store(true);
      RCLCPP_INFO(get_logger(), "[loop] cycle %zu start", cycle);
      RCLCPP_INFO(get_logger(), "[loop] fixed orientation active: %s",
        use_fixed_orientation_ ? "true" : "false");

      bool initial_ok = true;
      if (go_initial_before_target_) {
        initial_ok = moveToInitial("initial_before_target");
        RCLCPP_INFO(get_logger(), "[loop] initial move %s", initial_ok ? "SUCCESS" : "FAILED");
      }

      sleepLoopPeriod();

      const auto target = latestTargetCopy();
      if (!initial_ok) {
        RCLCPP_WARN(get_logger(),
          "[loop] initial move failed; skipping target hover for this cycle.");
      } else if (target) {
        RCLCPP_INFO(get_logger(),
          "[loop] latest target: score=%.4f hover=(%.3f, %.3f, %.3f)",
          target->score, target->x, target->y, target->z);
        const bool target_ok = executeParsedTarget(*target, false);
        RCLCPP_INFO(get_logger(), "[loop] target hover move %s", target_ok ? "SUCCESS" : "FAILED");
      } else {
        RCLCPP_WARN(get_logger(), "[loop] no AnyGrasp target received yet; skipping target hover.");
      }

      if (return_initial_after_target_) {
        const bool return_ok = returnToInitial();
        RCLCPP_INFO(get_logger(), "[loop] return initial %s", return_ok ? "SUCCESS" : "FAILED");
      }

      executing_cycle_.store(false);
      RCLCPP_INFO(get_logger(), "[loop] cycle %zu end", cycle);
      sleepLoopPeriod();
    }
  }

  // ── members ──────────────────────────────────────────────────────────── //

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp::CallbackGroup::SharedPtr anygrasp_cb_group_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;

  std::string input_topic_;
  std::string planning_group_;
  std::string approach_mode_;
  std::string initial_move_mode_;
  bool   dry_run_, plan_only_, execute_, once_;
  bool   auto_shutdown_after_once_, use_cartesian_path_, return_after_execute_;
  bool   loop_mode_, go_initial_before_target_, return_initial_after_target_;
  bool   use_fixed_orientation_;
  double min_score_;
  double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
  double max_step_, velocity_scaling_, acceleration_scaling_;
  double safe_z_, cartesian_eef_step_, cartesian_jump_threshold_;
  double min_cartesian_fraction_, return_z_;
  double loop_period_sec_;
  double initial_x_, initial_y_, initial_z_;
  double initial_qx_, initial_qy_, initial_qz_, initial_qw_;
  double initial_j1_, initial_j2_, initial_j3_, initial_j4_, initial_j5_, initial_j6_, initial_j7_;
  double fixed_qx_, fixed_qy_, fixed_qz_, fixed_qw_;
  int    robot_desc_timeout_sec_;

  bool processed_ = false;
  std::mutex latest_target_mutex_;
  ParsedPose latest_target_{0.0, 0.0, 0.0, 0.0};
  bool has_latest_target_ = false;
  std::atomic_bool executing_cycle_{false};
  std::atomic_bool stop_loop_{false};
  std::thread loop_thread_;
};

// ── main ──────────────────────────────────────────────────────────────────── //

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AnyGraspHoverMove>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  // Spin in background so AsyncParametersClient and MoveGroupInterface
  // can process callbacks during initMoveGroup().
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->initMoveGroup();

  spin_thread.join();
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
