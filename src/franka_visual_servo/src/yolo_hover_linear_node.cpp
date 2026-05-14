#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "moveit_msgs/msg/robot_trajectory.hpp"

// Extract first 3 floats from string regardless of format.
// Handles: "0.45 0.12 0.28", "x=0.45 y=0.12 z=0.28", {"x":0.45,...}, etc.
static bool parseXYZ(const std::string & s, double & x, double & y, double & z)
{
  std::regex num_re(R"([-+]?(?:[0-9]+\.?[0-9]*|[0-9]*\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
  std::vector<double> nums;
  for (auto it = std::sregex_iterator(s.begin(), s.end(), num_re);
       it != std::sregex_iterator() && nums.size() < 3; ++it)
  {
    nums.push_back(std::stod(it->str()));
  }
  if (nums.size() < 3) return false;
  x = nums[0]; y = nums[1]; z = nums[2];
  return true;
}

// ─────────────────────────────────────────────────────────────────────────── //

class YoloHoverLinearNode : public rclcpp::Node
{
public:
  YoloHoverLinearNode()
  : Node("yolo_hover_linear_node"),
    has_target_(false), processing_(false), done_(false),
    target_x_(0.0), target_y_(0.0), target_z_(0.0)
  {
    declare_parameter<std::string>("target_topic",      "/yolo/target_base");
    declare_parameter<std::string>("planning_group",    "fr3_arm");
    declare_parameter<std::string>("eef_link",          "fr3_hand_tcp");
    declare_parameter<double>("safe_z",                 0.45);
    declare_parameter<double>("hover_z",                0.04);
    declare_parameter<double>("x_offset",               0.24);
    declare_parameter<double>("y_offset",               0.005);
    declare_parameter<bool>("dry_run",                  true);
    declare_parameter<bool>("once",                     true);
    declare_parameter<double>("eef_step",               0.005);
    declare_parameter<double>("jump_threshold",         0.0);
    declare_parameter<double>("min_fraction",           0.95);
    declare_parameter<double>("velocity_scale",         0.05);
    declare_parameter<double>("acceleration_scale",     0.05);
    declare_parameter<bool>("use_fixed_orientation",    false);
    declare_parameter<double>("fixed_qx",               1.0);
    declare_parameter<double>("fixed_qy",               0.0);
    declare_parameter<double>("fixed_qz",               0.0);
    declare_parameter<double>("fixed_qw",               0.0);
    declare_parameter<bool>("move_to_ready_first",      true);
    declare_parameter<double>("ready_wait_sec",         1.0);
    declare_parameter<double>("ready_joint1",           0.0);
    declare_parameter<double>("ready_joint2",          -0.785);
    declare_parameter<double>("ready_joint3",           0.0);
    declare_parameter<double>("ready_joint4",          -2.356);
    declare_parameter<double>("ready_joint5",           0.0);
    declare_parameter<double>("ready_joint6",           1.571);
    declare_parameter<double>("ready_joint7",           0.785);
    declare_parameter<bool>("use_grid_linear_offset",   false);
    declare_parameter<double>("grid_col",               0.0);
    declare_parameter<double>("grid_row",               0.0);
    declare_parameter<bool>("use_target_linear_offset", false);
    declare_parameter<double>("x_offset_kx",           -0.8375);
    declare_parameter<double>("x_offset_ky",           -0.8178);
    declare_parameter<double>("x_offset_b",             0.3806);
    declare_parameter<double>("y_offset_kx",            0.8610);
    declare_parameter<double>("y_offset_ky",           -0.8059);
    declare_parameter<double>("y_offset_b",            -0.2468);
    declare_parameter<int>("robot_description_wait_timeout_sec", 30);

    target_topic_          = get_parameter("target_topic").as_string();
    planning_group_        = get_parameter("planning_group").as_string();
    eef_link_              = get_parameter("eef_link").as_string();
    safe_z_                = get_parameter("safe_z").as_double();
    hover_z_               = get_parameter("hover_z").as_double();
    x_offset_              = get_parameter("x_offset").as_double();
    y_offset_              = get_parameter("y_offset").as_double();
    dry_run_               = get_parameter("dry_run").as_bool();
    once_                  = get_parameter("once").as_bool();
    eef_step_              = get_parameter("eef_step").as_double();
    jump_threshold_        = get_parameter("jump_threshold").as_double();
    min_fraction_          = get_parameter("min_fraction").as_double();
    velocity_scale_        = get_parameter("velocity_scale").as_double();
    acceleration_scale_    = get_parameter("acceleration_scale").as_double();
    use_fixed_orientation_ = get_parameter("use_fixed_orientation").as_bool();
    fixed_qx_              = get_parameter("fixed_qx").as_double();
    fixed_qy_              = get_parameter("fixed_qy").as_double();
    fixed_qz_              = get_parameter("fixed_qz").as_double();
    fixed_qw_              = get_parameter("fixed_qw").as_double();
    move_to_ready_first_   = get_parameter("move_to_ready_first").as_bool();
    ready_wait_sec_        = get_parameter("ready_wait_sec").as_double();
    ready_joint1_          = get_parameter("ready_joint1").as_double();
    ready_joint2_          = get_parameter("ready_joint2").as_double();
    ready_joint3_          = get_parameter("ready_joint3").as_double();
    ready_joint4_          = get_parameter("ready_joint4").as_double();
    ready_joint5_          = get_parameter("ready_joint5").as_double();
    ready_joint6_          = get_parameter("ready_joint6").as_double();
    ready_joint7_          = get_parameter("ready_joint7").as_double();
    use_grid_linear_offset_  = get_parameter("use_grid_linear_offset").as_bool();
    grid_col_                = get_parameter("grid_col").as_double();
    grid_row_                = get_parameter("grid_row").as_double();
    use_target_linear_offset_ = get_parameter("use_target_linear_offset").as_bool();
    x_offset_kx_             = get_parameter("x_offset_kx").as_double();
    x_offset_ky_             = get_parameter("x_offset_ky").as_double();
    x_offset_b_              = get_parameter("x_offset_b").as_double();
    y_offset_kx_             = get_parameter("y_offset_kx").as_double();
    y_offset_ky_             = get_parameter("y_offset_ky").as_double();
    y_offset_b_              = get_parameter("y_offset_b").as_double();
    robot_desc_wait_timeout_sec_ = get_parameter("robot_description_wait_timeout_sec").as_int();

    RCLCPP_INFO(get_logger(),
      "Params: group=%s eef=%s safe_z=%.3f hover_z=%.3f "
      "x_offset=%.3f y_offset=%.3f dry_run=%s once=%s "
      "eef_step=%.4f min_fraction=%.2f vel=%.2f acc=%.2f "
      "use_fixed_orientation=%s "
      "use_grid_linear_offset=%s grid_col=%.1f grid_row=%.1f "
      "use_target_linear_offset=%s",
      planning_group_.c_str(), eef_link_.c_str(),
      safe_z_, hover_z_, x_offset_, y_offset_,
      dry_run_ ? "true" : "false",
      once_    ? "true" : "false",
      eef_step_, min_fraction_, velocity_scale_, acceleration_scale_,
      use_fixed_orientation_ ? "true" : "false",
      use_grid_linear_offset_ ? "true" : "false",
      grid_col_, grid_row_,
      use_target_linear_offset_ ? "true" : "false");

    if (use_target_linear_offset_) {
      RCLCPP_INFO(get_logger(),
        "Target-linear coeffs: "
        "x=(kx=%.4f ky=%.4f b=%.4f) y=(kx=%.4f ky=%.4f b=%.4f)",
        x_offset_kx_, x_offset_ky_, x_offset_b_,
        y_offset_kx_, y_offset_ky_, y_offset_b_);
    }
  }

  // Must be called after executor.spin() is running in a separate thread.
  void initMoveGroup()
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
    move_group_->setPoseReferenceFrame("base");
    move_group_->setPlanningTime(5.0);
    move_group_->setMaxVelocityScalingFactor(velocity_scale_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scale_);

    RCLCPP_INFO(get_logger(), "MoveGroup created. planning_frame=%s  eef=%s",
      move_group_->getPlanningFrame().c_str(),
      move_group_->getEndEffectorLink().c_str());

    RCLCPP_INFO(get_logger(), "Waiting 2s for CurrentStateMonitor...");
    std::this_thread::sleep_for(std::chrono::seconds(2));
    RCLCPP_INFO(get_logger(), "CurrentStateMonitor warm-up done.");

    if (move_to_ready_first_) {
      RCLCPP_INFO(get_logger(), "Moving to ready pose before subscribing...");
      if (!moveToReadyPose()) {
        RCLCPP_ERROR(get_logger(),
          "Ready pose failed — YOLO subscription will NOT be created. Node idle.");
        return;
      }
      RCLCPP_INFO(get_logger(), "Ready pose reached. Waiting %.1f s...", ready_wait_sec_);
      std::this_thread::sleep_for(
        std::chrono::duration<double>(ready_wait_sec_));
    }

    sub_ = create_subscription<std_msgs::msg::String>(
      target_topic_, 10,
      std::bind(&YoloHoverLinearNode::targetCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Subscribed to %s", target_topic_.c_str());
  }

  void startWorker()
  {
    worker_thread_ = std::thread(&YoloHoverLinearNode::workerLoop, this);
  }

  ~YoloHoverLinearNode()
  {
    if (worker_thread_.joinable()) worker_thread_.join();
  }

private:
  // ── Subscription callback: parse and store target only ───────────────── //

  void targetCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (once_ && done_) {
      RCLCPP_DEBUG(get_logger(), "once=true, already done. Ignoring.");
      return;
    }
    if (processing_) {
      RCLCPP_WARN(get_logger(), "Already processing a target. Ignoring.");
      return;
    }

    double tx = 0.0, ty = 0.0, tz = 0.0;
    if (!parseXYZ(msg->data, tx, ty, tz)) {
      RCLCPP_ERROR(get_logger(), "Failed to parse x,y,z from: '%s'", msg->data.c_str());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      target_x_ = tx;
      target_y_ = ty;
      target_z_ = tz;
      has_target_ = true;
    }

    RCLCPP_INFO(get_logger(),
      "Target queued: raw(%.4f, %.4f, %.4f)  with_offset(%.4f, %.4f)  hover_z=%.4f",
      tx, ty, tz, tx + x_offset_, ty + y_offset_, hover_z_);
  }

  // ── Worker thread ─────────────────────────────────────────────────────── //

  void workerLoop()
  {
    while (rclcpp::ok()) {
      if (once_ && done_) break;

      double tx = 0.0, ty = 0.0, tz = 0.0;
      bool got = false;
      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        if (has_target_ && !processing_) {
          tx = target_x_; ty = target_y_; tz = target_z_;
          has_target_ = false;
          processing_ = true;
          got = true;
        }
      }

      if (got) {
        processTarget(tx, ty, tz);
        processing_ = false;
        if (once_) {
          done_ = true;
          RCLCPP_INFO(get_logger(), "once=true — will not move again.");
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  // ── Robot description loader ─────────────────────────────────────────── //
  // Copies robot_description / robot_description_semantic from /move_group
  // so that MoveGroupInterface can find them on this node's parameter server.

  bool fetchRobotDescription()
  {
    RCLCPP_INFO(get_logger(),
      "Fetching robot_description from /move_group (timeout=%d s)...",
      robot_desc_wait_timeout_sec_);
    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(
      shared_from_this(), "/move_group");

    bool ready = false;
    for (int i = 0; i < robot_desc_wait_timeout_sec_ * 2 && rclcpp::ok(); ++i) {
      if (param_client->service_is_ready()) { ready = true; break; }
      if (i % 2 == 0) {
        RCLCPP_INFO(get_logger(),
          "Waiting for /move_group parameter service... (%d/%d s)",
          i / 2, robot_desc_wait_timeout_sec_);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (!ready) {
      RCLCPP_ERROR(get_logger(),
        "MoveIt /move_group is not running or not ready after %d s.\n"
        "Please run MoveIt in a separate terminal FIRST:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source ~/ros2_ws/install/setup.bash\n"
        "  ros2 launch franka_fr3_moveit_config moveit.launch.py "
        "robot_ip:=172.16.0.2 robot_type:=fr3 load_gripper:=true\n"
        "Wait for 'You can start planning now!' then retry this node.",
        robot_desc_wait_timeout_sec_);
      return false;
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

  // ── Ready pose ───────────────────────────────────────────────────────── //

  bool moveToReadyPose()
  {
    std::map<std::string, double> ready_joint_values;
    ready_joint_values["fr3_joint1"] = ready_joint1_;
    ready_joint_values["fr3_joint2"] = ready_joint2_;
    ready_joint_values["fr3_joint3"] = ready_joint3_;
    ready_joint_values["fr3_joint4"] = ready_joint4_;
    ready_joint_values["fr3_joint5"] = ready_joint5_;
    ready_joint_values["fr3_joint6"] = ready_joint6_;
    ready_joint_values["fr3_joint7"] = ready_joint7_;

    RCLCPP_INFO(get_logger(),
      "Ready joints: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
      ready_joint1_, ready_joint2_, ready_joint3_, ready_joint4_,
      ready_joint5_, ready_joint6_, ready_joint7_);

    move_group_->setJointValueTarget(ready_joint_values);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_->plan(plan));

    if (!success) {
      RCLCPP_ERROR(get_logger(), "Failed to plan to ready pose");
      return false;
    }

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "[dry_run] Ready pose plan succeeded, skipping execute");
      return true;
    }

    auto result = move_group_->execute(plan);
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Failed to execute ready pose");
      return false;
    }

    RCLCPP_INFO(get_logger(), "Ready pose reached successfully.");
    return true;
  }

  // ── Safety ────────────────────────────────────────────────────────────── //

  static constexpr double WS_X_MIN = -0.2, WS_X_MAX = 0.9;
  static constexpr double WS_Y_MIN = -0.8, WS_Y_MAX = 0.8;
  static constexpr double WS_Z_MIN = -0.1, WS_Z_MAX = 0.8;

  bool checkWorkspaceBounds(double gx, double gy, double gz)
  {
    if (gx < WS_X_MIN || gx > WS_X_MAX ||
        gy < WS_Y_MIN || gy > WS_Y_MAX ||
        gz < WS_Z_MIN || gz > WS_Z_MAX)
    {
      RCLCPP_ERROR(get_logger(),
        "Target (%.4f, %.4f, %.4f) is outside workspace "
        "x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f]. Aborting.",
        gx, gy, gz,
        WS_X_MIN, WS_X_MAX, WS_Y_MIN, WS_Y_MAX, WS_Z_MIN, WS_Z_MAX);
      return false;
    }
    return true;
  }

  // ── Per-phase Cartesian helper ─────────────────────────────────────────── //

  bool computeAndExecute(
    const std::vector<geometry_msgs::msg::Pose> & waypoints,
    const std::string & phase_name)
  {
    moveit_msgs::msg::RobotTrajectory traj;
    double fraction = move_group_->computeCartesianPath(
      waypoints, eef_step_, jump_threshold_, traj);

    RCLCPP_INFO(get_logger(), "[%s] Cartesian fraction: %.3f", phase_name.c_str(), fraction);

    if (fraction < min_fraction_) {
      RCLCPP_ERROR(get_logger(),
        "[%s] fraction %.3f < %.2f — aborting.", phase_name.c_str(), fraction, min_fraction_);
      return false;
    }

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "[%s] dry_run=true — skipping execute.", phase_name.c_str());
      return true;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = traj;
    auto result = move_group_->execute(plan);

    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(),
        "[%s] Execute failed (code=%d).", phase_name.c_str(), result.val);
      return false;
    }

    RCLCPP_INFO(get_logger(), "[%s] Execute success.", phase_name.c_str());
    return true;
  }

  // ── 3-phase Cartesian motion ──────────────────────────────────────────── //

  void processTarget(double tx, double ty, double tz)
  {
    (void)tz;

    geometry_msgs::msg::Pose current = move_group_->getCurrentPose(eef_link_).pose;

    RCLCPP_INFO(get_logger(),
      "Current EEF (%s): x=%.4f  y=%.4f  z=%.4f",
      eef_link_.c_str(), current.position.x, current.position.y, current.position.z);

    // Safety: reject (0, 0, 0) — CurrentStateMonitor not ready.
    if (std::abs(current.position.x) < 1e-6 &&
        std::abs(current.position.y) < 1e-6 &&
        std::abs(current.position.z) < 1e-6)
    {
      RCLCPP_ERROR(get_logger(),
        "Current EEF pose is (0, 0, 0). "
        "CurrentStateMonitor has not received /joint_states yet. Aborting.");
      return;
    }

    double final_x_offset = x_offset_;
    double final_y_offset = y_offset_;

    if (use_target_linear_offset_) {
      final_x_offset = x_offset_kx_ * tx + x_offset_ky_ * ty + x_offset_b_;
      final_y_offset = y_offset_kx_ * tx + y_offset_ky_ * ty + y_offset_b_;
      RCLCPP_INFO(get_logger(),
        "Target-linear offset enabled: target=(%.4f, %.4f), offset=(%.4f, %.4f)",
        tx, ty, final_x_offset, final_y_offset);
    } else if (use_grid_linear_offset_) {
      final_x_offset = 0.125 * grid_col_ - 0.1183 * grid_row_ + 0.2767;
      final_y_offset = 0.1067 * grid_col_ + 0.1233 * grid_row_ + 0.0428;
      RCLCPP_INFO(get_logger(),
        "Grid offset: col=%.1f row=%.1f → x_offset=%.4f y_offset=%.4f",
        grid_col_, grid_row_, final_x_offset, final_y_offset);
    } else {
      RCLCPP_INFO(get_logger(),
        "Fixed offset: offset=(%.4f, %.4f)", final_x_offset, final_y_offset);
    }

    double gx = tx + final_x_offset;
    double gy = ty + final_y_offset;

    RCLCPP_INFO(get_logger(),
      "target raw=(%.4f, %.4f, %.4f)  x_offset=%.4f y_offset=%.4f  "
      "with_offset=(%.4f, %.4f)  hover_z=%.4f",
      tx, ty, tz, final_x_offset, final_y_offset, gx, gy, hover_z_);

    if (!checkWorkspaceBounds(gx, gy, hover_z_)) return;

    // Resolve orientation.
    geometry_msgs::msg::Quaternion ori;
    if (use_fixed_orientation_) {
      ori.x = fixed_qx_; ori.y = fixed_qy_;
      ori.z = fixed_qz_; ori.w = fixed_qw_;
      RCLCPP_INFO(get_logger(),
        "Using fixed orientation q=(x=%.4f, y=%.4f, z=%.4f, w=%.4f)",
        ori.x, ori.y, ori.z, ori.w);
    } else {
      ori = current.orientation;
      RCLCPP_INFO(get_logger(),
        "Using current EEF orientation q=(x=%.4f, y=%.4f, z=%.4f, w=%.4f)",
        ori.x, ori.y, ori.z, ori.w);
    }

    // Phase 1: rise to safe_z (x, y unchanged)
    geometry_msgs::msg::Pose p1 = current;
    p1.position.z  = safe_z_;
    p1.orientation = ori;

    RCLCPP_INFO(get_logger(),
      "[Phase1] (%.4f, %.4f, %.4f) → z=%.4f",
      current.position.x, current.position.y, current.position.z, safe_z_);

    if (!computeAndExecute({p1}, "Phase1_RiseToSafeZ")) return;

    // Phase 2: move x,y at safe_z
    geometry_msgs::msg::Pose p2 = p1;
    p2.position.x  = gx;
    p2.position.y  = gy;
    p2.orientation = ori;

    RCLCPP_INFO(get_logger(),
      "[Phase2] (%.4f, %.4f) → (%.4f, %.4f) at z=%.4f",
      p1.position.x, p1.position.y, gx, gy, safe_z_);

    if (!computeAndExecute({p2}, "Phase2_MoveXY")) return;

    // Phase 3: descend to hover_z (x, y fixed)
    geometry_msgs::msg::Pose p3 = p2;
    p3.position.z  = hover_z_;
    p3.orientation = ori;

    RCLCPP_INFO(get_logger(),
      "[Phase3] z=%.4f → z=%.4f at (%.4f, %.4f)",
      safe_z_, hover_z_, gx, gy);

    if (!computeAndExecute({p3}, "Phase3_DescendToHoverZ")) return;

    RCLCPP_INFO(get_logger(),
      "All 3 phases complete. Final EEF: (%.4f, %.4f, %.4f).",
      gx, gy, hover_z_);
  }

  // ── Members ──────────────────────────────────────────────────────────── //

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  std::string target_topic_;
  std::string planning_group_;
  std::string eef_link_;
  double safe_z_;
  double hover_z_;
  double x_offset_;
  double y_offset_;
  bool   dry_run_;
  bool   once_;
  double eef_step_;
  double jump_threshold_;
  double min_fraction_;
  double velocity_scale_;
  double acceleration_scale_;
  bool   use_fixed_orientation_;
  double fixed_qx_, fixed_qy_, fixed_qz_, fixed_qw_;
  bool   move_to_ready_first_;
  double ready_wait_sec_;
  double ready_joint1_, ready_joint2_, ready_joint3_, ready_joint4_;
  double ready_joint5_, ready_joint6_, ready_joint7_;
  bool   use_grid_linear_offset_;
  double grid_col_;
  double grid_row_;
  bool   use_target_linear_offset_;
  double x_offset_kx_, x_offset_ky_, x_offset_b_;
  double y_offset_kx_, y_offset_ky_, y_offset_b_;
  int    robot_desc_wait_timeout_sec_;

  std::mutex        target_mutex_;
  std::atomic<bool> has_target_;
  std::atomic<bool> processing_;
  std::atomic<bool> done_;
  double            target_x_, target_y_, target_z_;

  std::thread worker_thread_;
};

// ── main ─────────────────────────────────────────────────────────────────── //

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<YoloHoverLinearNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);

  std::thread spin_thread([&executor]() { executor.spin(); });

  node->initMoveGroup();
  node->startWorker();

  spin_thread.join();
  rclcpp::shutdown();
  return 0;
}
