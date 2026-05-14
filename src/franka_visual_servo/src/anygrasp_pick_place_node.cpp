#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <franka_msgs/action/grasp.hpp>
#include <franka_msgs/action/move.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <regex>
#include <string>
#include <thread>
#include <vector>

using FrankaMove = franka_msgs::action::Move;
using FrankaGrasp = franka_msgs::action::Grasp;

struct AnyGraspTarget
{
  double score{0.0};
  double tx{0.0};
  double ty{0.0};
  double tz{0.0};
  double hx{0.0};
  double hy{0.0};
  double hz{0.0};
  double width{0.0};
  std::string source_frame_id{"camera_color_optical_frame"};
  Eigen::Matrix3d rotation{Eigen::Matrix3d::Identity()};
};

class AnyGraspPickPlaceNode : public rclcpp::Node
{
public:
  AnyGraspPickPlaceNode()
  : Node("anygrasp_pick_place_node")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/anygrasp/best_safe_grasp_pose_base");
    target_frame_ = declare_parameter<std::string>("target_frame", "base");
    planning_group_ = declare_parameter<std::string>("planning_group", "fr3_arm");
    eef_link_ = declare_parameter<std::string>("end_effector_link", "fr3_hand_tcp");
    gripper_move_action_ =
      declare_parameter<std::string>("gripper_move_action", "/franka_gripper/move");
    gripper_grasp_action_ =
      declare_parameter<std::string>("gripper_grasp_action", "/franka_gripper/grasp");

    dry_run_ = declare_parameter<bool>("dry_run", true);
    plan_only_ = declare_parameter<bool>("plan_only", true);
    execute_ = declare_parameter<bool>("execute", false);
    once_ = declare_parameter<bool>("once", true);
    do_descent_ = declare_parameter<bool>("do_descent", false);
    do_preopen_gripper_ = declare_parameter<bool>("do_preopen_gripper", true);
    do_gripper_ = declare_parameter<bool>("do_gripper", false);
    do_place_ = declare_parameter<bool>("do_place", false);
    use_camera_view_offset_ = declare_parameter<bool>("use_camera_view_offset", false);
    camera_view_x_offset_ = declare_parameter<double>("camera_view_x_offset", 0.0);
    camera_view_y_offset_ = declare_parameter<double>("camera_view_y_offset", 0.0);
    use_target_smoothing_ = declare_parameter<bool>("use_target_smoothing", false);
    smoothing_window_ = declare_parameter<int>("smoothing_window", 5);
    smoothing_method_ = declare_parameter<std::string>("smoothing_method", "median");
    require_stable_target_ = declare_parameter<bool>("require_stable_target", true);
    max_target_std_xy_ = declare_parameter<double>("max_target_std_xy", 0.010);
    max_target_std_z_ = declare_parameter<double>("max_target_std_z", 0.015);
    stable_target_timeout_sec_ = declare_parameter<double>("stable_target_timeout_sec", 5.0);
    min_smoothing_samples_ = declare_parameter<int>("min_smoothing_samples", 3);

    use_fixed_orientation_ = declare_parameter<bool>("use_fixed_orientation", true);
    fixed_qx_ = declare_parameter<double>("fixed_qx", 0.0108);
    fixed_qy_ = declare_parameter<double>("fixed_qy", -0.0128);
    fixed_qz_ = declare_parameter<double>("fixed_qz", -0.0037);
    fixed_qw_ = declare_parameter<double>("fixed_qw", 0.9999);

    use_anygrasp_orientation_ = declare_parameter<bool>("use_anygrasp_orientation", false);
    anygrasp_orientation_mode_ =
      declare_parameter<std::string>("anygrasp_orientation_mode", "off");
    planar_yaw_axis_index_ = declare_parameter<int>("planar_yaw_axis_index", 0);
    planar_yaw_offset_ = declare_parameter<double>("planar_yaw_offset", 0.0);
    topdown_orientation_source_ =
      declare_parameter<std::string>("topdown_orientation_source", "current_at_start");
    topdown_qx_ = declare_parameter<double>("topdown_qx", 0.0);
    topdown_qy_ = declare_parameter<double>("topdown_qy", 0.0);
    topdown_qz_ = declare_parameter<double>("topdown_qz", 0.0);
    topdown_qw_ = declare_parameter<double>("topdown_qw", 1.0);
    grasp_to_tcp_roll_ = declare_parameter<double>("grasp_to_tcp_roll", 0.0);
    grasp_to_tcp_pitch_ = declare_parameter<double>("grasp_to_tcp_pitch", 0.0);
    grasp_to_tcp_yaw_ = declare_parameter<double>("grasp_to_tcp_yaw", 0.0);
    orientation_test_mode_ = declare_parameter<bool>("orientation_test_mode", false);
    normalize_quaternion_ = declare_parameter<bool>("normalize_quaternion", true);
    print_anygrasp_rpy_ = declare_parameter<bool>("print_anygrasp_rpy", true);
    keep_grasp_orientation_during_place_ =
      declare_parameter<bool>("keep_grasp_orientation_during_place", true);

    safe_z_ = declare_parameter<double>("safe_z", 0.55);
    grasp_z_offset_ = declare_parameter<double>("grasp_z_offset", 0.0);
    lift_z_ = declare_parameter<double>("lift_z", 0.55);
    place_x_ = declare_parameter<double>("place_x", 0.45);
    place_y_ = declare_parameter<double>("place_y", -0.30);
    place_z_ = declare_parameter<double>("place_z", 0.35);
    place_safe_z_ = declare_parameter<double>("place_safe_z", 0.55);

    min_score_ = declare_parameter<double>("min_score", 0.2);
    x_min_ = declare_parameter<double>("x_min", 0.20);
    x_max_ = declare_parameter<double>("x_max", 0.70);
    y_min_ = declare_parameter<double>("y_min", -0.50);
    y_max_ = declare_parameter<double>("y_max", 0.50);
    z_min_ = declare_parameter<double>("z_min", 0.05);
    z_max_ = declare_parameter<double>("z_max", 0.80);

    cartesian_eef_step_ = declare_parameter<double>("cartesian_eef_step", 0.01);
    cartesian_jump_threshold_ = declare_parameter<double>("cartesian_jump_threshold", 0.0);
    min_cartesian_fraction_ = declare_parameter<double>("min_cartesian_fraction", 0.95);
    velocity_scaling_ = declare_parameter<double>("velocity_scaling", 0.05);
    acceleration_scaling_ = declare_parameter<double>("acceleration_scaling", 0.05);
    robot_desc_timeout_sec_ = declare_parameter<int>("robot_desc_timeout_sec", 30);

    initial_j1_ = declare_parameter<double>("initial_j1", 0.0);
    initial_j2_ = declare_parameter<double>("initial_j2", -0.785);
    initial_j3_ = declare_parameter<double>("initial_j3", 0.0);
    initial_j4_ = declare_parameter<double>("initial_j4", -2.356);
    initial_j5_ = declare_parameter<double>("initial_j5", 0.0);
    initial_j6_ = declare_parameter<double>("initial_j6", 1.571);
    initial_j7_ = declare_parameter<double>("initial_j7", 0.785);

    gripper_open_width_ = declare_parameter<double>("gripper_open_width", 0.08);
    gripper_close_width_ = declare_parameter<double>("gripper_close_width", 0.025);
    gripper_speed_ = declare_parameter<double>("gripper_speed", 0.03);
    gripper_force_ = declare_parameter<double>("gripper_force", 20.0);
    gripper_epsilon_inner_ = declare_parameter<double>("gripper_epsilon_inner", 0.005);
    gripper_epsilon_outer_ = declare_parameter<double>("gripper_epsilon_outer", 0.005);

    grasp_z_ = declare_parameter<double>("grasp_z", 0.12);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    if (orientation_test_mode_) {
      RCLCPP_WARN(get_logger(),
        "orientation_test_mode=true: descent, gripper, and place are disabled.");
      do_descent_ = false;
      do_preopen_gripper_ = false;
      do_gripper_ = false;
      do_place_ = false;
    }

    if (use_anygrasp_orientation_ && anygrasp_orientation_mode_ == "off") {
      RCLCPP_WARN(get_logger(),
        "use_anygrasp_orientation=true with anygrasp_orientation_mode=off; "
        "using legacy full orientation mode.");
      anygrasp_orientation_mode_ = "full";
    }
  }

  ~AnyGraspPickPlaceNode() override
  {
    stop_worker_.store(true);
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }

  void initMoveGroup()
  {
    if (!fetchRobotDescription()) {
      rclcpp::shutdown();
      return;
    }

    try {
      move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), planning_group_);
    } catch (const std::exception & e) {
      RCLCPP_FATAL(get_logger(), "MoveGroupInterface construction failed: %s", e.what());
      rclcpp::shutdown();
      return;
    }

    move_group_->setEndEffectorLink(eef_link_);
    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);
    move_group_->setPlanningTime(10.0);
    move_group_->startStateMonitor(5.0);
    initializeTopdownOrientation();

    move_client_ = rclcpp_action::create_client<FrankaMove>(
      shared_from_this(), gripper_move_action_);
    grasp_client_ = rclcpp_action::create_client<FrankaGrasp>(
      shared_from_this(), gripper_grasp_action_);

    sub_ = create_subscription<std_msgs::msg::String>(
      input_topic_, rclcpp::QoS(10),
      std::bind(&AnyGraspPickPlaceNode::callback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "AnyGrasp pick/place ready. topic=%s group=%s eef=%s",
      input_topic_.c_str(), planning_group_.c_str(), eef_link_.c_str());
    RCLCPP_INFO(get_logger(),
      "dry_run=%s plan_only=%s execute=%s once=%s do_descent=%s "
      "do_preopen_gripper=%s do_gripper=%s do_place=%s",
      tf(dry_run_), tf(plan_only_), tf(execute_), tf(once_),
      tf(do_descent_), tf(do_preopen_gripper_), tf(do_gripper_), tf(do_place_));
    RCLCPP_INFO(get_logger(), "gripper_move_action=%s", gripper_move_action_.c_str());
    RCLCPP_INFO(get_logger(), "gripper_grasp_action=%s", gripper_grasp_action_.c_str());
    RCLCPP_INFO(get_logger(),
      "gripper open_width=%.3f close_width=%.3f speed=%.3f force=%.3f epsilon=(%.4f, %.4f)",
      gripper_open_width_, gripper_close_width_, gripper_speed_, gripper_force_,
      gripper_epsilon_inner_, gripper_epsilon_outer_);
    RCLCPP_INFO(get_logger(),
      "use_camera_view_offset=%s camera_view_x_offset=%.4f camera_view_y_offset=%.4f target_frame=%s",
      tf(use_camera_view_offset_), camera_view_x_offset_, camera_view_y_offset_,
      target_frame_.c_str());
    RCLCPP_INFO(get_logger(),
      "target smoothing: enabled=%s window=%d method=%s require_stable=%s "
      "max_std_xy=%.4f max_std_z=%.4f timeout=%.2f min_samples=%d",
      tf(use_target_smoothing_), smoothing_window_, smoothing_method_.c_str(),
      tf(require_stable_target_), max_target_std_xy_, max_target_std_z_,
      stable_target_timeout_sec_, min_smoothing_samples_);
    RCLCPP_INFO(get_logger(),
      "anygrasp_orientation_mode=%s legacy_use_anygrasp_orientation=%s "
      "orientation_test_mode=%s grasp_to_tcp_rpy=(%.3f, %.3f, %.3f)",
      anygrasp_orientation_mode_.c_str(), tf(use_anygrasp_orientation_), tf(orientation_test_mode_),
      grasp_to_tcp_roll_, grasp_to_tcp_pitch_, grasp_to_tcp_yaw_);
    RCLCPP_INFO(get_logger(),
      "planar_yaw_axis_index=%d planar_yaw_offset=%.4f topdown_source=%s",
      planar_yaw_axis_index_, planar_yaw_offset_, topdown_orientation_source_.c_str());
    RCLCPP_INFO(get_logger(),
      "initial joint target=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
      initial_j1_, initial_j2_, initial_j3_, initial_j4_,
      initial_j5_, initial_j6_, initial_j7_);
  }

  void startWorker()
  {
    worker_thread_ = std::thread([this]() { workerLoop(); });
  }

private:
  static const char * tf(bool v) { return v ? "true" : "false"; }

  static std::optional<double> extractNumberAfterKey(
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

  static std::optional<std::string> objectTail(
    const std::string & text, const std::string & key)
  {
    const auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) {
      return std::nullopt;
    }
    return text.substr(pos);
  }

  static std::optional<std::string> extractStringAfterKey(
    const std::string & text, const std::string & key)
  {
    const std::string pattern = "\"" + key + "\"\\s*:\\s*\"([^\"]+)\"";
    std::regex re(pattern);
    std::smatch m;
    if (std::regex_search(text, m, re)) {
      return m[1].str();
    }
    return std::nullopt;
  }

  static std::optional<std::vector<double>> extractNumbers(
    const std::string & text, size_t count)
  {
    static const std::regex number_re("-?[0-9]+\\.?[0-9]*(?:[eE][-+]?[0-9]+)?");
    std::vector<double> values;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), number_re);
      it != std::sregex_iterator() && values.size() < count; ++it)
    {
      values.push_back(std::stod((*it)[0].str()));
    }
    if (values.size() != count) {
      return std::nullopt;
    }
    return values;
  }

  std::optional<AnyGraspTarget> parseAnyGraspJson(const std::string & text)
  {
    AnyGraspTarget target;
    auto score = extractNumberAfterKey(text, "score");
    if (!score) {
      RCLCPP_WARN(get_logger(), "JSON parse failed: score missing.");
      return std::nullopt;
    }
    target.score = *score;
    target.width = extractNumberAfterKey(text, "width").value_or(0.0);
    if (const auto source_frame = extractStringAfterKey(text, "source_frame_id")) {
      target.source_frame_id = *source_frame;
    } else if (const auto frame_id = extractStringAfterKey(text, "frame_id")) {
      target.source_frame_id = *frame_id;
    }

    const auto translation_tail = objectTail(text, "translation");
    const auto hover_tail = objectTail(text, "hover_translation");
    if (!translation_tail || !hover_tail) {
      RCLCPP_WARN(get_logger(), "JSON parse failed: translation or hover_translation missing.");
      return std::nullopt;
    }

    auto tx = extractNumberAfterKey(*translation_tail, "x");
    auto ty = extractNumberAfterKey(*translation_tail, "y");
    auto tz = extractNumberAfterKey(*translation_tail, "z");
    auto hx = extractNumberAfterKey(*hover_tail, "x");
    auto hy = extractNumberAfterKey(*hover_tail, "y");
    auto hz = extractNumberAfterKey(*hover_tail, "z");
    if (!tx || !ty || !tz || !hx || !hy || !hz) {
      RCLCPP_WARN(get_logger(), "JSON parse failed: translation numbers missing.");
      return std::nullopt;
    }
    target.tx = *tx;
    target.ty = *ty;
    target.tz = *tz;
    target.hx = *hx;
    target.hy = *hy;
    target.hz = *hz;

    const auto rpos = text.find("\"rotation_matrix\"");
    if (rpos == std::string::npos) {
      RCLCPP_WARN(get_logger(), "JSON parse failed: rotation_matrix missing.");
      return std::nullopt;
    }
    const std::string rtail = text.substr(rpos);
    const auto nums = extractNumbers(rtail, 9);
    if (!nums) {
      RCLCPP_WARN(get_logger(), "JSON parse failed: rotation_matrix needs 9 numbers.");
      return std::nullopt;
    }

    target.rotation <<
      (*nums)[0], (*nums)[1], (*nums)[2],
      (*nums)[3], (*nums)[4], (*nums)[5],
      (*nums)[6], (*nums)[7], (*nums)[8];

    return target;
  }

  void callback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (once_ && done_.load()) {
      return;
    }
    const auto parsed = parseAnyGraspJson(msg->data);
    if (!parsed) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      latest_target_ = *parsed;
      has_target_ = true;
      if (target_buffer_.empty()) {
        smoothing_started_ = std::chrono::steady_clock::now();
      }
      target_buffer_.push_back(*parsed);
      const size_t window = static_cast<size_t>(std::max(1, smoothing_window_));
      while (target_buffer_.size() > window) {
        target_buffer_.pop_front();
      }
    }
    RCLCPP_INFO(get_logger(),
      "AnyGrasp target received: score=%.4f translation=(%.3f, %.3f, %.3f) hover=(%.3f, %.3f, %.3f)",
      parsed->score, parsed->tx, parsed->ty, parsed->tz,
      parsed->hx, parsed->hy, parsed->hz);
    RCLCPP_INFO(get_logger(), "target smoothing buffer count=%zu", target_buffer_.size());
  }

  void workerLoop()
  {
    while (rclcpp::ok() && !stop_worker_.load()) {
      if (once_ && done_.load()) {
        break;
      }

      std::optional<AnyGraspTarget> target = nextTargetForExecution();

      if (target) {
        executeSequence(*target);
        processing_.store(false);
        if (once_) {
          done_.store(true);
          RCLCPP_INFO(get_logger(), "once=true: pick/place node is done.");
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  static double meanOf(const std::vector<double> & values)
  {
    if (values.empty()) {
      return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
  }

  static double medianOf(std::vector<double> values)
  {
    if (values.empty()) {
      return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
      return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
  }

  static double stddevOf(const std::vector<double> & values, double mean)
  {
    if (values.size() < 2) {
      return 0.0;
    }
    double accum = 0.0;
    for (const double v : values) {
      const double d = v - mean;
      accum += d * d;
    }
    return std::sqrt(accum / static_cast<double>(values.size()));
  }

  std::optional<AnyGraspTarget> selectSmoothedTarget(
    const std::deque<AnyGraspTarget> & buffer)
  {
    RCLCPP_INFO(get_logger(), "smoothing enabled: buffer count=%zu", buffer.size());
    if (buffer.size() < static_cast<size_t>(std::max(1, min_smoothing_samples_))) {
      RCLCPP_INFO(get_logger(),
        "smoothing waiting: need at least %d samples, have %zu",
        min_smoothing_samples_, buffer.size());
      return std::nullopt;
    }

    std::vector<double> tx, ty, tz, hx, hy, hz, widths;
    tx.reserve(buffer.size());
    ty.reserve(buffer.size());
    tz.reserve(buffer.size());
    hx.reserve(buffer.size());
    hy.reserve(buffer.size());
    hz.reserve(buffer.size());
    widths.reserve(buffer.size());

    size_t best_idx = 0;
    double best_score = buffer.front().score;
    for (size_t i = 0; i < buffer.size(); ++i) {
      const auto & t = buffer[i];
      RCLCPP_INFO(get_logger(),
        "smoothing sample[%zu]: score=%.4f target=(%.4f, %.4f, %.4f) hover=(%.4f, %.4f, %.4f)",
        i, t.score, t.tx, t.ty, t.tz, t.hx, t.hy, t.hz);
      tx.push_back(t.tx);
      ty.push_back(t.ty);
      tz.push_back(t.tz);
      hx.push_back(t.hx);
      hy.push_back(t.hy);
      hz.push_back(t.hz);
      widths.push_back(t.width);
      if (t.score > best_score) {
        best_score = t.score;
        best_idx = i;
      }
    }

    const double mean_tx = meanOf(tx);
    const double mean_ty = meanOf(ty);
    const double mean_tz = meanOf(tz);
    const double mean_hx = meanOf(hx);
    const double mean_hy = meanOf(hy);
    const double mean_hz = meanOf(hz);

    const double med_tx = medianOf(tx);
    const double med_ty = medianOf(ty);
    const double med_tz = medianOf(tz);
    const double med_hx = medianOf(hx);
    const double med_hy = medianOf(hy);
    const double med_hz = medianOf(hz);

    const double std_x = stddevOf(tx, mean_tx);
    const double std_y = stddevOf(ty, mean_ty);
    const double std_z = stddevOf(tz, mean_tz);

    RCLCPP_INFO(get_logger(),
      "smoothing mean target=(%.4f, %.4f, %.4f) hover=(%.4f, %.4f, %.4f)",
      mean_tx, mean_ty, mean_tz, mean_hx, mean_hy, mean_hz);
    RCLCPP_INFO(get_logger(),
      "smoothing median target=(%.4f, %.4f, %.4f) hover=(%.4f, %.4f, %.4f)",
      med_tx, med_ty, med_tz, med_hx, med_hy, med_hz);
    RCLCPP_INFO(get_logger(),
      "smoothing std: std_x=%.5f std_y=%.5f std_z=%.5f",
      std_x, std_y, std_z);

    if (require_stable_target_ &&
      (std_x > max_target_std_xy_ || std_y > max_target_std_xy_ || std_z > max_target_std_z_))
    {
      RCLCPP_WARN(get_logger(),
        "smoothing rejected: std_x/std_y/std_z=(%.5f, %.5f, %.5f), limits xy=%.5f z=%.5f",
        std_x, std_y, std_z, max_target_std_xy_, max_target_std_z_);
      return std::nullopt;
    }

    AnyGraspTarget selected = buffer[best_idx];
    if (smoothing_method_ == "mean") {
      selected.tx = mean_tx;
      selected.ty = mean_ty;
      selected.tz = mean_tz;
      selected.hx = mean_hx;
      selected.hy = mean_hy;
      selected.hz = mean_hz;
      selected.width = meanOf(widths);
    } else {
      if (smoothing_method_ != "median") {
        RCLCPP_WARN(get_logger(),
          "Unsupported smoothing_method='%s'; using median.",
          smoothing_method_.c_str());
      }
      selected.tx = med_tx;
      selected.ty = med_ty;
      selected.tz = med_tz;
      selected.hx = med_hx;
      selected.hy = med_hy;
      selected.hz = med_hz;
      selected.width = medianOf(widths);
    }
    selected.rotation = buffer[best_idx].rotation;
    selected.source_frame_id = buffer[best_idx].source_frame_id;
    selected.score = best_score;

    RCLCPP_INFO(get_logger(),
      "smoothing stable: selected target=(%.4f, %.4f, %.4f) hover=(%.4f, %.4f, %.4f)",
      selected.tx, selected.ty, selected.tz, selected.hx, selected.hy, selected.hz);
    RCLCPP_INFO(get_logger(),
      "selected rotation source index=%zu score=%.4f source_frame=%s",
      best_idx, best_score, selected.source_frame_id.c_str());
    return selected;
  }

  std::optional<AnyGraspTarget> nextTargetForExecution()
  {
    std::deque<AnyGraspTarget> buffer_copy;
    std::optional<AnyGraspTarget> immediate_target;
    bool has_immediate = false;
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      if (processing_.load()) {
        return std::nullopt;
      }
      if (!use_target_smoothing_) {
        if (has_target_) {
          immediate_target = latest_target_;
          has_target_ = false;
          processing_.store(true);
        }
        return immediate_target;
      }

      buffer_copy = target_buffer_;
      has_immediate = !target_buffer_.empty();
    }

    if (!use_target_smoothing_) {
      return immediate_target;
    }

    if (!has_immediate) {
      return std::nullopt;
    }

    const auto selected = selectSmoothedTarget(buffer_copy);
    if (selected) {
      std::lock_guard<std::mutex> lock(target_mutex_);
      target_buffer_.clear();
      has_target_ = false;
      processing_.store(true);
      return selected;
    }

    const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - smoothing_started_).count();
    if (stable_target_timeout_sec_ > 0.0 && elapsed > stable_target_timeout_sec_) {
      RCLCPP_ERROR(get_logger(),
        "stable target timeout: %.2f sec elapsed without stable target. Aborting.",
        elapsed);
      std::lock_guard<std::mutex> lock(target_mutex_);
      target_buffer_.clear();
      has_target_ = false;
      if (once_) {
        done_.store(true);
      }
    }
    return std::nullopt;
  }

  bool finitePose(double x, double y, double z) const
  {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }

  bool inWorkspace(double x, double y, double z, const std::string & name) const
  {
    if (!finitePose(x, y, z)) {
      RCLCPP_ERROR(get_logger(), "%s rejected: NaN/inf position.", name.c_str());
      return false;
    }
    if (x < x_min_ || x > x_max_ || y < y_min_ || y > y_max_ || z < z_min_ || z > z_max_) {
      RCLCPP_ERROR(get_logger(),
        "%s rejected: pos=(%.3f, %.3f, %.3f) outside x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f]",
        name.c_str(), x, y, z, x_min_, x_max_, y_min_, y_max_, z_min_, z_max_);
      return false;
    }
    return true;
  }

  geometry_msgs::msg::Quaternion normalize(
    const geometry_msgs::msg::Quaternion & q) const
  {
    geometry_msgs::msg::Quaternion out = q;
    const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n > 1e-9 && normalize_quaternion_) {
      out.x /= n;
      out.y /= n;
      out.z /= n;
      out.w /= n;
    }
    return out;
  }

  geometry_msgs::msg::Quaternion fixedOrientation() const
  {
    geometry_msgs::msg::Quaternion q;
    q.x = fixed_qx_;
    q.y = fixed_qy_;
    q.z = fixed_qz_;
    q.w = fixed_qw_;
    return normalize(q);
  }

  geometry_msgs::msg::Quaternion topdownParamOrientation() const
  {
    geometry_msgs::msg::Quaternion q;
    q.x = topdown_qx_;
    q.y = topdown_qy_;
    q.z = topdown_qz_;
    q.w = topdown_qw_;
    return normalize(q);
  }

  Eigen::Matrix3d msgQuatToMatrix(const geometry_msgs::msg::Quaternion & q_msg) const
  {
    Eigen::Quaterniond q(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
    if (normalize_quaternion_) {
      q.normalize();
    }
    return q.toRotationMatrix();
  }

  // RPY convention used here is R = Rz(yaw) * Ry(pitch) * Rx(roll).
  static Eigen::Matrix3d matrixFromRpy(double roll, double pitch, double yaw)
  {
    const Eigen::AngleAxisd rx(roll, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd ry(pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd rz(yaw, Eigen::Vector3d::UnitZ());
    return (rz * ry * rx).toRotationMatrix();
  }

  geometry_msgs::msg::Quaternion eigenToMsg(const Eigen::Quaterniond & q_in) const
  {
    Eigen::Quaterniond q = q_in;
    if (normalize_quaternion_) {
      q.normalize();
    }
    geometry_msgs::msg::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
  }

  static Eigen::Vector3d rpyFromMatrix(const Eigen::Matrix3d & r)
  {
    const double roll = std::atan2(r(2, 1), r(2, 2));
    const double pitch = std::atan2(
      -r(2, 0), std::sqrt(r(2, 1) * r(2, 1) + r(2, 2) * r(2, 2)));
    const double yaw = std::atan2(r(1, 0), r(0, 0));
    return Eigen::Vector3d(roll, pitch, yaw);
  }

  bool quaternionSafe(const geometry_msgs::msg::Quaternion & q, const std::string & name) const
  {
    const double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (!std::isfinite(n) || n < 0.5) {
      RCLCPP_ERROR(get_logger(), "%s rejected: quaternion norm %.6f is invalid.", name.c_str(), n);
      return false;
    }
    return true;
  }

  void initializeTopdownOrientation()
  {
    if (topdown_orientation_source_ == "params") {
      topdown_orientation_ = topdownParamOrientation();
      topdown_orientation_ready_ = quaternionSafe(topdown_orientation_, "topdown params orientation");
    } else if (topdown_orientation_source_ == "current_at_start") {
      try {
        topdown_orientation_ = normalize(move_group_->getCurrentPose(eef_link_).pose.orientation);
        topdown_orientation_ready_ =
          quaternionSafe(topdown_orientation_, "topdown current_at_start orientation");
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(),
          "Failed to read current TCP orientation for topdown source: %s", e.what());
        topdown_orientation_ = topdownParamOrientation();
        topdown_orientation_ready_ =
          quaternionSafe(topdown_orientation_, "fallback topdown params orientation");
      }
    } else {
      RCLCPP_ERROR(get_logger(),
        "Unsupported topdown_orientation_source='%s'. Use current_at_start or params.",
        topdown_orientation_source_.c_str());
      topdown_orientation_ = topdownParamOrientation();
      topdown_orientation_ready_ =
        quaternionSafe(topdown_orientation_, "fallback topdown params orientation");
    }

    const Eigen::Vector3d rpy = rpyFromMatrix(msgQuatToMatrix(topdown_orientation_));
    RCLCPP_INFO(get_logger(),
      "topdown source=%s q=(%.6f, %.6f, %.6f, %.6f) RPY(rad)=(%.4f, %.4f, %.4f)",
      topdown_orientation_source_.c_str(),
      topdown_orientation_.x, topdown_orientation_.y,
      topdown_orientation_.z, topdown_orientation_.w,
      rpy.x(), rpy.y(), rpy.z());
  }

  std::optional<geometry_msgs::msg::Quaternion> fullAnyGraspTcpOrientation(
    const AnyGraspTarget & target)
  {
    const Eigen::Matrix3d r_base_grasp = target.rotation;
    if (!r_base_grasp.allFinite()) {
      RCLCPP_ERROR(get_logger(), "AnyGrasp rotation_matrix rejected: NaN/inf.");
      return std::nullopt;
    }

    // Roll-pitch-yaw correction from AnyGrasp grasp frame to Franka TCP frame.
    // The composed matrix is R_grasp_to_tcp = Rz(yaw) * Ry(pitch) * Rx(roll).
    const Eigen::AngleAxisd rx(grasp_to_tcp_roll_, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd ry(grasp_to_tcp_pitch_, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd rz(grasp_to_tcp_yaw_, Eigen::Vector3d::UnitZ());
    const Eigen::Matrix3d r_grasp_to_tcp = (rz * ry * rx).toRotationMatrix();
    const Eigen::Matrix3d r_base_tcp = r_base_grasp * r_grasp_to_tcp;

    const Eigen::Vector3d grasp_rpy = rpyFromMatrix(r_base_grasp);
    const Eigen::Vector3d tcp_rpy = rpyFromMatrix(r_base_tcp);
    RCLCPP_INFO(get_logger(),
      "AnyGrasp rotation_matrix:\n"
      "  [%.6f %.6f %.6f]\n"
      "  [%.6f %.6f %.6f]\n"
      "  [%.6f %.6f %.6f]",
      r_base_grasp(0, 0), r_base_grasp(0, 1), r_base_grasp(0, 2),
      r_base_grasp(1, 0), r_base_grasp(1, 1), r_base_grasp(1, 2),
      r_base_grasp(2, 0), r_base_grasp(2, 1), r_base_grasp(2, 2));
    if (print_anygrasp_rpy_) {
      RCLCPP_INFO(get_logger(),
        "AnyGrasp RPY(rad)=(%.4f, %.4f, %.4f)",
        grasp_rpy.x(), grasp_rpy.y(), grasp_rpy.z());
    }
    RCLCPP_INFO(get_logger(),
      "grasp_to_tcp RPY(rad)=(%.4f, %.4f, %.4f)",
      grasp_to_tcp_roll_, grasp_to_tcp_pitch_, grasp_to_tcp_yaw_);

    Eigen::Quaterniond q(r_base_tcp);
    auto msg = eigenToMsg(q);
    RCLCPP_INFO(get_logger(),
      "final TCP quaternion xyzw=(%.6f, %.6f, %.6f, %.6f)",
      msg.x, msg.y, msg.z, msg.w);
    RCLCPP_INFO(get_logger(),
      "final TCP RPY(rad)=(%.4f, %.4f, %.4f)",
      tcp_rpy.x(), tcp_rpy.y(), tcp_rpy.z());

    if (!quaternionSafe(msg, "AnyGrasp TCP orientation")) {
      return std::nullopt;
    }
    return msg;
  }

  std::optional<geometry_msgs::msg::Quaternion> planarYawAnyGraspTcpOrientation(
    const AnyGraspTarget & target)
  {
    if (!topdown_orientation_ready_) {
      RCLCPP_ERROR(get_logger(), "planar_yaw rejected: topdown orientation is not ready.");
      return std::nullopt;
    }
    if (planar_yaw_axis_index_ < 0 || planar_yaw_axis_index_ > 2) {
      RCLCPP_ERROR(get_logger(),
        "planar_yaw rejected: planar_yaw_axis_index=%d is outside [0, 2].",
        planar_yaw_axis_index_);
      return std::nullopt;
    }

    const Eigen::Matrix3d r_base_grasp = target.rotation;
    if (!r_base_grasp.allFinite()) {
      RCLCPP_ERROR(get_logger(), "planar_yaw rejected: rotation_matrix has NaN/inf.");
      return std::nullopt;
    }

    const Eigen::Vector3d axis = r_base_grasp.col(planar_yaw_axis_index_);
    const double xy_norm = std::sqrt(axis.x() * axis.x() + axis.y() * axis.y());
    RCLCPP_INFO(get_logger(),
      "planar_yaw selected axis[%d]=(%.6f, %.6f, %.6f), xy_norm=%.6f",
      planar_yaw_axis_index_, axis.x(), axis.y(), axis.z(), xy_norm);
    if (!std::isfinite(xy_norm) || xy_norm < 1e-3) {
      RCLCPP_ERROR(get_logger(),
        "planar_yaw rejected: selected axis XY projection norm %.6f < 1e-3.",
        xy_norm);
      return std::nullopt;
    }

    const double extracted_yaw = std::atan2(axis.y(), axis.x());
    const double final_yaw = extracted_yaw + planar_yaw_offset_;
    const Eigen::Vector3d topdown_rpy = rpyFromMatrix(msgQuatToMatrix(topdown_orientation_));

    const Eigen::Matrix3d r_base_tcp =
      matrixFromRpy(topdown_rpy.x(), topdown_rpy.y(), final_yaw);
    const Eigen::Vector3d final_rpy = rpyFromMatrix(r_base_tcp);
    const auto msg = eigenToMsg(Eigen::Quaterniond(r_base_tcp));

    RCLCPP_INFO(get_logger(), "anygrasp_orientation_mode=planar_yaw");
    RCLCPP_INFO(get_logger(),
      "extracted yaw=%.4f planar_yaw_offset=%.4f final_yaw=%.4f",
      extracted_yaw, planar_yaw_offset_, final_yaw);
    RCLCPP_INFO(get_logger(),
      "topdown source=%s topdown RPY(rad)=(%.4f, %.4f, %.4f)",
      topdown_orientation_source_.c_str(), topdown_rpy.x(), topdown_rpy.y(), topdown_rpy.z());
    RCLCPP_INFO(get_logger(),
      "final TCP RPY(rad)=(%.4f, %.4f, %.4f)",
      final_rpy.x(), final_rpy.y(), final_rpy.z());
    RCLCPP_INFO(get_logger(),
      "final TCP quaternion xyzw=(%.6f, %.6f, %.6f, %.6f)",
      msg.x, msg.y, msg.z, msg.w);

    if (!quaternionSafe(msg, "planar_yaw TCP orientation")) {
      return std::nullopt;
    }
    return msg;
  }

  std::optional<geometry_msgs::msg::Quaternion> selectedTcpOrientation(
    const AnyGraspTarget & target,
    const geometry_msgs::msg::Pose & current_pose)
  {
    RCLCPP_INFO(get_logger(),
      "anygrasp_orientation_mode=%s use_anygrasp_orientation_legacy=%s",
      anygrasp_orientation_mode_.c_str(), tf(use_anygrasp_orientation_));

    if (anygrasp_orientation_mode_ == "full") {
      return fullAnyGraspTcpOrientation(target);
    }
    if (anygrasp_orientation_mode_ == "planar_yaw") {
      return planarYawAnyGraspTcpOrientation(target);
    }
    if (anygrasp_orientation_mode_ == "off") {
      if (use_fixed_orientation_) {
        return fixedOrientation();
      }
      return current_pose.orientation;
    }

    RCLCPP_ERROR(get_logger(),
      "Unsupported anygrasp_orientation_mode='%s'. Use off, full, or planar_yaw.",
      anygrasp_orientation_mode_.c_str());
    return std::nullopt;
  }

  bool applyCameraViewOffset(AnyGraspTarget & target)
  {
    RCLCPP_INFO(get_logger(),
      "raw target base translation=(%.4f, %.4f, %.4f)",
      target.tx, target.ty, target.tz);
    RCLCPP_INFO(get_logger(),
      "raw hover base translation=(%.4f, %.4f, %.4f)",
      target.hx, target.hy, target.hz);

    if (!use_camera_view_offset_) {
      RCLCPP_INFO(get_logger(), "use_camera_view_offset=false: no camera-view XY offset applied.");
      return true;
    }

    RCLCPP_INFO(get_logger(),
      "camera_view_x_offset=%.4f camera_view_y_offset=%.4f source_frame_id=%s target_frame=%s",
      camera_view_x_offset_, camera_view_y_offset_,
      target.source_frame_id.c_str(), target_frame_.c_str());

    geometry_msgs::msg::TransformStamped tf_base_cam;
    try {
      tf_base_cam = tf_buffer_->lookupTransform(
        target_frame_,
        target.source_frame_id,
        tf2::TimePointZero,
        std::chrono::milliseconds(500));
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(),
        "Failed to transform camera-view offset from %s to %s: %s",
        target.source_frame_id.c_str(), target_frame_.c_str(), e.what());
      return false;
    }

    const auto & q_msg = tf_base_cam.transform.rotation;
    Eigen::Quaterniond q_base_cam(q_msg.w, q_msg.x, q_msg.y, q_msg.z);
    if (!std::isfinite(q_base_cam.norm()) || q_base_cam.norm() < 1e-9) {
      RCLCPP_ERROR(get_logger(), "Invalid TF quaternion while converting camera-view offset.");
      return false;
    }
    q_base_cam.normalize();

    const Eigen::Vector3d offset_cam(camera_view_x_offset_, camera_view_y_offset_, 0.0);
    const Eigen::Vector3d offset_base = q_base_cam.toRotationMatrix() * offset_cam;

    RCLCPP_INFO(get_logger(),
      "converted base offset=(%.4f, %.4f, %.4f); z component is logged but not applied",
      offset_base.x(), offset_base.y(), offset_base.z());

    target.tx += offset_base.x();
    target.ty += offset_base.y();
    target.hx += offset_base.x();
    target.hy += offset_base.y();

    RCLCPP_INFO(get_logger(),
      "offset-applied target base translation=(%.4f, %.4f, %.4f)",
      target.tx, target.ty, target.tz);
    RCLCPP_INFO(get_logger(),
      "offset-applied hover base translation=(%.4f, %.4f, %.4f)",
      target.hx, target.hy, target.hz);
    return true;
  }

  geometry_msgs::msg::Pose makePose(
    double x, double y, double z, const geometry_msgs::msg::Quaternion & q) const
  {
    geometry_msgs::msg::Pose p;
    p.position.x = x;
    p.position.y = y;
    p.position.z = z;
    p.orientation = q;
    return p;
  }

  std::map<std::string, double> initialJointTarget() const
  {
    return {
      {"fr3_joint1", initial_j1_},
      {"fr3_joint2", initial_j2_},
      {"fr3_joint3", initial_j3_},
      {"fr3_joint4", initial_j4_},
      {"fr3_joint5", initial_j5_},
      {"fr3_joint6", initial_j6_},
      {"fr3_joint7", initial_j7_}
    };
  }

  bool moveInitialJoint(const std::string & phase)
  {
    RCLCPP_INFO(get_logger(),
      "[%s] initial joint target=[%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
      phase.c_str(), initial_j1_, initial_j2_, initial_j3_, initial_j4_,
      initial_j5_, initial_j6_, initial_j7_);

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "[%s] dry_run=true: no initial joint planning/execution.", phase.c_str());
      return true;
    }

    move_group_->setJointValueTarget(initialJointTarget());
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool plan_ok = static_cast<bool>(move_group_->plan(plan));
    if (!plan_ok) {
      RCLCPP_ERROR(get_logger(), "[%s] initial joint move planning FAILED.", phase.c_str());
      return false;
    }
    RCLCPP_INFO(get_logger(), "[%s] initial joint move planning SUCCESS.", phase.c_str());

    if (plan_only_ || !execute_) {
      RCLCPP_INFO(get_logger(), "[%s] plan_only=true or execute=false: not executing.", phase.c_str());
      return true;
    }

    const auto result = move_group_->execute(plan);
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "[%s] initial joint move execute FAILED.", phase.c_str());
      return false;
    }
    RCLCPP_INFO(get_logger(), "[%s] initial joint move execute SUCCESS.", phase.c_str());
    return true;
  }

  bool cartesianMove(
    const std::vector<geometry_msgs::msg::Pose> & waypoints,
    const std::string & phase)
  {
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double fraction = move_group_->computeCartesianPath(
      waypoints, cartesian_eef_step_, cartesian_jump_threshold_, trajectory);

    RCLCPP_INFO(get_logger(), "[%s] Cartesian fraction=%.4f", phase.c_str(), fraction);
    if (fraction < min_cartesian_fraction_) {
      RCLCPP_ERROR(get_logger(),
        "[%s] fraction %.4f < min_cartesian_fraction %.4f: not executing.",
        phase.c_str(), fraction, min_cartesian_fraction_);
      return false;
    }

    if (dry_run_) {
      RCLCPP_INFO(get_logger(), "[%s] dry_run=true: no execution.", phase.c_str());
      return true;
    }
    if (plan_only_ || !execute_) {
      RCLCPP_INFO(get_logger(), "[%s] plan_only=true or execute=false: not executing.", phase.c_str());
      return true;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = trajectory;
    const auto result = move_group_->execute(plan);
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "[%s] execute FAILED.", phase.c_str());
      return false;
    }
    RCLCPP_INFO(get_logger(), "[%s] execute SUCCESS.", phase.c_str());
    return true;
  }

  bool gripperOpen()
  {
    if (dry_run_ || plan_only_ || !execute_) {
      RCLCPP_INFO(get_logger(), "dry_run/plan_only/execute gate: skipping gripper open.");
      return true;
    }
    if (!move_client_->wait_for_action_server(std::chrono::seconds(15))) {
      RCLCPP_ERROR(
        get_logger(),
        "Gripper move action server unavailable: name=%s type=franka_msgs/action/Move",
        gripper_move_action_.c_str());
      return false;
    }
    FrankaMove::Goal goal;
    goal.width = gripper_open_width_;
    goal.speed = gripper_speed_;
    RCLCPP_INFO(get_logger(),
      "Sending gripper open: action=%s width=%.3f speed=%.3f",
      gripper_move_action_.c_str(), goal.width, goal.speed);
    auto future_handle = move_client_->async_send_goal(goal);
    if (future_handle.wait_for(std::chrono::seconds(5)) != std::future_status::ready ||
      !future_handle.get())
    {
      RCLCPP_ERROR(get_logger(), "Gripper open goal failed.");
      return false;
    }
    auto result_future = move_client_->async_get_result(future_handle.get());
    if (result_future.wait_for(std::chrono::seconds(15)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Gripper open result timeout: action=%s", gripper_move_action_.c_str());
      return false;
    }
    const bool success = result_future.get().result->success;
    if (!success) {
      RCLCPP_ERROR(get_logger(), "Gripper open result failed: action=%s", gripper_move_action_.c_str());
      return false;
    }
    RCLCPP_INFO(get_logger(), "Gripper open SUCCESS.");
    return true;
  }

  bool gripperClose()
  {
    if (!do_gripper_) {
      RCLCPP_INFO(get_logger(), "do_gripper=false: skipping gripper close.");
      return true;
    }
    if (dry_run_ || plan_only_ || !execute_) {
      RCLCPP_INFO(get_logger(), "dry_run/plan_only/execute gate: skipping gripper close.");
      return true;
    }
    if (!grasp_client_->wait_for_action_server(std::chrono::seconds(15))) {
      RCLCPP_ERROR(
        get_logger(),
        "Gripper grasp action server unavailable: name=%s type=franka_msgs/action/Grasp",
        gripper_grasp_action_.c_str());
      return false;
    }
    FrankaGrasp::Goal goal;
    goal.width = gripper_close_width_;
    goal.speed = gripper_speed_;
    goal.force = gripper_force_;
    goal.epsilon.inner = gripper_epsilon_inner_;
    goal.epsilon.outer = gripper_epsilon_outer_;
    RCLCPP_INFO(get_logger(),
      "Sending gripper close: action=%s width=%.3f speed=%.3f force=%.3f epsilon=(%.4f, %.4f)",
      gripper_grasp_action_.c_str(), goal.width, goal.speed, goal.force,
      goal.epsilon.inner, goal.epsilon.outer);
    auto future_handle = grasp_client_->async_send_goal(goal);
    if (future_handle.wait_for(std::chrono::seconds(5)) != std::future_status::ready ||
      !future_handle.get())
    {
      RCLCPP_ERROR(get_logger(), "Gripper close goal failed.");
      return false;
    }
    auto result_future = grasp_client_->async_get_result(future_handle.get());
    if (result_future.wait_for(std::chrono::seconds(15)) != std::future_status::ready) {
      RCLCPP_ERROR(get_logger(), "Gripper close result timeout: action=%s", gripper_grasp_action_.c_str());
      return false;
    }
    const bool success = result_future.get().result->success;
    if (!success) {
      RCLCPP_ERROR(get_logger(), "Gripper close result failed: action=%s", gripper_grasp_action_.c_str());
      return false;
    }
    RCLCPP_INFO(get_logger(), "Gripper close SUCCESS.");
    return true;
  }

  bool executeSequence(const AnyGraspTarget & raw_target)
  {
    AnyGraspTarget target = raw_target;
    RCLCPP_INFO(get_logger(),
      "anygrasp_orientation_mode=%s orientation_test_mode=%s",
      anygrasp_orientation_mode_.c_str(), tf(orientation_test_mode_));

    if (!applyCameraViewOffset(target)) {
      return false;
    }

    if (target.score < min_score_) {
      RCLCPP_WARN(get_logger(), "Rejected: score %.4f < min_score %.4f", target.score, min_score_);
      return false;
    }
    if (!inWorkspace(target.tx, target.ty, target.tz, "grasp translation")) {
      return false;
    }
    if (!inWorkspace(target.hx, target.hy, target.hz, "hover translation")) {
      return false;
    }

    geometry_msgs::msg::Pose current_pose;
    try {
      current_pose = move_group_->getCurrentPose(eef_link_).pose;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Failed to read current pose: %s", e.what());
      return false;
    }

    const auto maybe_grasp_q = selectedTcpOrientation(target, current_pose);
    if (!maybe_grasp_q) {
      return false;
    }
    geometry_msgs::msg::Quaternion grasp_q = *maybe_grasp_q;

    if (!quaternionSafe(grasp_q, "selected TCP orientation")) {
      return false;
    }

    const double grasp_z = grasp_z_;
    geometry_msgs::msg::Pose safe_pose = makePose(target.hx, target.hy, safe_z_, grasp_q);
    geometry_msgs::msg::Pose hover_pose = makePose(target.hx, target.hy, target.hz, grasp_q);
    geometry_msgs::msg::Pose grasp_pose = makePose(target.tx, target.ty, grasp_z, grasp_q);
    geometry_msgs::msg::Pose lift_pose = makePose(target.tx, target.ty, lift_z_, grasp_q);

    if (!inWorkspace(safe_pose.position.x, safe_pose.position.y, safe_pose.position.z, "safe hover")) {
      return false;
    }
    if (!inWorkspace(grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z, "grasp pose")) {
      return false;
    }
    if (!inWorkspace(lift_pose.position.x, lift_pose.position.y, lift_pose.position.z, "lift pose")) {
      return false;
    }

    RCLCPP_INFO(get_logger(),
      "target poses: safe=(%.3f, %.3f, %.3f) hover=(%.3f, %.3f, %.3f) grasp=(%.3f, %.3f, %.3f) lift=(%.3f, %.3f, %.3f)",
      safe_pose.position.x, safe_pose.position.y, safe_pose.position.z,
      hover_pose.position.x, hover_pose.position.y, hover_pose.position.z,
      grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z,
      lift_pose.position.x, lift_pose.position.y, lift_pose.position.z);

    if (!moveInitialJoint("initial_before_pick")) {
      return false;
    }

    if (!cartesianMove({safe_pose, hover_pose}, "hover_with_selected_orientation")) {
      return false;
    }

    if (orientation_test_mode_) {
      RCLCPP_WARN(get_logger(),
        "orientation_test_mode=true: skipping descent, gripper, and place; lifting and returning initial.");
      (void)cartesianMove({lift_pose}, "orientation_test_lift");
      (void)moveInitialJoint("orientation_test_return_initial");
      return true;
    }

    if (do_preopen_gripper_) {
      RCLCPP_INFO(get_logger(), "do_preopen_gripper=true: opening gripper before descent.");
      if (!gripperOpen()) {
        RCLCPP_ERROR(get_logger(), "Preopen gripper failed; returning to initial joint pose.");
        (void)moveInitialJoint("return_initial_after_preopen_failure");
        return false;
      }
    } else {
      RCLCPP_INFO(get_logger(),
        "do_preopen_gripper=false: skipping gripper open and continuing to descent if enabled.");
    }

    if (do_descent_) {
      if (!cartesianMove({grasp_pose}, "descent_to_grasp")) {
        (void)moveInitialJoint("return_initial_after_descent_failure");
        return false;
      }
    } else {
      RCLCPP_INFO(get_logger(), "do_descent=false: skipping descent to grasp pose.");
    }

    if (do_gripper_) {
      if (!gripperClose()) {
        RCLCPP_ERROR(get_logger(), "Gripper close failed; returning to initial joint pose.");
        (void)moveInitialJoint("return_initial_after_gripper_failure");
        return false;
      }
    } else {
      RCLCPP_INFO(get_logger(), "do_gripper=false: skipping gripper close.");
    }

    if (!cartesianMove({lift_pose}, "lift_after_grasp_or_hover")) {
      (void)moveInitialJoint("return_initial_after_lift_failure");
      return false;
    }

    if (do_place_) {
      geometry_msgs::msg::Quaternion place_q = grasp_q;
      if (!keep_grasp_orientation_during_place_) {
        place_q = use_fixed_orientation_ ? fixedOrientation() : current_pose.orientation;
      }
      geometry_msgs::msg::Pose place_safe = makePose(place_x_, place_y_, place_safe_z_, place_q);
      geometry_msgs::msg::Pose place_pose = makePose(place_x_, place_y_, place_z_, place_q);
      if (!inWorkspace(place_safe.position.x, place_safe.position.y, place_safe.position.z, "place safe")) {
        return false;
      }
      if (!inWorkspace(place_pose.position.x, place_pose.position.y, place_pose.position.z, "place pose")) {
        return false;
      }
      if (!cartesianMove({place_safe, place_pose}, "place_move")) {
        (void)moveInitialJoint("return_initial_after_place_failure");
        return false;
      }
      if (do_gripper_) {
        if (!gripperOpen()) {
          RCLCPP_ERROR(get_logger(), "Place gripper open failed; returning to initial joint pose.");
          (void)moveInitialJoint("return_initial_after_place_open_failure");
          return false;
        }
      }
    } else {
      RCLCPP_INFO(get_logger(), "do_place=false: skipping place.");
    }

    (void)moveInitialJoint("return_initial_after_sequence");
    RCLCPP_INFO(get_logger(), "AnyGrasp pick/place sequence complete.");
    return true;
  }

  bool fetchRobotDescription()
  {
    RCLCPP_INFO(get_logger(),
      "Fetching robot_description from /move_group (timeout=%d s)...",
      robot_desc_timeout_sec_);

    auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(
      shared_from_this(), "/move_group");

    bool ready = false;
    for (int i = 0; i < robot_desc_timeout_sec_ * 2 && rclcpp::ok(); ++i) {
      if (param_client->service_is_ready()) {
        ready = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (!ready) {
      RCLCPP_ERROR(get_logger(), "/move_group parameter service is not ready.");
      return false;
    }

    auto future = param_client->get_parameters(
      {"robot_description", "robot_description_semantic"});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
      if (!rclcpp::ok() || std::chrono::steady_clock::now() > deadline) {
        RCLCPP_ERROR(get_logger(), "Timed out waiting for robot description.");
        return false;
      }
    }

    bool ok = true;
    for (auto & p : future.get()) {
      if (p.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
        RCLCPP_ERROR(get_logger(), "'%s' not set in /move_group.", p.get_name().c_str());
        ok = false;
        continue;
      }
      try {
        declare_parameter(p.get_name(), p.get_parameter_value());
      } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
        set_parameter(p);
      }
    }
    return ok;
  }

  std::string input_topic_;
  std::string target_frame_;
  std::string planning_group_;
  std::string eef_link_;
  std::string gripper_move_action_;
  std::string gripper_grasp_action_;

  bool dry_run_{true};
  bool plan_only_{true};
  bool execute_{false};
  bool once_{true};
  bool do_descent_{false};
  bool do_preopen_gripper_{true};
  bool do_gripper_{false};
  bool do_place_{false};
  bool use_camera_view_offset_{false};
  bool use_target_smoothing_{false};
  bool require_stable_target_{true};
  bool use_fixed_orientation_{true};
  bool use_anygrasp_orientation_{false};
  bool orientation_test_mode_{false};
  bool normalize_quaternion_{true};
  bool print_anygrasp_rpy_{true};
  bool keep_grasp_orientation_during_place_{true};
  bool topdown_orientation_ready_{false};

  double fixed_qx_{0.0}, fixed_qy_{0.0}, fixed_qz_{0.0}, fixed_qw_{1.0};
  double camera_view_x_offset_{0.0}, camera_view_y_offset_{0.0};
  double max_target_std_xy_{0.010}, max_target_std_z_{0.015};
  double stable_target_timeout_sec_{5.0};
  double topdown_qx_{0.0}, topdown_qy_{0.0}, topdown_qz_{0.0}, topdown_qw_{1.0};
  double grasp_to_tcp_roll_{0.0}, grasp_to_tcp_pitch_{0.0}, grasp_to_tcp_yaw_{0.0};
  double planar_yaw_offset_{0.0};
  double safe_z_{0.55}, grasp_z_offset_{0.0}, grasp_z_{0.12}, lift_z_{0.55};
  double place_x_{0.45}, place_y_{-0.30}, place_z_{0.35}, place_safe_z_{0.55};
  double min_score_{0.2};
  double x_min_{0.20}, x_max_{0.70}, y_min_{-0.50}, y_max_{0.50}, z_min_{0.05}, z_max_{0.80};
  double cartesian_eef_step_{0.01}, cartesian_jump_threshold_{0.0}, min_cartesian_fraction_{0.95};
  double velocity_scaling_{0.05}, acceleration_scaling_{0.05};
  double initial_j1_{0.0}, initial_j2_{-0.785}, initial_j3_{0.0}, initial_j4_{-2.356};
  double initial_j5_{0.0}, initial_j6_{1.571}, initial_j7_{0.785};
  double gripper_open_width_{0.08}, gripper_close_width_{0.025};
  double gripper_speed_{0.03}, gripper_force_{20.0};
  double gripper_epsilon_inner_{0.005}, gripper_epsilon_outer_{0.005};
  int robot_desc_timeout_sec_{30};
  int planar_yaw_axis_index_{0};
  int smoothing_window_{5};
  int min_smoothing_samples_{3};
  std::string anygrasp_orientation_mode_{"off"};
  std::string topdown_orientation_source_{"current_at_start"};
  std::string smoothing_method_{"median"};
  geometry_msgs::msg::Quaternion topdown_orientation_;

  std::mutex target_mutex_;
  AnyGraspTarget latest_target_;
  std::deque<AnyGraspTarget> target_buffer_;
  std::chrono::steady_clock::time_point smoothing_started_{std::chrono::steady_clock::now()};
  bool has_target_{false};
  std::atomic_bool processing_{false};
  std::atomic_bool done_{false};
  std::atomic_bool stop_worker_{false};
  std::thread worker_thread_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp_action::Client<FrankaMove>::SharedPtr move_client_;
  rclcpp_action::Client<FrankaGrasp>::SharedPtr grasp_client_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AnyGraspPickPlaceNode>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spin_thread([&executor]() { executor.spin(); });

  node->initMoveGroup();
  node->startWorker();

  spin_thread.join();
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
