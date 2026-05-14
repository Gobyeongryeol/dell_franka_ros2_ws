#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/string.hpp>
#include <gpd_ros2_msgs/srv/detect_constrained_grasps.hpp>
#include <gpd_ros2_msgs/msg/grasp_config.hpp>
#include <gpd_ros2_msgs/msg/grasp_config_list.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <regex>
#include <string>
#include <vector>

using DetectGrasps = gpd_ros2_msgs::srv::DetectConstrainedGrasps;
using GraspConfig  = gpd_ros2_msgs::msg::GraspConfig;
using Clock        = std::chrono::steady_clock;

// Extract first 3 floats from string regardless of format (same as yolo_hover_linear_node).
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

static bool extractJsonNumberByKey(
  const std::string & s, const std::string & key, double & value)
{
  const std::string num_pat =
    R"([-+]?(?:[0-9]+\.?[0-9]*|[0-9]*\.[0-9]+)(?:[eE][-+]?[0-9]+)?)";
  std::regex re("\"" + key + "\"\\s*:\\s*(" + num_pat + ")");
  std::smatch match;
  if (!std::regex_search(s, match, re) || match.size() < 2) {
    return false;
  }
  value = std::stod(match[1].str());
  return true;
}

static bool parseTargetBaseJson(
  const std::string & s, double & x, double & y, double & z)
{
  return extractJsonNumberByKey(s, "x_base", x) &&
         extractJsonNumberByKey(s, "y_base", y) &&
         extractJsonNumberByKey(s, "z_base", z);
}

static bool extractJsonStringByKey(
  const std::string & s, const std::string & key, std::string & value)
{
  std::regex re("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"");
  std::smatch match;
  if (!std::regex_search(s, match, re) || match.size() < 2) {
    return false;
  }
  value = match[1].str();
  return true;
}

static bool extractFirstAvailableNumber(
  const std::string & s, const std::vector<std::string> & keys, double & value)
{
  for (const auto & key : keys) {
    if (extractJsonNumberByKey(s, key, value)) {
      return true;
    }
  }
  return false;
}

static bool parseTarget3dJson(
  const std::string & s,
  double & x,
  double & y,
  double & z,
  std::string & frame_id)
{
  bool ok_x = extractFirstAvailableNumber(
    s, {"x_m", "camera_x", "x_camera", "x"}, x);
  bool ok_y = extractFirstAvailableNumber(
    s, {"y_m", "camera_y", "y_camera", "y"}, y);
  bool ok_z = extractFirstAvailableNumber(
    s, {"z_m", "camera_z", "z_camera", "z"}, z);
  if (!(ok_x && ok_y && ok_z)) {
    std::string camera_xyz;
    if (extractJsonStringByKey(s, "camera_xyz", camera_xyz) &&
        parseXYZ(camera_xyz, x, y, z))
    {
      ok_x = ok_y = ok_z = true;
    }
  }
  extractJsonStringByKey(s, "frame_id", frame_id);
  return ok_x && ok_y && ok_z;
}

class GpdClientNode : public rclcpp::Node
{
public:
  GpdClientNode()
  : Node("gpd_client_node"),
    service_in_progress_(false),
    done_(false),
    last_request_time_(Clock::time_point::min()),
    has_yolo_target_base_(false),
    has_yolo_target_camera_(false),
    target_base_x_(0.0), target_base_y_(0.0), target_base_z_(0.0),
    target_camera_x_(0.0), target_camera_y_(0.0), target_camera_z_(0.0)
  {
    // ── existing params ────────────────────────────────────────────────────
    declare_parameter<std::string>("cloud_topic",          "/camera/camera/depth/color/points");
    declare_parameter<std::string>("service_name",         "/detect_constrained_grasps");
    declare_parameter<bool>("dry_run",                     true);
    declare_parameter<bool>("once",                        true);
    declare_parameter<double>("min_request_interval_sec",  5.0);

    cloud_topic_  = get_parameter("cloud_topic").as_string();
    service_name_ = get_parameter("service_name").as_string();
    dry_run_      = get_parameter("dry_run").as_bool();
    once_         = get_parameter("once").as_bool();
    min_interval_ = std::chrono::duration<double>(
                      get_parameter("min_request_interval_sec").as_double());

    // ── YOLO crop params ───────────────────────────────────────────────────
    declare_parameter<bool>("use_yolo_target_crop",         false);
    declare_parameter<std::string>("crop_target_source",     "camera_3d");
    declare_parameter<std::string>("yolo_target_3d_topic",   "/yolo/target_3d");
    declare_parameter<std::string>("yolo_target_base_topic", "/yolo/target_base");
    declare_parameter<std::string>("target_camera_frame",    "camera_color_optical_frame");
    declare_parameter<std::string>("target_frame",           "base");
    declare_parameter<std::string>("cloud_frame",            "camera_depth_optical_frame");
    declare_parameter<double>("crop_radius_x",               0.12);
    declare_parameter<double>("crop_radius_y",               0.12);
    declare_parameter<double>("crop_radius_z",               0.15);
    declare_parameter<int>("crop_min_points",                100);
    declare_parameter<bool>("publish_cropped_cloud",         true);
    declare_parameter<std::string>("cropped_cloud_topic",    "/gpd/cropped_cloud");
    declare_parameter<bool>("auto_expand_crop_if_empty",     true);
    declare_parameter<double>("expanded_crop_radius_x",      0.30);
    declare_parameter<double>("expanded_crop_radius_y",      0.30);
    declare_parameter<double>("expanded_crop_radius_z",      0.35);

    use_yolo_target_crop_   = get_parameter("use_yolo_target_crop").as_bool();
    crop_target_source_     = get_parameter("crop_target_source").as_string();
    yolo_target_3d_topic_   = get_parameter("yolo_target_3d_topic").as_string();
    yolo_target_base_topic_ = get_parameter("yolo_target_base_topic").as_string();
    target_camera_frame_    = get_parameter("target_camera_frame").as_string();
    target_frame_           = get_parameter("target_frame").as_string();
    cloud_frame_            = get_parameter("cloud_frame").as_string();
    crop_radius_x_          = get_parameter("crop_radius_x").as_double();
    crop_radius_y_          = get_parameter("crop_radius_y").as_double();
    crop_radius_z_          = get_parameter("crop_radius_z").as_double();
    crop_min_points_        = get_parameter("crop_min_points").as_int();
    publish_cropped_cloud_  = get_parameter("publish_cropped_cloud").as_bool();
    cropped_cloud_topic_    = get_parameter("cropped_cloud_topic").as_string();
    auto_expand_crop_if_empty_ = get_parameter("auto_expand_crop_if_empty").as_bool();
    expanded_crop_radius_x_ = get_parameter("expanded_crop_radius_x").as_double();
    expanded_crop_radius_y_ = get_parameter("expanded_crop_radius_y").as_double();
    expanded_crop_radius_z_ = get_parameter("expanded_crop_radius_z").as_double();

    if (crop_target_source_ != "camera_3d" && crop_target_source_ != "base") {
      RCLCPP_WARN(
        get_logger(),
        "[GPD YOLO crop] Unsupported crop_target_source='%s'. Falling back to camera_3d.",
        crop_target_source_.c_str());
      crop_target_source_ = "camera_3d";
    }

    // ── ROS interfaces ─────────────────────────────────────────────────────
    client_ = create_client<DetectGrasps>(service_name_);

    rclcpp::QoS best_qos(1);
    best_qos.transient_local();
    best_qos.reliable();
    best_grasp_pub_ = create_publisher<GraspConfig>("/best_gpd_grasp", best_qos);

    if (use_yolo_target_crop_) {
      tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
      tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

      yolo_target_3d_sub_ = create_subscription<std_msgs::msg::String>(
        yolo_target_3d_topic_, 10,
        std::bind(&GpdClientNode::yoloTarget3dCallback, this, std::placeholders::_1));
      yolo_target_base_sub_ = create_subscription<std_msgs::msg::String>(
        yolo_target_base_topic_, 10,
        std::bind(&GpdClientNode::yoloTargetBaseCallback, this, std::placeholders::_1));

      if (publish_cropped_cloud_) {
        cropped_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          cropped_cloud_topic_, rclcpp::SensorDataQoS());
      }

      RCLCPP_INFO(get_logger(),
        "[GPD YOLO crop] enabled  crop_target_source=%s  target_3d_topic=%s  target_base_topic=%s",
        crop_target_source_.c_str(), yolo_target_3d_topic_.c_str(), yolo_target_base_topic_.c_str());
      RCLCPP_INFO(get_logger(),
        "[GPD YOLO crop] target_camera_frame=%s  target_frame=%s  cloud_frame=%s",
        target_camera_frame_.c_str(), target_frame_.c_str(), cloud_frame_.c_str());
      RCLCPP_INFO(get_logger(),
        "[GPD YOLO crop] crop_radius=(%.3f, %.3f, %.3f)  min_points=%d",
        crop_radius_x_, crop_radius_y_, crop_radius_z_, crop_min_points_);
      RCLCPP_INFO(get_logger(),
        "[GPD YOLO crop] auto_expand_crop_if_empty=%s  expanded_radius=(%.3f, %.3f, %.3f)",
        auto_expand_crop_if_empty_ ? "true" : "false",
        expanded_crop_radius_x_, expanded_crop_radius_y_, expanded_crop_radius_z_);
    }

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, 1,
      std::bind(&GpdClientNode::cloudCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
      "GPD client ready. cloud_topic=%s  service=%s  dry_run=%s  once=%s  interval=%.1fs",
      cloud_topic_.c_str(), service_name_.c_str(),
      dry_run_ ? "true" : "false", once_ ? "true" : "false",
      min_interval_.count());
    RCLCPP_INFO(get_logger(), "Waiting for pointcloud...");
  }

private:
  // ── YOLO target subscribers ──────────────────────────────────────────────
  void yoloTargetBaseCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    double tx = 0.0, ty = 0.0, tz = 0.0;
    if (!parseTargetBaseJson(msg->data, tx, ty, tz)) {
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] Failed to parse x_base/y_base/z_base from JSON: '%s'",
        msg->data.c_str());
      if (!parseXYZ(msg->data, tx, ty, tz)) {
        RCLCPP_WARN(get_logger(),
          "[GPD YOLO crop] Fallback parseXYZ also failed. Ignoring target.");
        return;
      }
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] Falling back to first three numeric values: "
        "(%.3f, %.3f, %.3f)", tx, ty, tz);
    }
    std::lock_guard<std::mutex> lock(target_mutex_);
    target_base_x_ = tx;
    target_base_y_ = ty;
    target_base_z_ = tz;
    has_yolo_target_base_ = true;
  }

  void yoloTarget3dCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    double tx = 0.0, ty = 0.0, tz = 0.0;
    std::string frame_id;
    if (!parseTarget3dJson(msg->data, tx, ty, tz, frame_id)) {
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] Failed to parse camera-frame 3D target from JSON: '%s'",
        msg->data.c_str());
      return;
    }
    if (frame_id.empty()) {
      frame_id = target_camera_frame_;
    }
    std::lock_guard<std::mutex> lock(target_mutex_);
    target_camera_x_ = tx;
    target_camera_y_ = ty;
    target_camera_z_ = tz;
    target_camera_frame_from_msg_ = frame_id;
    has_yolo_target_camera_ = true;
  }

  // ── Crop pointcloud around YOLO target ──────────────────────────────────
  // Returns true on success and sets cropped_out. The crop center is transformed
  // from target_frame to the cloud's own frame so filtering stays in camera coords.
  bool cropCloud(const sensor_msgs::msg::PointCloud2 & in,
                 sensor_msgs::msg::PointCloud2 & out)
  {
    out = sensor_msgs::msg::PointCloud2();
    const std::string msg_cloud_frame = in.header.frame_id;
    std::string actual_cloud_frame = msg_cloud_frame.empty() ? cloud_frame_ : msg_cloud_frame;

    RCLCPP_INFO(get_logger(),
      "[GPD YOLO crop] crop_target_source=%s", crop_target_source_.c_str());
    RCLCPP_INFO(get_logger(),
      "[GPD YOLO crop] cloud msg frame=%s", msg_cloud_frame.c_str());
    RCLCPP_INFO(get_logger(),
      "[GPD YOLO crop] param cloud_frame=%s", cloud_frame_.c_str());
    RCLCPP_INFO(get_logger(),
      "[GPD YOLO crop] using cloud_frame=%s", actual_cloud_frame.c_str());

    geometry_msgs::msg::PointStamped target_input;
    geometry_msgs::msg::PointStamped target_in_cloud;
    if (in.header.stamp.sec != 0 || in.header.stamp.nanosec != 0) {
      target_input.header.stamp = in.header.stamp;
    } else {
      target_input.header.stamp = now();
    }

    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      if (crop_target_source_ == "camera_3d") {
        if (!has_yolo_target_camera_) {
          RCLCPP_WARN(get_logger(),
            "[GPD YOLO crop] No /yolo/target_3d received yet, skipping GPD call.");
          return false;
        }
        const std::string input_frame = (
          target_camera_frame_from_msg_.empty() ? target_camera_frame_ : target_camera_frame_from_msg_
        );
        target_input.header.frame_id = input_frame;
        target_input.point.x = target_camera_x_;
        target_input.point.y = target_camera_y_;
        target_input.point.z = target_camera_z_;
        RCLCPP_INFO(get_logger(),
          "[GPD YOLO crop] target camera frame=%s", input_frame.c_str());
        RCLCPP_INFO(get_logger(),
          "[GPD YOLO crop] target in camera=(%.3f, %.3f, %.3f)",
          target_camera_x_, target_camera_y_, target_camera_z_);
      } else {
        if (!has_yolo_target_base_) {
          RCLCPP_WARN(get_logger(),
            "[GPD YOLO crop] No /yolo/target_base received yet, skipping GPD call.");
          return false;
        }
        target_input.header.frame_id = target_frame_;
        target_input.point.x = target_base_x_;
        target_input.point.y = target_base_y_;
        target_input.point.z = target_base_z_;
        RCLCPP_INFO(get_logger(),
          "[GPD YOLO crop] target in base=(%.3f, %.3f, %.3f)",
          target_base_x_, target_base_y_, target_base_z_);
      }
    }

    try {
      if (actual_cloud_frame == target_input.header.frame_id) {
        target_in_cloud = target_input;
      } else {
        target_in_cloud = tf_buffer_->transform(
          target_input, actual_cloud_frame, tf2::durationFromSec(0.2));
      }
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] TF failed (%s -> %s): %s",
        target_input.header.frame_id.c_str(), actual_cloud_frame.c_str(), ex.what());
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] TF failed, skipping crop/GPD request");
      return false;
    }

    const double cx = target_in_cloud.point.x;
    const double cy = target_in_cloud.point.y;
    const double cz = target_in_cloud.point.z;
    RCLCPP_INFO(get_logger(),
      "[GPD YOLO crop] target in cloud=(%.3f, %.3f, %.3f)", cx, cy, cz);
    RCLCPP_INFO(get_logger(),
      "[GPD YOLO crop] crop radius=(%.3f, %.3f, %.3f)",
      crop_radius_x_, crop_radius_y_, crop_radius_z_);

    // Find x, y, z field byte offsets
    uint32_t x_off = 0, y_off = 0, z_off = 0;
    bool found_x = false, found_y = false, found_z = false;
    uint8_t x_type = 0, y_type = 0, z_type = 0;
    for (const auto & f : in.fields) {
      if (f.name == "x") { x_off = f.offset; x_type = f.datatype; found_x = true; }
      if (f.name == "y") { y_off = f.offset; y_type = f.datatype; found_y = true; }
      if (f.name == "z") { z_off = f.offset; z_type = f.datatype; found_z = true; }
    }
    if (!found_x || !found_y || !found_z) {
      RCLCPP_ERROR(get_logger(), "[GPD YOLO crop] PointCloud2 missing x/y/z fields.");
      return false;
    }
    if (
      x_type != sensor_msgs::msg::PointField::FLOAT32 ||
      y_type != sensor_msgs::msg::PointField::FLOAT32 ||
      z_type != sensor_msgs::msg::PointField::FLOAT32)
    {
      RCLCPP_ERROR(get_logger(),
        "[GPD YOLO crop] PointCloud2 x/y/z fields are not FLOAT32. Skipping crop.");
      return false;
    }

    const uint32_t ps = in.point_step;
    const uint32_t n_total = in.width * in.height;

    RCLCPP_INFO(get_logger(),
      "[GPD YOLO crop] input cloud points=%u", n_total);

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    uint32_t valid_points = 0;

    for (uint32_t row = 0; row < in.height; ++row) {
      for (uint32_t col = 0; col < in.width; ++col) {
        size_t offset = static_cast<size_t>(row) * in.row_step
                      + static_cast<size_t>(col) * ps;
        float px, py, pz;
        std::memcpy(&px, in.data.data() + offset + x_off, sizeof(float));
        std::memcpy(&py, in.data.data() + offset + y_off, sizeof(float));
        std::memcpy(&pz, in.data.data() + offset + z_off, sizeof(float));

        if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) continue;
        ++valid_points;
        min_x = std::min(min_x, static_cast<double>(px));
        min_y = std::min(min_y, static_cast<double>(py));
        min_z = std::min(min_z, static_cast<double>(pz));
        max_x = std::max(max_x, static_cast<double>(px));
        max_y = std::max(max_y, static_cast<double>(py));
        max_z = std::max(max_z, static_cast<double>(pz));
      }
    }

    if (valid_points > 0) {
      RCLCPP_INFO(get_logger(),
        "[GPD YOLO crop] cloud range x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f]",
        min_x, max_x, min_y, max_y, min_z, max_z);
    } else {
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] cloud has no finite xyz points after filtering NaN/inf");
    }

    if (
      valid_points > 0 &&
      (cx < min_x || cx > max_x || cy < min_y || cy > max_y || cz < min_z || cz > max_z)
    ) {
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] target is outside cloud range. Check target source/frame.");
    }

    auto cropWithRadius =
      [&](double rx, double ry, double rz, std::vector<uint8_t> & buf_out) -> uint32_t
    {
      buf_out.clear();
      buf_out.reserve(static_cast<size_t>(n_total) * ps);
      for (uint32_t row = 0; row < in.height; ++row) {
        for (uint32_t col = 0; col < in.width; ++col) {
          const size_t offset = static_cast<size_t>(row) * in.row_step +
                                static_cast<size_t>(col) * ps;
          float px, py, pz;
          std::memcpy(&px, in.data.data() + offset + x_off, sizeof(float));
          std::memcpy(&py, in.data.data() + offset + y_off, sizeof(float));
          std::memcpy(&pz, in.data.data() + offset + z_off, sizeof(float));
          if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) {
            continue;
          }
          if (std::abs(px - cx) < rx &&
              std::abs(py - cy) < ry &&
              std::abs(pz - cz) < rz)
          {
            buf_out.insert(
              buf_out.end(),
              in.data.data() + offset,
              in.data.data() + offset + ps);
          }
        }
      }
      return static_cast<uint32_t>(buf_out.size() / ps);
    };

    double used_rx = crop_radius_x_;
    double used_ry = crop_radius_y_;
    double used_rz = crop_radius_z_;
    std::vector<uint8_t> buf;
    uint32_t n_cropped = cropWithRadius(used_rx, used_ry, used_rz, buf);
    RCLCPP_INFO(get_logger(),
      "[GPD YOLO crop] cropped cloud points=%u", n_cropped);

    if (n_cropped == 0 && auto_expand_crop_if_empty_) {
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] crop=0, retrying with expanded radius=(%.3f, %.3f, %.3f)",
        expanded_crop_radius_x_, expanded_crop_radius_y_, expanded_crop_radius_z_);
      used_rx = expanded_crop_radius_x_;
      used_ry = expanded_crop_radius_y_;
      used_rz = expanded_crop_radius_z_;
      n_cropped = cropWithRadius(used_rx, used_ry, used_rz, buf);
      RCLCPP_INFO(get_logger(),
        "[GPD YOLO crop] expanded cropped cloud points=%u", n_cropped);
    }

    out.header = in.header;
    out.header.frame_id = actual_cloud_frame;
    out.header.stamp = in.header.stamp;
    out.height = 1;
    out.width = n_cropped;
    out.fields = in.fields;
    out.is_bigendian = in.is_bigendian;
    out.point_step = ps;
    out.row_step = n_cropped * ps;
    out.is_dense = false;
    out.data = std::move(buf);

    if (n_cropped == 0) {
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] DEBUG crop=0:\n"
        "  target_cloud=(%.3f, %.3f, %.3f)\n"
        "  crop box x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f]\n"
        "  cloud range x=[%.3f, %.3f] y=[%.3f, %.3f] z=[%.3f, %.3f]",
        cx, cy, cz,
        cx - used_rx, cx + used_rx,
        cy - used_ry, cy + used_ry,
        cz - used_rz, cz + used_rz,
        min_x, max_x, min_y, max_y, min_z, max_z);
    }

    if (static_cast<int>(n_cropped) < crop_min_points_) {
      RCLCPP_WARN(get_logger(),
        "[GPD YOLO crop] Only %u points after crop (min=%d). Skipping GPD call.",
        n_cropped, crop_min_points_);
      return false;
    }

    return true;
  }

  // ── Pointcloud callback ──────────────────────────────────────────────────
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (once_ && done_) return;

    if (service_in_progress_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "Service already in progress, skipping new cloud...");
      return;
    }

    const auto now = Clock::now();
    if (last_request_time_ != Clock::time_point::min() &&
        (now - last_request_time_) < min_interval_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
        "Waiting for interval (%.1fs), skipping cloud...", min_interval_.count());
      return;
    }

    if (!client_->wait_for_service(std::chrono::seconds(0))) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
        "Service %s not available yet, skipping...", service_name_.c_str());
      return;
    }

    // ── decide which cloud to send ────────────────────────────────────────
    sensor_msgs::msg::PointCloud2 cloud_to_send;
    bool use_cropped = false;

    if (use_yolo_target_crop_) {
      sensor_msgs::msg::PointCloud2 cropped;
      const bool ready_for_gpd = cropCloud(*msg, cropped);
      if (publish_cropped_cloud_ && cropped_cloud_pub_ && !cropped.header.frame_id.empty()) {
        cropped_cloud_pub_->publish(cropped);
      }
      if (!ready_for_gpd) {
        return;  // warning already printed inside cropCloud
      }
      cloud_to_send = std::move(cropped);
      use_cropped   = true;
    } else {
      cloud_to_send = *msg;
    }

    service_in_progress_ = true;
    last_request_time_   = now;
    if (once_) done_ = true;

    if (use_cropped) {
      RCLCPP_INFO(get_logger(),
        "[GPD YOLO crop] calling GPD with cropped cloud (%u pts)",
        cloud_to_send.width);
    } else {
      RCLCPP_INFO(get_logger(),
        "Received pointcloud (%u pts). Calling GPD service...",
        msg->width * msg->height);
    }

    auto req = std::make_shared<DetectGrasps::Request>();
    req->cloud_indexed.cloud_sources.cloud = cloud_to_send;

    std_msgs::msg::Int64 cam_idx;
    cam_idx.data = 0;
    req->cloud_indexed.cloud_sources.camera_source.push_back(cam_idx);

    geometry_msgs::msg::Point vp;
    vp.x = 0.0; vp.y = 0.0; vp.z = 0.0;
    req->cloud_indexed.cloud_sources.view_points.push_back(vp);

    req->params_policy = DetectGrasps::Request::USE_CFG_FILE;

    client_->async_send_request(req,
      [this](rclcpp::Client<DetectGrasps>::SharedFuture f) {
        handleResponse(f);
      });
  }

  // ── GPD response handler ─────────────────────────────────────────────────
  void handleResponse(rclcpp::Client<DetectGrasps>::SharedFuture future)
  {
    service_in_progress_ = false;

    auto res = future.get();
    const auto & grasps = res->grasp_configs.grasps;

    if (use_yolo_target_crop_) {
      RCLCPP_INFO(get_logger(),
        "[GPD YOLO crop] GPD returned %zu grasp(s)", grasps.size());
    } else {
      RCLCPP_INFO(get_logger(), "GPD returned %zu grasp(s)", grasps.size());
    }

    if (grasps.empty()) {
      RCLCPP_WARN(get_logger(), "No grasps detected.");
      return;
    }

    size_t best_idx = 0;
    for (size_t i = 1; i < grasps.size(); ++i) {
      if (grasps[i].score.data > grasps[best_idx].score.data)
        best_idx = i;
    }

    const auto & g = grasps[best_idx];
    RCLCPP_INFO(get_logger(),
      "Best grasp [%zu/%zu]  score=%.4f  width=%.4f\n"
      "  position  (%.3f, %.3f, %.3f)\n"
      "  approach  (%.3f, %.3f, %.3f)\n"
      "  binormal  (%.3f, %.3f, %.3f)\n"
      "  axis      (%.3f, %.3f, %.3f)",
      best_idx, grasps.size(), g.score.data, g.width.data,
      g.position.x,  g.position.y,  g.position.z,
      g.approach.x,  g.approach.y,  g.approach.z,
      g.binormal.x,  g.binormal.y,  g.binormal.z,
      g.axis.x,      g.axis.y,      g.axis.z);

    if (dry_run_)
      RCLCPP_INFO(get_logger(), "[dry_run] Robot motion skipped.");

    best_grasp_pub_->publish(g);
    RCLCPP_INFO(get_logger(), "Published best grasp to /best_gpd_grasp");

    if (once_)
      RCLCPP_INFO(get_logger(), "once=true: keeping node alive for topic inspection");
  }

  // ── state ────────────────────────────────────────────────────────────────
  std::atomic<bool>      service_in_progress_;
  bool                   done_;
  Clock::time_point      last_request_time_;
  std::chrono::duration<double> min_interval_;

  // yolo crop state
  std::mutex   target_mutex_;
  bool         has_yolo_target_base_;
  bool         has_yolo_target_camera_;
  double       target_base_x_, target_base_y_, target_base_z_;
  double       target_camera_x_, target_camera_y_, target_camera_z_;
  std::string  target_camera_frame_from_msg_;

  // ── params ───────────────────────────────────────────────────────────────
  bool        dry_run_;
  bool        once_;
  std::string cloud_topic_;
  std::string service_name_;

  bool        use_yolo_target_crop_;
  std::string crop_target_source_;
  std::string yolo_target_3d_topic_;
  std::string yolo_target_base_topic_;
  std::string target_camera_frame_;
  std::string target_frame_;
  std::string cloud_frame_;
  double      crop_radius_x_;
  double      crop_radius_y_;
  double      crop_radius_z_;
  int         crop_min_points_;
  bool        publish_cropped_cloud_;
  std::string cropped_cloud_topic_;
  bool        auto_expand_crop_if_empty_;
  double      expanded_crop_radius_x_;
  double      expanded_crop_radius_y_;
  double      expanded_crop_radius_z_;

  // ── ROS ──────────────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         yolo_target_3d_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr         yolo_target_base_sub_;
  rclcpp::Client<DetectGrasps>::SharedPtr                        client_;
  rclcpp::Publisher<GraspConfig>::SharedPtr                      best_grasp_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr    cropped_cloud_pub_;

  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GpdClientNode>());
  rclcpp::shutdown();
  return 0;
}
