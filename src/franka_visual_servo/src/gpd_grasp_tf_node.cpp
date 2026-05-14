#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <gpd_ros2_msgs/msg/grasp_config.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using GraspConfig = gpd_ros2_msgs::msg::GraspConfig;

class GpdGraspTfNode : public rclcpp::Node
{
public:
  GpdGraspTfNode()
  : Node("gpd_grasp_tf_node"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter<std::string>("input_frame",  "camera_depth_optical_frame");
    declare_parameter<std::string>("target_frame", "fr3_link0");
    declare_parameter<std::string>("input_topic",  "/best_gpd_grasp");
    declare_parameter<std::string>("output_topic", "/best_gpd_grasp_fr3");
    declare_parameter<bool>("dry_run", true);
    declare_parameter<bool>("use_transient_local_qos", false);

    // empirical linear calibration (same method as yolo_hover_linear_node)
    declare_parameter<bool>("use_target_linear_offset", false);
    declare_parameter<double>("x_offset_kx", 0.0);
    declare_parameter<double>("x_offset_ky", 0.0);
    declare_parameter<double>("x_offset_b",  0.0);
    declare_parameter<double>("y_offset_kx", 0.0);
    declare_parameter<double>("y_offset_ky", 0.0);
    declare_parameter<double>("y_offset_b",  0.0);

    // hover / pregrasp safety
    declare_parameter<bool>("hover_only",       false);
    declare_parameter<double>("pregrasp_z_offset", 0.10);
    declare_parameter<double>("min_safe_z",        0.05);

    input_frame_  = get_parameter("input_frame").as_string();
    target_frame_ = get_parameter("target_frame").as_string();
    dry_run_      = get_parameter("dry_run").as_bool();
    use_transient_local_qos_ = get_parameter("use_transient_local_qos").as_bool();

    use_target_linear_offset_ = get_parameter("use_target_linear_offset").as_bool();
    x_offset_kx_ = get_parameter("x_offset_kx").as_double();
    x_offset_ky_ = get_parameter("x_offset_ky").as_double();
    x_offset_b_  = get_parameter("x_offset_b").as_double();
    y_offset_kx_ = get_parameter("y_offset_kx").as_double();
    y_offset_ky_ = get_parameter("y_offset_ky").as_double();
    y_offset_b_  = get_parameter("y_offset_b").as_double();

    hover_only_        = get_parameter("hover_only").as_bool();
    pregrasp_z_offset_ = get_parameter("pregrasp_z_offset").as_double();
    min_safe_z_        = get_parameter("min_safe_z").as_double();

    const auto input_topic  = get_parameter("input_topic").as_string();
    const auto output_topic = get_parameter("output_topic").as_string();

    auto sub_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    if (use_transient_local_qos_) {
      sub_qos.transient_local();
    }
    sub_ = create_subscription<GraspConfig>(
      input_topic, sub_qos,
      std::bind(&GpdGraspTfNode::graspCallback, this, std::placeholders::_1));

    auto pub_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    if (use_transient_local_qos_) {
      pub_qos.transient_local();
    }
    pub_ = create_publisher<GraspConfig>(output_topic, pub_qos);
    if (use_transient_local_qos_) {
      RCLCPP_INFO(
        get_logger(),
        "[GPD calibration] publishing %s with transient_local QoS",
        output_topic.c_str());
    } else {
      RCLCPP_INFO(
        get_logger(),
        "[GPD calibration] publishing %s with default QoS",
        output_topic.c_str());
    }

    RCLCPP_INFO(get_logger(),
      "gpd_grasp_tf_node ready.  %s -> %s  input=%s  output=%s  dry_run=%s"
      "  use_linear_offset=%s  hover_only=%s  transient_local_qos=%s",
      input_frame_.c_str(), target_frame_.c_str(),
      input_topic.c_str(), output_topic.c_str(),
      dry_run_ ? "true" : "false",
      use_target_linear_offset_ ? "true" : "false",
      hover_only_ ? "true" : "false",
      use_transient_local_qos_ ? "true" : "false");
  }

private:
  bool transformPoint(
    const geometry_msgs::msg::Point & in,
    geometry_msgs::msg::Point & out,
    const rclcpp::Time & stamp)
  {
    geometry_msgs::msg::PointStamped ps_in, ps_out;
    ps_in.header.frame_id = input_frame_;
    ps_in.header.stamp    = stamp;
    ps_in.point           = in;
    try {
      tf_buffer_.transform(ps_in, ps_out, target_frame_,
                           tf2::durationFromSec(0.5));
    } catch (const tf2::TransformException & e) {
      RCLCPP_ERROR(get_logger(), "Point transform failed: %s", e.what());
      return false;
    }
    out = ps_out.point;
    return true;
  }

  bool transformVector(
    const geometry_msgs::msg::Vector3 & in,
    geometry_msgs::msg::Vector3 & out,
    const rclcpp::Time & stamp)
  {
    geometry_msgs::msg::Vector3Stamped vs_in, vs_out;
    vs_in.header.frame_id = input_frame_;
    vs_in.header.stamp    = stamp;
    vs_in.vector          = in;
    try {
      tf_buffer_.transform(vs_in, vs_out, target_frame_,
                           tf2::durationFromSec(0.5));
    } catch (const tf2::TransformException & e) {
      RCLCPP_ERROR(get_logger(), "Vector transform failed: %s", e.what());
      return false;
    }
    out = vs_out.vector;
    return true;
  }

  void graspCallback(const GraspConfig::SharedPtr msg)
  {
    RCLCPP_INFO(get_logger(),
      "Received best grasp in %s  pos=(%.3f, %.3f, %.3f)  score=%.4f",
      input_frame_.c_str(),
      msg->position.x, msg->position.y, msg->position.z,
      msg->score.data);
    RCLCPP_INFO(get_logger(), "Transform target frame: %s", target_frame_.c_str());

    const auto stamp = now();

    // --- transform position ---
    geometry_msgs::msg::Point pos_out;
    if (!transformPoint(msg->position, pos_out, stamp)) return;

    RCLCPP_INFO(get_logger(),
      "[GPD calibration] raw grasp base position: x=%.4f, y=%.4f, z=%.4f",
      pos_out.x, pos_out.y, pos_out.z);

    // --- apply empirical linear calibration (Method A) ---
    if (use_target_linear_offset_) {
      const double raw_x = pos_out.x;
      const double raw_y = pos_out.y;
      const double x_offset = x_offset_kx_ * raw_x + x_offset_ky_ * raw_y + x_offset_b_;
      const double y_offset = y_offset_kx_ * raw_x + y_offset_ky_ * raw_y + y_offset_b_;
      pos_out.x += x_offset;
      pos_out.y += y_offset;
      RCLCPP_INFO(get_logger(),
        "[GPD calibration] offset: dx=%.4f, dy=%.4f  corrected: x=%.4f, y=%.4f, z=%.4f",
        x_offset, y_offset, pos_out.x, pos_out.y, pos_out.z);
    }

    // --- safety: clamp z above min_safe_z ---
    if (pos_out.z < min_safe_z_) {
      RCLCPP_WARN(get_logger(),
        "[GPD safety] z=%.4f is below min_safe_z=%.4f, clamping.",
        pos_out.z, min_safe_z_);
      pos_out.z = min_safe_z_;
    }

    // --- hover_only: lift to pregrasp height ---
    geometry_msgs::msg::Point pos_publish = pos_out;
    if (hover_only_) {
      pos_publish.z = pos_out.z + pregrasp_z_offset_;
      RCLCPP_INFO(get_logger(),
        "[GPD hover_only] pregrasp position: x=%.4f, y=%.4f, z=%.4f  (offset +%.3f)",
        pos_publish.x, pos_publish.y, pos_publish.z, pregrasp_z_offset_);
    }

    // --- transform orientation vectors ---
    geometry_msgs::msg::Vector3 approach_out, binormal_out, axis_out;
    if (!transformVector(msg->approach, approach_out, stamp)) return;
    if (!transformVector(msg->binormal, binormal_out, stamp)) return;
    if (!transformVector(msg->axis,     axis_out,     stamp)) return;

    RCLCPP_INFO(get_logger(),
      "Transformed approach  (%.3f, %.3f, %.3f)",
      approach_out.x, approach_out.y, approach_out.z);
    RCLCPP_INFO(get_logger(),
      "Transformed binormal  (%.3f, %.3f, %.3f)",
      binormal_out.x, binormal_out.y, binormal_out.z);
    RCLCPP_INFO(get_logger(),
      "Transformed axis      (%.3f, %.3f, %.3f)",
      axis_out.x, axis_out.y, axis_out.z);

    // --- transform sample point ---
    geometry_msgs::msg::Point sample_out;
    const bool sample_ok = transformPoint(msg->sample, sample_out, stamp);

    // --- publish ---
    GraspConfig out;
    out.position = pos_publish;
    out.approach = approach_out;
    out.binormal = binormal_out;
    out.axis     = axis_out;
    out.width    = msg->width;
    out.score    = msg->score;
    out.sample   = sample_ok ? sample_out : msg->sample;

    pub_->publish(out);
    RCLCPP_INFO(get_logger(), "Published transformed grasp to %s",
      pub_->get_topic_name());

    if (dry_run_)
      RCLCPP_INFO(get_logger(), "[dry_run] No robot motion will be executed.");
  }

  bool        dry_run_;
  bool        use_transient_local_qos_;
  std::string input_frame_;
  std::string target_frame_;

  bool   use_target_linear_offset_;
  double x_offset_kx_, x_offset_ky_, x_offset_b_;
  double y_offset_kx_, y_offset_ky_, y_offset_b_;

  bool   hover_only_;
  double pregrasp_z_offset_;
  double min_safe_z_;

  tf2_ros::Buffer            tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Subscription<GraspConfig>::SharedPtr sub_;
  rclcpp::Publisher<GraspConfig>::SharedPtr    pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GpdGraspTfNode>());
  rclcpp::shutdown();
  return 0;
}
