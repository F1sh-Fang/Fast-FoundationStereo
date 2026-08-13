#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>


class DepthAlignNode : public rclcpp::Node
{
public:
  DepthAlignNode()
  : Node("fast_foundationstereo_depth_align"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    const auto depth_topic = declare_parameter<std::string>(
      "depth_topic", "/fast_foundationstereo/depth");
    const auto source_info_topic = declare_parameter<std::string>(
      "source_camera_info_topic", "/fast_foundationstereo/camera_info");
    const auto target_info_topic = declare_parameter<std::string>(
      "target_camera_info_topic", "/camera/camera/color/camera_info");
    const auto aligned_depth_topic = declare_parameter<std::string>(
      "aligned_depth_topic", "/fast_foundationstereo/aligned_depth_to_color");
    const auto aligned_info_topic = declare_parameter<std::string>(
      "aligned_camera_info_topic",
      "/fast_foundationstereo/aligned_depth_to_color/camera_info");
    performance_log_interval_ = declare_parameter<int>(
      "performance_log_interval", 0);

    const auto sensor_qos = rclcpp::SensorDataQoS().keep_last(1);
    source_info_subscription_ =
      create_subscription<sensor_msgs::msg::CameraInfo>(
      source_info_topic, sensor_qos,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        const bool changed = !source_info_ ||
          source_info_->width != message->width ||
          source_info_->height != message->height ||
          source_info_->header.frame_id != message->header.frame_id ||
          source_info_->k != message->k || source_info_->p != message->p;
        source_info_ = std::move(message);
        calibration_ready_ = calibration_ready_ && !changed;
      });
    target_info_subscription_ =
      create_subscription<sensor_msgs::msg::CameraInfo>(
      target_info_topic, sensor_qos,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        const bool changed = !target_info_ ||
          target_info_->width != message->width ||
          target_info_->height != message->height ||
          target_info_->header.frame_id != message->header.frame_id ||
          target_info_->distortion_model != message->distortion_model ||
          target_info_->k != message->k || target_info_->d != message->d;
        target_info_ = std::move(message);
        calibration_ready_ = calibration_ready_ && !changed;
      });
    depth_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic, sensor_qos,
      std::bind(&DepthAlignNode::depth_callback, this, std::placeholders::_1));

    aligned_depth_publisher_ = create_publisher<sensor_msgs::msg::Image>(
      aligned_depth_topic, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort());
    aligned_info_publisher_ =
      create_publisher<sensor_msgs::msg::CameraInfo>(
      aligned_info_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable());

    RCLCPP_INFO(
      get_logger(), "C++ depth alignment: %s -> %s",
      depth_topic.c_str(), aligned_depth_topic.c_str());
  }

private:
  struct ProjectedRay
  {
    float x;
    float y;
    float z;
  };

  bool prepare_calibration(const std::string & source_frame)
  {
    if (!source_info_ || !target_info_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Waiting for source and color CameraInfo");
      return false;
    }
    if (source_info_->width == 0 || source_info_->height == 0 ||
      target_info_->width == 0 || target_info_->height == 0)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "CameraInfo dimensions are invalid");
      return false;
    }
    if (!target_info_->distortion_model.empty() &&
      target_info_->distortion_model != "plumb_bob" &&
      target_info_->distortion_model != "rational_polynomial")
    {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Unsupported color distortion model: %s",
        target_info_->distortion_model.c_str());
      return false;
    }

    const double source_fx = source_info_->p[0] > 0.0 ?
      source_info_->p[0] : source_info_->k[0];
    const double source_fy = source_info_->p[5] > 0.0 ?
      source_info_->p[5] : source_info_->k[4];
    const double source_cx = source_info_->p[0] > 0.0 ?
      source_info_->p[2] : source_info_->k[2];
    const double source_cy = source_info_->p[5] > 0.0 ?
      source_info_->p[6] : source_info_->k[5];
    if (source_fx <= 0.0 || source_fy <= 0.0 ||
      target_info_->k[0] <= 0.0 || target_info_->k[4] <= 0.0)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "CameraInfo intrinsics are invalid");
      return false;
    }

    const std::string target_frame = target_info_->header.frame_id;
    if (source_frame.empty() || target_frame.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "Camera frame_id is empty");
      return false;
    }
    try {
      const auto transform = tf_buffer_.lookupTransform(
        target_frame, source_frame, tf2::TimePointZero);
      const auto & q_msg = transform.transform.rotation;
      tf2::Quaternion quaternion(q_msg.x, q_msg.y, q_msg.z, q_msg.w);
      quaternion.normalize();
      const tf2::Matrix3x3 rotation(quaternion);
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          rotation_[row * 3 + column] = static_cast<float>(
            rotation[row][column]);
        }
      }
      const auto & translation = transform.transform.translation;
      translation_ = {
        static_cast<float>(translation.x),
        static_cast<float>(translation.y),
        static_cast<float>(translation.z)};
    } catch (const tf2::TransformException & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Waiting for %s <- %s TF: %s", target_frame.c_str(),
        source_frame.c_str(), error.what());
      return false;
    }

    source_width_ = source_info_->width;
    source_height_ = source_info_->height;
    target_width_ = target_info_->width;
    target_height_ = target_info_->height;
    projected_rays_.resize(
      static_cast<size_t>(source_width_) * source_height_);
    for (uint32_t v = 0; v < source_height_; ++v) {
      for (uint32_t u = 0; u < source_width_; ++u) {
        const float ray_x = static_cast<float>(
          (static_cast<double>(u) - source_cx) / source_fx);
        const float ray_y = static_cast<float>(
          (static_cast<double>(v) - source_cy) / source_fy);
        auto & ray = projected_rays_[
          static_cast<size_t>(v) * source_width_ + u];
        ray.x = rotation_[0] * ray_x + rotation_[1] * ray_y + rotation_[2];
        ray.y = rotation_[3] * ray_x + rotation_[4] * ray_y + rotation_[5];
        ray.z = rotation_[6] * ray_x + rotation_[7] * ray_y + rotation_[8];
      }
    }
    z_buffer_.resize(static_cast<size_t>(target_width_) * target_height_);

    aligned_message_.height = target_height_;
    aligned_message_.width = target_width_;
    aligned_message_.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    aligned_message_.is_bigendian = false;
    aligned_message_.step = target_width_ * sizeof(float);
    aligned_message_.data.resize(z_buffer_.size() * sizeof(float));

    target_fx_ = static_cast<float>(target_info_->k[0]);
    target_skew_ = static_cast<float>(target_info_->k[1]);
    target_cx_ = static_cast<float>(target_info_->k[2]);
    target_fy_ = static_cast<float>(target_info_->k[4]);
    target_cy_ = static_cast<float>(target_info_->k[5]);
    distortion_.fill(0.0F);
    const size_t coefficient_count = std::min<size_t>(
      distortion_.size(), target_info_->d.size());
    for (size_t index = 0; index < coefficient_count; ++index) {
      distortion_[index] = static_cast<float>(target_info_->d[index]);
    }
    has_distortion_ = std::any_of(
      distortion_.begin(), distortion_.end(),
      [](float coefficient) {return coefficient != 0.0F;});
    use_rational_distortion_ = target_info_->d.size() >= 8;
    target_frame_ = target_frame;
    source_frame_ = source_frame;
    aligned_info_ = *target_info_;
    aligned_info_.header.frame_id = target_frame_;
    calibration_ready_ = true;
    RCLCPP_INFO(
      get_logger(), "Depth alignment calibrated: %ux%u %s -> %ux%u %s",
      source_width_, source_height_, source_frame_.c_str(), target_width_,
      target_height_, target_frame_.c_str());
    return true;
  }

  void depth_callback(sensor_msgs::msg::Image::ConstSharedPtr depth_message)
  {
    const bool measure_performance = performance_log_interval_ > 0;
    std::chrono::steady_clock::time_point processing_start;
    if (measure_performance) {
      processing_start = std::chrono::steady_clock::now();
    }
    if (depth_message->encoding != sensor_msgs::image_encodings::TYPE_32FC1) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Expected 32FC1 depth, received %s", depth_message->encoding.c_str());
      return;
    }
    if (depth_message->is_bigendian) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Big-endian depth images are not supported");
      return;
    }
    const std::string source_frame = depth_message->header.frame_id.empty() &&
      source_info_ ? source_info_->header.frame_id : depth_message->header.frame_id;
    if (!calibration_ready_ || source_frame != source_frame_ ||
      depth_message->width != source_width_ ||
      depth_message->height != source_height_)
    {
      if (!prepare_calibration(source_frame)) {
        return;
      }
    }
    if (depth_message->width != source_width_ ||
      depth_message->height != source_height_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Depth dimensions %ux%u do not match source CameraInfo %ux%u",
        depth_message->width, depth_message->height, source_width_,
        source_height_);
      return;
    }
    const size_t row_bytes = static_cast<size_t>(source_width_) * sizeof(float);
    const size_t required_bytes = source_height_ == 0 ? 0 :
      (static_cast<size_t>(source_height_ - 1) * depth_message->step) + row_bytes;
    if (depth_message->step < row_bytes ||
      depth_message->step % alignof(float) != 0 ||
      reinterpret_cast<uintptr_t>(depth_message->data.data()) % alignof(float) != 0 ||
      depth_message->data.size() < required_bytes)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Depth image buffer layout is invalid for 32FC1 dimensions and step");
      return;
    }

    std::fill(z_buffer_.begin(), z_buffer_.end(), 0.0F);
    for (uint32_t v = 0; v < source_height_; ++v) {
        const auto * depth_row = reinterpret_cast<const float *>(
          depth_message->data.data() + static_cast<size_t>(v) * depth_message->step);
        for (uint32_t u = 0; u < source_width_; ++u) {
        const float source_z = depth_row[u];
        if (!std::isfinite(source_z) || source_z <= 0.0F) {
          continue;
        }
        const auto & ray = projected_rays_[
          static_cast<size_t>(v) * source_width_ + u];
        const float target_x = ray.x * source_z + translation_[0];
        const float target_y = ray.y * source_z + translation_[1];
        const float target_z = ray.z * source_z + translation_[2];
        if (!std::isfinite(target_z) || target_z <= 0.0F) {
          continue;
        }

        const float inverse_z = 1.0F / target_z;
        const float normalized_x = target_x * inverse_z;
        const float normalized_y = target_y * inverse_z;
        float distorted_x = normalized_x;
        float distorted_y = normalized_y;
        if (has_distortion_) {
          const float radius_2 = normalized_x * normalized_x +
            normalized_y * normalized_y;
          float radial = std::fma(
            radius_2,
            std::fma(radius_2, std::fma(distortion_[4], radius_2,
              distortion_[1]), distortion_[0]),
            1.0F);
          if (use_rational_distortion_) {
            const float denominator = std::fma(
              radius_2,
              std::fma(radius_2, std::fma(distortion_[7], radius_2,
                distortion_[6]), distortion_[5]),
              1.0F);
            radial /= denominator;
          }
          distorted_x = normalized_x * radial +
            2.0F * distortion_[2] * normalized_x * normalized_y +
            distortion_[3] *
            (radius_2 + 2.0F * normalized_x * normalized_x);
          distorted_y = normalized_y * radial +
            distortion_[2] *
            (radius_2 + 2.0F * normalized_y * normalized_y) +
            2.0F * distortion_[3] * normalized_x * normalized_y;
        }
        const float pixel_u = target_fx_ * distorted_x +
          target_skew_ * distorted_y + target_cx_;
        const float pixel_v = target_fy_ * distorted_y + target_cy_;
        if (!(pixel_u >= -0.5F &&
          pixel_u < static_cast<float>(target_width_) - 0.5F &&
          pixel_v >= -0.5F &&
          pixel_v < static_cast<float>(target_height_) - 0.5F))
        {
          continue;
        }
        const auto target_u = static_cast<uint32_t>(pixel_u + 0.5F);
        const auto target_v = static_cast<uint32_t>(pixel_v + 0.5F);
        auto & output_z = z_buffer_[
          static_cast<size_t>(target_v) * target_width_ + target_u];
        if (output_z == 0.0F || target_z < output_z) {
          output_z = target_z;
        }
        }
      }

    aligned_message_.header = depth_message->header;
    aligned_message_.header.frame_id = target_frame_;
    std::memcpy(
      aligned_message_.data.data(), z_buffer_.data(), aligned_message_.data.size());
    aligned_depth_publisher_->publish(aligned_message_);

    aligned_info_.header = depth_message->header;
    aligned_info_.header.frame_id = target_frame_;
    aligned_info_publisher_->publish(aligned_info_);

    if (measure_performance &&
      ++processed_frames_ % static_cast<uint64_t>(performance_log_interval_) == 0)
    {
      const auto elapsed = std::chrono::steady_clock::now() - processing_start;
      const double milliseconds =
        std::chrono::duration<double, std::milli>(elapsed).count();
      RCLCPP_INFO(
        get_logger(), "Depth alignment processing: %.3f ms (%ux%u -> %ux%u)",
        milliseconds, source_width_, source_height_, target_width_, target_height_);
    }
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr source_info_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr target_info_;
  bool calibration_ready_{false};
  bool has_distortion_{false};
  bool use_rational_distortion_{false};
  int performance_log_interval_{0};
  uint64_t processed_frames_{0};
  uint32_t source_width_{0};
  uint32_t source_height_{0};
  uint32_t target_width_{0};
  uint32_t target_height_{0};
  float target_fx_{0.0F};
  float target_fy_{0.0F};
  float target_skew_{0.0F};
  float target_cx_{0.0F};
  float target_cy_{0.0F};
  std::array<float, 3> translation_{};
  std::array<float, 8> distortion_{};
  std::array<float, 9> rotation_{};
  std::string source_frame_;
  std::string target_frame_;
  std::vector<ProjectedRay> projected_rays_;
  std::vector<float> z_buffer_;
  sensor_msgs::msg::Image aligned_message_;
  sensor_msgs::msg::CameraInfo aligned_info_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
    source_info_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
    target_info_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr aligned_depth_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr
    aligned_info_publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthAlignNode>());
  rclcpp::shutdown();
  return 0;
}
