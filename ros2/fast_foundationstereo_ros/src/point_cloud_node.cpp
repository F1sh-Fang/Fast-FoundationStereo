#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <string>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace
{
struct PointXYZRGB
{
  float x;
  float y;
  float z;
  float rgb;
};
static_assert(sizeof(PointXYZRGB) == 16);

float pack_rgb(const uint8_t red, const uint8_t green, const uint8_t blue)
{
  const uint32_t packed =
    (static_cast<uint32_t>(red) << 16U) |
    (static_cast<uint32_t>(green) << 8U) |
    static_cast<uint32_t>(blue);
  float value;
  std::memcpy(&value, &packed, sizeof(value));
  return value;
}
}  // namespace

class PointCloudNode : public rclcpp::Node
{
public:
  PointCloudNode() : Node("fast_foundationstereo_point_cloud")
  {
    const auto depth_topic = declare_parameter<std::string>(
      "depth_topic", "/fast_foundationstereo/depth");
    const auto camera_info_topic = declare_parameter<std::string>(
      "camera_info_topic", "/fast_foundationstereo/camera_info");
    const auto color_topic = declare_parameter<std::string>(
      "color_image_topic", "/camera/camera/color/image_raw");
    const auto cloud_topic = declare_parameter<std::string>(
      "point_cloud_topic", "/fast_foundationstereo/points");
    color_sync_slop_ns_ = static_cast<int64_t>(
      declare_parameter<double>("color_sync_slop", 0.05) * 1e9);
    color_cache_size_ = static_cast<size_t>(std::max<int64_t>(
      1, declare_parameter<int64_t>("color_cache_size", 16)));
    stride_ = static_cast<int>(std::max<int64_t>(
      1, declare_parameter<int64_t>("point_cloud_stride", 1)));
    max_rate_ = declare_parameter<double>("point_cloud_max_rate", 0.0);

    auto sensor_qos = rclcpp::SensorDataQoS().keep_last(1);
    // Keep PointCloud2 compatible with RViz's default Reliable subscriber.
    // Depth/color inputs remain best-effort sensor streams.
    auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
    cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      cloud_topic, cloud_qos);
    depth_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic, sensor_qos,
      std::bind(&PointCloudNode::depth_callback, this, std::placeholders::_1));
    camera_info_subscription_ =
      create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic, sensor_qos,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr message) {
        camera_info_ = std::move(message);
      });
    color_subscription_ = create_subscription<sensor_msgs::msg::Image>(
      color_topic, sensor_qos,
      std::bind(&PointCloudNode::color_callback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "C++ point cloud: %s -> %s, stride=%d, max_rate=%.1f Hz",
      depth_topic.c_str(), cloud_topic.c_str(), stride_, max_rate_);
  }

private:
  static int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp)
  {
    return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
  }

  void color_callback(sensor_msgs::msg::Image::ConstSharedPtr message)
  {
    color_cache_.push_back(std::move(message));
    while (color_cache_.size() > color_cache_size_) {
      color_cache_.pop_front();
    }
  }

  sensor_msgs::msg::Image::ConstSharedPtr matching_color(
    const builtin_interfaces::msg::Time & stamp)
  {
    if (color_cache_.empty()) {
      return nullptr;
    }
    const int64_t target = stamp_ns(stamp);
    auto best = std::min_element(
      color_cache_.begin(), color_cache_.end(),
      [target](const auto & lhs, const auto & rhs) {
        return std::llabs(stamp_ns(lhs->header.stamp) - target) <
               std::llabs(stamp_ns(rhs->header.stamp) - target);
      });
    if (std::llabs(stamp_ns((*best)->header.stamp) - target) > color_sync_slop_ns_) {
      return nullptr;
    }
    return *best;
  }

  bool rate_limited(const builtin_interfaces::msg::Time & stamp)
  {
    if (max_rate_ <= 0.0) {
      return false;
    }
    const int64_t current = stamp_ns(stamp);
    const int64_t interval = static_cast<int64_t>(1e9 / max_rate_);
    if (last_published_stamp_ns_ != 0 &&
      current - last_published_stamp_ns_ < interval)
    {
      return true;
    }
    last_published_stamp_ns_ = current;
    return false;
  }

  void depth_callback(sensor_msgs::msg::Image::ConstSharedPtr depth_message)
  {
    if (cloud_publisher_->get_subscription_count() == 0 ||
      !camera_info_ || rate_limited(depth_message->header.stamp))
    {
      return;
    }
    if (depth_message->encoding != sensor_msgs::image_encodings::TYPE_32FC1) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Expected 32FC1 depth, received %s", depth_message->encoding.c_str());
      return;
    }

    const uint32_t source_width = depth_message->width;
    const uint32_t source_height = depth_message->height;
    if (camera_info_->p[0] <= 0.0 || camera_info_->p[5] <= 0.0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Invalid CameraInfo.P");
      return;
    }
    const float fx = static_cast<float>(camera_info_->p[0]);
    const float fy = static_cast<float>(camera_info_->p[5]);
    const float cx = static_cast<float>(camera_info_->p[2]);
    const float cy = static_cast<float>(camera_info_->p[6]);

    cv::Mat color;
    if (const auto color_message = matching_color(depth_message->header.stamp)) {
      try {
        color = cv_bridge::toCvCopy(
          color_message, sensor_msgs::image_encodings::RGB8)->image;
        if (color.cols != static_cast<int>(source_width) ||
          color.rows != static_cast<int>(source_height))
        {
          cv::resize(
            color, color, cv::Size(source_width, source_height), 0.0, 0.0,
            cv::INTER_LINEAR);
        }
      } catch (const cv_bridge::Exception & error) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000, "Color conversion failed: %s",
          error.what());
      }
    }

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = depth_message->header;
    cloud.height = (source_height + stride_ - 1) / stride_;
    cloud.width = (source_width + stride_ - 1) / stride_;
    cloud.fields.resize(4);
    set_field(cloud.fields[0], "x", 0);
    set_field(cloud.fields[1], "y", 4);
    set_field(cloud.fields[2], "z", 8);
    set_field(cloud.fields[3], "rgb", 12);
    cloud.is_bigendian = false;
    cloud.point_step = sizeof(PointXYZRGB);
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.is_dense = false;
    cloud.data.resize(static_cast<size_t>(cloud.row_step) * cloud.height);
    auto * points = reinterpret_cast<PointXYZRGB *>(cloud.data.data());
    const float nan = std::numeric_limits<float>::quiet_NaN();

    for (uint32_t out_v = 0; out_v < cloud.height; ++out_v) {
      const uint32_t v = out_v * stride_;
      const auto * depth_row = reinterpret_cast<const float *>(
        depth_message->data.data() + static_cast<size_t>(v) * depth_message->step);
      const auto * color_row = color.empty() ? nullptr : color.ptr<cv::Vec3b>(v);
      for (uint32_t out_u = 0; out_u < cloud.width; ++out_u) {
        const uint32_t u = out_u * stride_;
        const float z = depth_row[u];
        auto & point = points[static_cast<size_t>(out_v) * cloud.width + out_u];
        if (std::isfinite(z) && z > 0.0F) {
          point.x = (static_cast<float>(u) - cx) * z / fx;
          point.y = (static_cast<float>(v) - cy) * z / fy;
          point.z = z;
          point.rgb = color_row ?
            pack_rgb(color_row[u][0], color_row[u][1], color_row[u][2]) : 0.0F;
        } else {
          point = {nan, nan, nan, 0.0F};
        }
      }
    }
    cloud_publisher_->publish(std::move(cloud));
  }

  static void set_field(
    sensor_msgs::msg::PointField & field, const std::string & name,
    const uint32_t offset)
  {
    field.name = name;
    field.offset = offset;
    field.datatype = sensor_msgs::msg::PointField::FLOAT32;
    field.count = 1;
  }

  int64_t color_sync_slop_ns_{50000000};
  int64_t last_published_stamp_ns_{0};
  size_t color_cache_size_{16};
  int stride_{1};
  double max_rate_{0.0};
  std::deque<sensor_msgs::msg::Image::ConstSharedPtr> color_cache_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
    camera_info_subscription_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudNode>());
  rclcpp::shutdown();
  return 0;
}
