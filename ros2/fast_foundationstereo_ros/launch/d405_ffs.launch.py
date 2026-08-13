from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument(
            "left_image_topic",
            default_value="/camera/camera/infra1/image_rect_raw",
        ),
        DeclareLaunchArgument(
            "right_image_topic",
            default_value="/camera/camera/infra2/image_rect_raw",
        ),
        DeclareLaunchArgument(
            "left_camera_info_topic",
            default_value="/camera/camera/infra1/camera_info",
        ),
        DeclareLaunchArgument(
            "right_camera_info_topic",
            default_value="/camera/camera/infra2/camera_info",
        ),
        DeclareLaunchArgument(
            "color_image_topic",
            default_value="/camera/camera/color/image_raw",
        ),
        DeclareLaunchArgument(
            "color_camera_info_topic",
            default_value="/camera/camera/color/camera_info",
        ),
        DeclareLaunchArgument("align_depth_to_color", default_value="true"),
        DeclareLaunchArgument(
            "aligned_depth_topic",
            default_value="/fast_foundationstereo/aligned_depth_to_color",
        ),
        DeclareLaunchArgument(
            "aligned_camera_info_topic",
            default_value=(
                "/fast_foundationstereo/aligned_depth_to_color/camera_info"
            ),
        ),
        DeclareLaunchArgument("color_sync_slop", default_value="0.05"),
        DeclareLaunchArgument(
            "image_endpoint", default_value="tcp://0.0.0.0:5560"
        ),
        DeclareLaunchArgument(
            "result_endpoint", default_value="tcp://127.0.0.1:5561"
        ),
        DeclareLaunchArgument("publish_point_cloud", default_value="false"),
        DeclareLaunchArgument("point_cloud_stride", default_value="1"),
        DeclareLaunchArgument("point_cloud_max_rate", default_value="0.0"),
        DeclareLaunchArgument("baseline_override", default_value="0.0"),
        DeclareLaunchArgument("approximate_sync", default_value="false"),
    ]

    bridge = Node(
        package="fast_foundationstereo_ros",
        executable="stereo_zmq_bridge",
        name="fast_foundationstereo_zmq_bridge",
        output="screen",
        parameters=[{
            "left_image_topic": LaunchConfiguration("left_image_topic"),
            "right_image_topic": LaunchConfiguration("right_image_topic"),
            "left_camera_info_topic": LaunchConfiguration(
                "left_camera_info_topic"
            ),
            "right_camera_info_topic": LaunchConfiguration(
                "right_camera_info_topic"
            ),
            "color_image_topic": LaunchConfiguration("color_image_topic"),
            "color_sync_slop": LaunchConfiguration("color_sync_slop"),
            "image_endpoint": LaunchConfiguration("image_endpoint"),
            "result_endpoint": LaunchConfiguration("result_endpoint"),
            # PointCloud2 runs in C++ so serialization cannot block this bridge.
            "publish_point_cloud": False,
            "baseline_override": LaunchConfiguration("baseline_override"),
            "approximate_sync": LaunchConfiguration("approximate_sync"),
        }],
    )
    depth_align = Node(
        package="fast_foundationstereo_ros",
        executable="depth_align_node",
        name="fast_foundationstereo_depth_align",
        output="screen",
        condition=IfCondition(LaunchConfiguration("align_depth_to_color")),
        parameters=[{
            "depth_topic": "/fast_foundationstereo/depth",
            "source_camera_info_topic": "/fast_foundationstereo/camera_info",
            "target_camera_info_topic": LaunchConfiguration(
                "color_camera_info_topic"
            ),
            "aligned_depth_topic": LaunchConfiguration(
                "aligned_depth_topic"
            ),
            "aligned_camera_info_topic": LaunchConfiguration(
                "aligned_camera_info_topic"
            ),
        }],
    )
    point_cloud = Node(
        package="fast_foundationstereo_ros",
        executable="point_cloud_node",
        name="fast_foundationstereo_point_cloud",
        output="screen",
        condition=IfCondition(LaunchConfiguration("publish_point_cloud")),
        parameters=[{
            "depth_topic": LaunchConfiguration("aligned_depth_topic"),
            "camera_info_topic": LaunchConfiguration(
                "aligned_camera_info_topic"
            ),
            "color_image_topic": LaunchConfiguration(
                "color_image_topic"
            ),
            "point_cloud_topic": "/fast_foundationstereo/points",
            "color_sync_slop": LaunchConfiguration("color_sync_slop"),
            "point_cloud_stride": LaunchConfiguration("point_cloud_stride"),
            "point_cloud_max_rate": LaunchConfiguration("point_cloud_max_rate"),
        }],
    )
    return LaunchDescription(arguments + [bridge, depth_align, point_cloud])
