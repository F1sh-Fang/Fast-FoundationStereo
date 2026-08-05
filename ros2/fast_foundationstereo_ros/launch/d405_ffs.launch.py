from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
        DeclareLaunchArgument("image_endpoint", default_value="tcp://0.0.0.0:5560"),
        DeclareLaunchArgument("result_endpoint", default_value="tcp://127.0.0.1:5561"),
        DeclareLaunchArgument("publish_point_cloud", default_value="false"),
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
            "left_camera_info_topic": LaunchConfiguration("left_camera_info_topic"),
            "right_camera_info_topic": LaunchConfiguration("right_camera_info_topic"),
            "image_endpoint": LaunchConfiguration("image_endpoint"),
            "result_endpoint": LaunchConfiguration("result_endpoint"),
            "publish_point_cloud": LaunchConfiguration("publish_point_cloud"),
            "baseline_override": LaunchConfiguration("baseline_override"),
            "approximate_sync": LaunchConfiguration("approximate_sync"),
        }],
    )
    return LaunchDescription(arguments + [bridge])
