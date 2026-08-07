cmake_minimum_required(VERSION 3.8)
project(fast_foundationstereo_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

add_executable(point_cloud_node src/point_cloud_node.cpp)
target_compile_features(point_cloud_node PUBLIC cxx_std_17)
target_compile_options(point_cloud_node PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3>
)
ament_target_dependencies(point_cloud_node cv_bridge rclcpp sensor_msgs)

install(TARGETS point_cloud_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS
  fast_foundationstereo_ros/stereo_zmq_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME stereo_zmq_bridge
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

ament_package()
