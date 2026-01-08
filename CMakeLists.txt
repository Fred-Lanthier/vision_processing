cmake_minimum_required(VERSION 3.0.2)
project(vision_processing)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  geometry_msgs
  moveit_core
  moveit_ros_planning
  moveit_ros_planning_interface
  message_generation
  message_filters
  eigen_conversions
)

find_package(OpenCV REQUIRED)

## Add service files
add_service_files(
  FILES
  ProcessImage.srv
  ProcessRobot.srv
)

## Python setup
catkin_python_setup()

## Generate added messages and services
generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS 
    rospy 
    std_msgs 
    sensor_msgs 
    cv_bridge
    geometry_msgs
    moveit_core
    moveit_ros_planning
    moveit_ros_planning_interface
    message_runtime
    eigen_conversions
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(franka_state_camera_reader src/vision_processing/gazebo_simulation/franka_state_camera_reader.cpp)

target_link_libraries(
  franka_state_camera_reader 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)
set(PYTHON_EXECUTABLE "/home/flanthier/Github/src/vision_processing/venv_sam3/bin/python3")

catkin_install_python(PROGRAMS
  scripts/vision_server_food_segmentation.py
  scripts/vision_server_robot_segmentation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
