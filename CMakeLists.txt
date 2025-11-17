cmake_minimum_required(VERSION 3.0.2)
project(vision_processing)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  message_generation
)

## Add service files
add_service_files(
  FILES
  ProcessImage.srv
)

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
    message_runtime
)
set(PYTHON_EXECUTABLE "/home/flanthier/Documents/src/vision_processing/venv_py310/bin/python3")

catkin_install_python(PROGRAMS
  scripts/vision_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
