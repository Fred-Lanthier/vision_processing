cmake_minimum_required(VERSION 3.0.2)
project(vision_processing)

## [MODIF 1] Fix impératif pour pybind11 sur Noetic
cmake_policy(SET CMP0057 NEW)

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
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Fixer le chemin du python AVANT de chercher pybind11
set(PYTHON_EXECUTABLE "/home/flanthier/Github/src/vision_processing/venv_sam3/bin/python3")

# Trouver le chemin d'inclusion de Pybind11 via Python
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE PYBIND11_VENV_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(OpenCV REQUIRED)
find_package(pinocchio REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

## 1. Services et Messages
add_service_files(
  FILES
  ProcessImage.srv
  ProcessRobot.srv
  Sam3Segment.srv
)

## 2. Configuration Python (DOIT être avant generate_messages)
catkin_python_setup()

## 3. Génération des messages
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
    tf2
    tf2_ros
    tf2_geometry_msgs
)

## Inclusions
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PINOCCHIO_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PYBIND11_VENV_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

## 4. Module Pybind11 (Fast IK)
pybind11_add_module(fast_ik_module src/vision_processing/fast_ik.cpp)
target_compile_options(fast_ik_module PRIVATE -O3)

target_link_libraries(fast_ik_module PUBLIC 
    pinocchio::pinocchio 
    Eigen3::Eigen
    ${catkin_LIBRARIES}
)

set_target_properties(fast_ik_module PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}/vision_processing
)

## 5. Module Pybind11 (Fast Perception)
pybind11_add_module(fast_perception_module src/vision_processing/fast_perception.cpp)

target_link_libraries(fast_perception_module PUBLIC 
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
)

set_target_properties(fast_perception_module PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}/vision_processing
)

## Autres exécutables
add_executable(franka_state_camera_reader src/vision_processing/gazebo_simulation/franka_state_camera_reader.cpp)
target_link_libraries(
  franka_state_camera_reader 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

## Installation
catkin_install_python(PROGRAMS
  scripts/vision_server_food_segmentation.py
  scripts/vision_server_robot_segmentation.py
  scripts/trajectory_node.py
  scripts/inference_node.py
  scripts/casf_generative_node_PC.py
  scripts/safe_generative_node.py
  scripts/condition_pcd_from_perception.py
  scripts/point_cloud_projector_node.py
  scripts/persistent_cloud_node.py
  scripts/nominal_fork_pose_follower.py
  scripts/nominal_trajectory_follower.py
  scripts/perception_processing_node.py
  scripts/trajectory_executor.py
  scripts/trajectory_controller_bridge.py
  scripts/circle_trajectory_publisher.py
  scripts/face_detection_mock.py
  scripts/fake_finger_publisher.py
  scripts/sam3_server.py
  test_novelty/cbf_safety_node_Bernstein.py
  test_novelty/sam3_sam2_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(TARGETS fast_ik_module
  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

install(TARGETS fast_perception_module
  LIBRARY DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
