/**
 * @file franka_state_camera_reader.cpp
 * @brief Lecture complète de l'état du robot Franka + caméra
 */

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>

class FrankaStateCameraReader {
public:
  FrankaStateCameraReader() : has_new_image_(false), has_new_depth_(false) {
    ros::NodeHandle nh;

    // Load robot model
    ROS_INFO("Loading robot model...");
    robot_model_loader::RobotModelLoader robot_model_loader(
        "robot_description");
    robot_model_ = robot_model_loader.getModel();

    if (!robot_model_) {
      ROS_ERROR("Failed to load robot model!");
      return;
    }

    robot_state_.reset(new moveit::core::RobotState(robot_model_));
    robot_state_->setToDefaultValues();

    joint_model_group_ = robot_model_->getJointModelGroup("panda_arm");

    ee_link_ = robot_model_->getLinkModel("panda_hand_tcp");
    if (!ee_link_) {
      ROS_WARN("Using panda_link8 as end effector");
      ee_link_ = robot_model_->getLinkModel("panda_link8");
    }

    ROS_INFO("Robot model loaded!");

    // Publishers pour les données du robot
    ee_pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>(
        "franka/end_effector_pose", 10);
    ee_velocity_pub_ = nh.advertise<geometry_msgs::TwistStamped>(
        "franka/end_effector_velocity", 10);

    // Subscribers
    joint_state_sub_ =
        nh.subscribe("/joint_states", 10,
                     &FrankaStateCameraReader::jointStateCallback, this);

    // Camera subscribers
    camera_rgb_sub_ =
        nh.subscribe("/camera_wrist/color/image_raw", 10,
                     &FrankaStateCameraReader::cameraRGBCallback, this);
    camera_depth_sub_ =
        nh.subscribe("/camera_wrist/aligned_depth_to_color/image_raw", 10,
                     &FrankaStateCameraReader::cameraDepthCallback, this);

    ROS_INFO("Franka State + Camera Reader initialized!");
    ROS_INFO("Subscribing to:");
    ROS_INFO("  - /joint_states");
    ROS_INFO("  - /camera_wrist/color/image_raw");
    ROS_INFO("  - /camera_wrist/aligned_depth_to_color/image_raw");
  }

  void jointStateCallback(const sensor_msgs::JointState::ConstPtr &msg) {
    // Extract panda joints
    std::vector<double> joint_positions;
    std::vector<double> joint_velocities;

    for (size_t i = 0; i < msg->name.size(); ++i) {
      if (msg->name[i].find("panda_joint") != std::string::npos) {
        joint_positions.push_back(msg->position[i]);
        if (!msg->velocity.empty())
          joint_velocities.push_back(msg->velocity[i]);
      }
    }

    if (joint_positions.size() != 7)
      return;

    // Store joint angles
    {
      std::lock_guard<std::mutex> lock(joint_mutex_);
      current_joint_angles_ = joint_positions;
      current_joint_velocities_ = joint_velocities;
      current_stamp_ = msg->header.stamp;
    }

    // Update robot state
    robot_state_->setJointGroupPositions(joint_model_group_, joint_positions);
    if (joint_velocities.size() == 7)
      robot_state_->setJointGroupVelocities(joint_model_group_,
                                            joint_velocities);

    robot_state_->update();

    // Publish end effector pose
    publishEndEffectorPose(msg->header.stamp);

    // Publish end effector velocity if we have velocities
    if (joint_velocities.size() == 7)
      publishEndEffectorVelocity(msg->header.stamp);
  }

  void cameraRGBCallback(const sensor_msgs::Image::ConstPtr &msg) {
    try {
      cv_bridge::CvImagePtr cv_ptr =
          cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

      std::lock_guard<std::mutex> lock(image_mutex_);
      current_rgb_image_ = cv_ptr->image.clone();
      has_new_image_ = true;
      image_stamp_ = msg->header.stamp;
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  void cameraDepthCallback(const sensor_msgs::Image::ConstPtr &msg) {
    try {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg);

      std::lock_guard<std::mutex> lock(depth_mutex_);
      current_depth_image_ = cv_ptr->image.clone();
      has_new_depth_ = true;
      depth_stamp_ = msg->header.stamp;
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  void publishEndEffectorPose(const ros::Time &stamp) {
    const Eigen::Isometry3d &ee_transform =
        robot_state_->getGlobalLinkTransform(ee_link_);

    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = robot_model_->getModelFrame();

    pose_msg.pose.position.x = ee_transform.translation().x();
    pose_msg.pose.position.y = ee_transform.translation().y();
    pose_msg.pose.position.z = ee_transform.translation().z();

    Eigen::Quaterniond quat(ee_transform.rotation());
    pose_msg.pose.orientation.x = quat.x();
    pose_msg.pose.orientation.y = quat.y();
    pose_msg.pose.orientation.z = quat.z();
    pose_msg.pose.orientation.w = quat.w();

    ee_pose_pub_.publish(pose_msg);
  }

  void publishEndEffectorVelocity(const ros::Time &stamp) {
    Eigen::Vector3d reference_point(0.0, 0.0, 0.0);
    Eigen::MatrixXd jacobian;
    robot_state_->getJacobian(joint_model_group_, ee_link_, reference_point,
                              jacobian);

    std::vector<double> joint_velocities;
    robot_state_->copyJointGroupVelocities(joint_model_group_,
                                           joint_velocities);

    if (joint_velocities.size() != 7)
      return;

    Eigen::VectorXd joint_vel_eigen(7);
    for (size_t i = 0; i < 7; ++i)
      joint_vel_eigen[i] = joint_velocities[i];

    Eigen::VectorXd twist = jacobian * joint_vel_eigen;

    geometry_msgs::TwistStamped velocity_msg;
    velocity_msg.header.stamp = stamp;
    velocity_msg.header.frame_id = robot_model_->getModelFrame();

    velocity_msg.twist.linear.x = twist[0];
    velocity_msg.twist.linear.y = twist[1];
    velocity_msg.twist.linear.z = twist[2];
    velocity_msg.twist.angular.x = twist[3];
    velocity_msg.twist.angular.y = twist[4];
    velocity_msg.twist.angular.z = twist[5];

    ee_velocity_pub_.publish(velocity_msg);
  }

  // Getters thread-safe
  std::vector<double> getJointAngles() {
    std::lock_guard<std::mutex> lock(joint_mutex_);
    return current_joint_angles_;
  }

  cv::Mat getRGBImage() {
    std::lock_guard<std::mutex> lock(image_mutex_);
    has_new_image_ = false;
    return current_rgb_image_.clone();
  }

  cv::Mat getDepthImage() {
    std::lock_guard<std::mutex> lock(depth_mutex_);
    has_new_depth_ = false;
    return current_depth_image_.clone();
  }

  bool hasNewImage() {
    std::lock_guard<std::mutex> lock(image_mutex_);
    return has_new_image_;
  }

  bool hasNewDepth() {
    std::lock_guard<std::mutex> lock(depth_mutex_);
    return has_new_depth_;
  }

private:
  moveit::core::RobotModelPtr robot_model_;
  moveit::core::RobotStatePtr robot_state_;
  const moveit::core::JointModelGroup *joint_model_group_;
  const moveit::core::LinkModel *ee_link_;

  ros::Subscriber joint_state_sub_;
  ros::Subscriber camera_rgb_sub_;
  ros::Subscriber camera_depth_sub_;
  ros::Publisher ee_pose_pub_;
  ros::Publisher ee_velocity_pub_;

  // Data storage
  std::mutex joint_mutex_;
  std::mutex image_mutex_;
  std::mutex depth_mutex_;

  std::vector<double> current_joint_angles_;
  std::vector<double> current_joint_velocities_;
  cv::Mat current_rgb_image_;
  cv::Mat current_depth_image_;

  ros::Time current_stamp_;
  ros::Time image_stamp_;
  ros::Time depth_stamp_;

  bool has_new_image_;
  bool has_new_depth_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "franka_state_camera_reader");

  ROS_INFO("==============================================");
  ROS_INFO("Franka State + Camera Reader");
  ROS_INFO("==============================================");

  FrankaStateCameraReader reader;

  ros::spin();

  return 0;
}