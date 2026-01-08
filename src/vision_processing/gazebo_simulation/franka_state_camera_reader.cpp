#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>

// MoveIt
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>

// Eigen conversions
#include <eigen_conversions/eigen_msg.h>

class FrankaUltimateSyncReader {
public:
  FrankaUltimateSyncReader() {
    ros::NodeHandle nh;

    // 1. Initialisation MoveIt (Cinématique)
    robot_model_loader::RobotModelLoader robot_model_loader(
        "robot_description");
    robot_model_ = robot_model_loader.getModel();
    robot_state_.reset(new moveit::core::RobotState(robot_model_));
    joint_model_group_ = robot_model_->getJointModelGroup("panda_arm");
    ee_link_ = robot_model_->getLinkModel(
        "panda_hand_tcp"); // Assurez-vous que ce lien existe dans votre URDF

    // 2. Configuration des Subscribers (Entrées)
    wrist_rgb_sub_.subscribe(nh, "/camera_wrist/color/image_raw", 1);
    wrist_depth_sub_.subscribe(
        nh, "/camera_wrist/aligned_depth_to_color/image_raw", 1);
    static_rgb_sub_.subscribe(nh, "/camera_static/color/image_raw", 1);
    static_depth_sub_.subscribe(
        nh, "/camera_static/aligned_depth_to_color/image_raw", 1);
    joint_sub_.subscribe(nh, "/joint_states", 1);

    // 3. Configuration des Publishers (Sorties Synchronisées)
    // On utilise un namespace "synced" pour clarifier que ces données sont
    // alignées
    pub_wrist_rgb_ =
        nh.advertise<sensor_msgs::Image>("/synced/camera_wrist/rgb", 1);
    pub_wrist_depth_ =
        nh.advertise<sensor_msgs::Image>("/synced/camera_wrist/depth", 1);
    pub_static_rgb_ =
        nh.advertise<sensor_msgs::Image>("/synced/camera_static/rgb", 1);
    pub_static_depth_ =
        nh.advertise<sensor_msgs::Image>("/synced/camera_static/depth", 1);
    pub_ee_pose_ =
        nh.advertise<geometry_msgs::PoseStamped>("/synced/ee_pose", 1);
    pub_joint_ =
        nh.advertise<sensor_msgs::JointState>("/synced/joint_states", 1);

    // 4. Synchroniseur
    // Queue size de 10 est suffisant pour absorber le jitter de Gazebo
    sync_.reset(new Sync(MySyncPolicy(10), wrist_rgb_sub_, wrist_depth_sub_,
                         static_rgb_sub_, static_depth_sub_, joint_sub_));

    sync_->registerCallback(boost::bind(&FrankaUltimateSyncReader::syncCallback,
                                        this, _1, _2, _3, _4, _5));

    ROS_INFO("Franka Synchronizer Ready: Publishing to /synced/* topics");
  }

  void syncCallback(const sensor_msgs::ImageConstPtr &w_rgb,
                    const sensor_msgs::ImageConstPtr &w_depth,
                    const sensor_msgs::ImageConstPtr &s_rgb,
                    const sensor_msgs::ImageConstPtr &s_depth,
                    const sensor_msgs::JointStateConstPtr &joints) {

    // --- ÉTAPE 0 : Définir le "Temps Maître" ---
    // On utilise le timestamp de la caméra poignet comme référence absolue pour
    // ce frame.
    ros::Time master_time = w_rgb->header.stamp;

    // --- ÉTAPE 1 : Calcul de la Pose EE (FK) ---
    std::vector<double> q;
    // Extraction propre des joints du panda (en ignorant les joints "finger" ou
    // autres)
    for (size_t i = 0; i < joints->name.size(); ++i) {
      if (joints->name[i].find("panda_joint") != std::string::npos)
        q.push_back(joints->position[i]);
    }

    // Vérification de sécurité
    if (q.size() < 7) {
      ROS_WARN_THROTTLE(1, "Joint state incomplet reçu.");
      return;
    }

    // Mise à jour du modèle cinématique
    robot_state_->setJointGroupPositions(joint_model_group_, q);
    robot_state_->update();
    const Eigen::Isometry3d &ee_transform =
        robot_state_->getGlobalLinkTransform(ee_link_);

    // Création du message de Pose
    geometry_msgs::PoseStamped ee_pose_msg;
    ee_pose_msg.header.stamp = master_time;
    ee_pose_msg.header.frame_id = "world"; // Ou "panda_link0" selon votre setup
    tf::poseEigenToMsg(ee_transform, ee_pose_msg.pose);

    // --- ÉTAPE 2 : Republication avec Timestamp Unique ---

    // Pour republier, on fait une copie superficielle (ne copie pas les pixels,
    // juste les headers) et on écrase le timestamp pour garantir l'alignement
    // parfait.

    sensor_msgs::Image out_w_rgb = *w_rgb;
    out_w_rgb.header.stamp = master_time;

    sensor_msgs::Image out_w_depth = *w_depth;
    out_w_depth.header.stamp = master_time;

    sensor_msgs::Image out_s_rgb = *s_rgb;
    out_s_rgb.header.stamp = master_time;

    sensor_msgs::Image out_s_depth = *s_depth;
    out_s_depth.header.stamp = master_time;

    sensor_msgs::JointState out_joints = *joints;
    out_joints.header.stamp = master_time;

    // Publication
    pub_wrist_rgb_.publish(out_w_rgb);
    pub_wrist_depth_.publish(out_w_depth);
    pub_static_rgb_.publish(out_s_rgb);
    pub_static_depth_.publish(out_s_depth);
    pub_ee_pose_.publish(ee_pose_msg);
    pub_joint_.publish(out_joints);

    // Debugging léger (1 fois par seconde max)
    ROS_INFO_THROTTLE(
        1, "Published synced frame at %f. EE Pos: [%.3f, %.3f, %.3f]",
        master_time.toSec(), ee_pose_msg.pose.position.x,
        ee_pose_msg.pose.position.y, ee_pose_msg.pose.position.z);
  }

private:
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image,
      sensor_msgs::Image, sensor_msgs::JointState>
      MySyncPolicy;

  typedef message_filters::Synchronizer<MySyncPolicy> Sync;

  // Subscribers
  message_filters::Subscriber<sensor_msgs::Image> wrist_rgb_sub_,
      wrist_depth_sub_;
  message_filters::Subscriber<sensor_msgs::Image> static_rgb_sub_,
      static_depth_sub_;
  message_filters::Subscriber<sensor_msgs::JointState> joint_sub_;

  // Publishers
  ros::Publisher pub_wrist_rgb_, pub_wrist_depth_;
  ros::Publisher pub_static_rgb_, pub_static_depth_;
  ros::Publisher pub_ee_pose_;
  ros::Publisher pub_joint_;

  boost::shared_ptr<Sync> sync_;

  // MoveIt
  moveit::core::RobotModelPtr robot_model_;
  moveit::core::RobotStatePtr robot_state_;
  const moveit::core::JointModelGroup *joint_model_group_;
  const moveit::core::LinkModel *ee_link_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "franka_ultimate_sync");
  FrankaUltimateSyncReader reader;
  ros::spin();
  return 0;
}