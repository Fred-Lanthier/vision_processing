// #include <cv_bridge/cv_bridge.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <message_filters/subscriber.h>
// #include <message_filters/sync_policies/approximate_time.h>
// #include <message_filters/synchronizer.h>
// #include <ros/ros.h>
// #include <sensor_msgs/Image.h>
// #include <sensor_msgs/JointState.h>

// // MoveIt
// #include <moveit/robot_model_loader/robot_model_loader.h>
// #include <moveit/robot_state/robot_state.h>

// // Eigen conversions
// #include <eigen_conversions/eigen_msg.h>

// class FrankaUltimateSyncReader {
// public:
//   FrankaUltimateSyncReader() {
//     ros::NodeHandle nh;

//     // 1. Initialisation MoveIt (Cinématique)
//     robot_model_loader::RobotModelLoader robot_model_loader(
//         "robot_description");
//     robot_model_ = robot_model_loader.getModel();
//     robot_state_.reset(new moveit::core::RobotState(robot_model_));
//     joint_model_group_ = robot_model_->getJointModelGroup("panda_arm");
//     ee_link_ = robot_model_->getLinkModel("panda_hand_tcp");

//     // 2. Configuration des Subscribers (Entrées)
//     wrist_rgb_sub_.subscribe(nh, "/camera_wrist/color/image_raw", 1);
//     wrist_depth_sub_.subscribe(
//         nh, "/camera_wrist/aligned_depth_to_color/image_raw", 1);

//     // --- CAMÉRA STATIQUE DÉSACTIVÉE ---
//     // static_rgb_sub_.subscribe(nh, "/camera_static/color/image_raw", 1);
//     // static_depth_sub_.subscribe(nh,
//     // "/camera_static/aligned_depth_to_color/image_raw", 1);

//     joint_sub_.subscribe(nh, "/joint_states", 1);

//     // 3. Configuration des Publishers (Sorties Synchronisées)
//     pub_wrist_rgb_ =
//         nh.advertise<sensor_msgs::Image>("/synced/camera_wrist/rgb", 1);
//     pub_wrist_depth_ =
//         nh.advertise<sensor_msgs::Image>("/synced/camera_wrist/depth", 1);

//     // --- CAMÉRA STATIQUE DÉSACTIVÉE ---
//     // pub_static_rgb_ =
//     // nh.advertise<sensor_msgs::Image>("/synced/camera_static/rgb", 1);
//     // pub_static_depth_ =
//     // nh.advertise<sensor_msgs::Image>("/synced/camera_static/depth", 1);

//     pub_ee_pose_ =
//         nh.advertise<geometry_msgs::PoseStamped>("/synced/ee_pose", 1);
//     pub_joint_ =
//         nh.advertise<sensor_msgs::JointState>("/synced/joint_states", 1);

//     // 4. Synchroniseur
//     // Queue size de 10 est suffisant pour absorber le jitter de Gazebo

//     // MODIFICATION CRITIQUE : On passe de 5 entrées à 3 entrées dans le
//     // constructeur Sync
//     sync_.reset(new Sync(MySyncPolicy(10), wrist_rgb_sub_, wrist_depth_sub_,
//                          // static_rgb_sub_,  <-- Retiré
//                          // static_depth_sub_, <-- Retiré
//                          joint_sub_));

//     // MODIFICATION CRITIQUE : On change le bind pour n'avoir que 3 arguments
//     // _1 = wrist_rgb, _2 = wrist_depth, _3 = joints
//     sync_->registerCallback(
//         boost::bind(&FrankaUltimateSyncReader::syncCallback, this, _1, _2,
//         _3));

//     ROS_INFO("Franka Synchronizer Ready: Publishing to /synced/* topics
//     (WRIST "
//              "ONLY)");
//   }

//   // MODIFICATION CRITIQUE : Signature du callback allégée
//   void syncCallback(const sensor_msgs::ImageConstPtr &w_rgb,
//                     const sensor_msgs::ImageConstPtr &w_depth,
//                     // const sensor_msgs::ImageConstPtr &s_rgb, <-- Retiré
//                     // const sensor_msgs::ImageConstPtr &s_depth, <-- Retiré
//                     const sensor_msgs::JointStateConstPtr &joints) {

//     // --- ÉTAPE 0 : Définir le "Temps Maître" ---
//     ros::Time master_time = w_rgb->header.stamp;

//     // --- ÉTAPE 1 : Calcul de la Pose EE (FK) ---
//     std::vector<double> q;
//     for (size_t i = 0; i < joints->name.size(); ++i) {
//       if (joints->name[i].find("panda_joint") != std::string::npos)
//         q.push_back(joints->position[i]);
//     }

//     if (q.size() < 7) {
//       ROS_WARN_THROTTLE(1, "Joint state incomplet reçu.");
//       return;
//     }

//     robot_state_->setJointGroupPositions(joint_model_group_, q);
//     robot_state_->update();
//     const Eigen::Isometry3d &ee_transform =
//         robot_state_->getGlobalLinkTransform(ee_link_);

//     geometry_msgs::PoseStamped ee_pose_msg;
//     ee_pose_msg.header.stamp = master_time;
//     ee_pose_msg.header.frame_id = "world";
//     tf::poseEigenToMsg(ee_transform, ee_pose_msg.pose);

//     // --- ÉTAPE 2 : Republication avec Timestamp Unique ---

//     sensor_msgs::Image out_w_rgb = *w_rgb;
//     out_w_rgb.header.stamp = master_time;

//     sensor_msgs::Image out_w_depth = *w_depth;
//     out_w_depth.header.stamp = master_time;

//     // --- CAMÉRA STATIQUE DÉSACTIVÉE ---
//     // sensor_msgs::Image out_s_rgb = *s_rgb;
//     // out_s_rgb.header.stamp = master_time;
//     // sensor_msgs::Image out_s_depth = *s_depth;
//     // out_s_depth.header.stamp = master_time;

//     sensor_msgs::JointState out_joints = *joints;
//     out_joints.header.stamp = master_time;

//     // Publication
//     pub_wrist_rgb_.publish(out_w_rgb);
//     pub_wrist_depth_.publish(out_w_depth);

//     // pub_static_rgb_.publish(out_s_rgb);
//     // pub_static_depth_.publish(out_s_depth);

//     pub_ee_pose_.publish(ee_pose_msg);
//     pub_joint_.publish(out_joints);

//     // Debugging léger (1 fois par seconde max)
//     // ROS_INFO_THROTTLE(
//     //     1, "Published synced frame at %f. EE Pos: [%.3f, %.3f, %.3f]",
//     //     master_time.toSec(), ee_pose_msg.pose.position.x,
//     //     ee_pose_msg.pose.position.y, ee_pose_msg.pose.position.z);
//   }

// private:
//   // MODIFICATION CRITIQUE : La Policy ne doit contenir que les types qu'on
//   // utilise vraiment ! On retire 2 sensor_msgs::Image de la liste des
//   templates typedef message_filters::sync_policies::ApproximateTime<
//       sensor_msgs::Image, // Wrist RGB
//       sensor_msgs::Image, // Wrist Depth
//       // sensor_msgs::Image,    // <-- REMOVED (Static RGB)
//       // sensor_msgs::Image,    // <-- REMOVED (Static Depth)
//       sensor_msgs::JointState // Joints
//       >
//       MySyncPolicy;

//   typedef message_filters::Synchronizer<MySyncPolicy> Sync;

//   // Subscribers
//   message_filters::Subscriber<sensor_msgs::Image> wrist_rgb_sub_,
//       wrist_depth_sub_;
//   // message_filters::Subscriber<sensor_msgs::Image> static_rgb_sub_,
//   // static_depth_sub_; <-- Retiré
//   message_filters::Subscriber<sensor_msgs::JointState> joint_sub_;

//   // Publishers
//   ros::Publisher pub_wrist_rgb_, pub_wrist_depth_;
//   // ros::Publisher pub_static_rgb_, pub_static_depth_; <-- Retiré
//   ros::Publisher pub_ee_pose_;
//   ros::Publisher pub_joint_;

//   boost::shared_ptr<Sync> sync_;

//   // MoveIt
//   moveit::core::RobotModelPtr robot_model_;
//   moveit::core::RobotStatePtr robot_state_;
//   const moveit::core::JointModelGroup *joint_model_group_;
//   const moveit::core::LinkModel *ee_link_;
// };

// int main(int argc, char **argv) {
//   ros::init(argc, argv, "franka_ultimate_sync");
//   FrankaUltimateSyncReader reader;
//   ros::spin();
//   return 0;
// }

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>

// PLUS DE MOVEIT ICI ! (Source de blocage supprimée)

class FrankaPureSync {
public:
  FrankaPureSync() {
    ros::NodeHandle nh;

    // 1. Subscribers (Seulement Poignet + Joints)
    wrist_rgb_sub_.subscribe(nh, "/camera_wrist/color/image_raw", 1);
    wrist_depth_sub_.subscribe(
        nh, "/camera_wrist/aligned_depth_to_color/image_raw", 1);
    joint_sub_.subscribe(nh, "/joint_states", 1);

    // 2. Publishers
    pub_wrist_rgb_ =
        nh.advertise<sensor_msgs::Image>("/synced/camera_wrist/rgb", 1);
    pub_wrist_depth_ =
        nh.advertise<sensor_msgs::Image>("/synced/camera_wrist/depth", 1);
    pub_joint_ =
        nh.advertise<sensor_msgs::JointState>("/synced/joint_states", 1);

    // 3. Synchro (3 entrées seulement)
    sync_.reset(new Sync(MySyncPolicy(10), wrist_rgb_sub_, wrist_depth_sub_,
                         joint_sub_));

    sync_->registerCallback(
        boost::bind(&FrankaPureSync::syncCallback, this, _1, _2, _3));

    ROS_INFO("🚀 Franka Pure Sync Ready (No FK, No MoveIt overhead)");
  }

  void syncCallback(const sensor_msgs::ImageConstPtr &w_rgb,
                    const sensor_msgs::ImageConstPtr &w_depth,
                    const sensor_msgs::JointStateConstPtr &joints) {

    ros::Time master_time = w_rgb->header.stamp;

    // On republie simplement avec le même timestamp
    sensor_msgs::Image out_w_rgb = *w_rgb;
    out_w_rgb.header.stamp = master_time;

    sensor_msgs::Image out_w_depth = *w_depth;
    out_w_depth.header.stamp = master_time;

    sensor_msgs::JointState out_joints = *joints;
    out_joints.header.stamp = master_time;

    pub_wrist_rgb_.publish(out_w_rgb);
    pub_wrist_depth_.publish(out_w_depth);
    pub_joint_.publish(out_joints);
  }

private:
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::JointState>
      MySyncPolicy;

  typedef message_filters::Synchronizer<MySyncPolicy> Sync;

  message_filters::Subscriber<sensor_msgs::Image> wrist_rgb_sub_,
      wrist_depth_sub_;
  message_filters::Subscriber<sensor_msgs::JointState> joint_sub_;

  ros::Publisher pub_wrist_rgb_, pub_wrist_depth_;
  ros::Publisher pub_joint_;

  boost::shared_ptr<Sync> sync_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "franka_state_camera_reader");
  FrankaPureSync syncer;
  ros::spin();
  return 0;
}

// #include <geometry_msgs/PoseStamped.h>
// #include <message_filters/subscriber.h>
// #include <message_filters/sync_policies/approximate_time.h>
// #include <message_filters/synchronizer.h>
// #include <ros/ros.h>
// #include <sensor_msgs/Image.h>
// #include <sensor_msgs/JointState.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.h>
// #include <tf2_ros/transform_listener.h>

// class FrankaPureSync {
// public:
//   FrankaPureSync() : tf_listener_(tf_buffer_) {
//     ros::NodeHandle nh;

//     // 1. Subscribers
//     // Queue size = 1 pour les images (on veut la dernière)
//     // Queue size = 100 pour les joints (pour absorber le flood 1000Hz sans
//     // warning)
//     wrist_rgb_sub_.subscribe(nh, "/camera_wrist/color/image_raw", 1);
//     wrist_depth_sub_.subscribe(
//         nh, "/camera_wrist/aligned_depth_to_color/image_raw", 1);
//     joint_sub_.subscribe(nh, "/joint_states", 100);

//     // 2. Publishers
//     pub_wrist_rgb_ =
//         nh.advertise<sensor_msgs::Image>("/synced/camera_wrist/rgb", 1);
//     pub_wrist_depth_ =
//         nh.advertise<sensor_msgs::Image>("/synced/camera_wrist/depth", 1);
//     pub_joint_ =
//         nh.advertise<sensor_msgs::JointState>("/synced/joint_states", 1);
//     pub_ee_pose_ =
//         nh.advertise<geometry_msgs::PoseStamped>("/synced/ee_pose", 1);

//     // 3. Synchro
//     // Intervall 6Hz = ~160ms. Slop à 100ms (0.1) est suffisant.
//     sync_.reset(new Sync(MySyncPolicy(10), wrist_rgb_sub_, wrist_depth_sub_,
//                          joint_sub_));
//     sync_->registerCallback(
//         boost::bind(&FrankaPureSync::syncCallback, this, _1, _2, _3));

//     ROS_INFO("🚀 Franka 6Hz Sync Node (Async + Non-Blocking) Ready");
//   }

//   void syncCallback(const sensor_msgs::ImageConstPtr &w_rgb,
//                     const sensor_msgs::ImageConstPtr &w_depth,
//                     const sensor_msgs::JointStateConstPtr &joints) {

//     // Pas de throttling manuel ici car la caméra force déjà le rythme à 6Hz.

//     geometry_msgs::PoseStamped ee_pose_msg;
//     ee_pose_msg.header.stamp = w_rgb->header.stamp;
//     ee_pose_msg.header.frame_id = "world";

//     bool tf_success = false;
//     geometry_msgs::TransformStamped transformStamped;

//     try {
//       // Time(0) est crucial ici pour ne pas bloquer le thread qui doit aussi
//       // vider la queue des joints
//       transformStamped =
//           tf_buffer_.lookupTransform("world", "panda_hand_tcp",
//           ros::Time(0));
//       tf_success = true;
//     } catch (tf2::TransformException &ex) {
//       ROS_WARN_THROTTLE(5.0, "TF Fail: %s", ex.what());
//     }

//     if (tf_success) {
//       ee_pose_msg.pose.position.x = transformStamped.transform.translation.x;
//       ee_pose_msg.pose.position.y = transformStamped.transform.translation.y;
//       ee_pose_msg.pose.position.z = transformStamped.transform.translation.z;
//       ee_pose_msg.pose.orientation = transformStamped.transform.rotation;

//       pub_wrist_rgb_.publish(w_rgb);
//       pub_wrist_depth_.publish(w_depth);
//       pub_joint_.publish(joints);
//       pub_ee_pose_.publish(ee_pose_msg);
//     }
//   }

// private:
//   typedef message_filters::sync_policies::ApproximateTime<
//       sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::JointState>
//       MySyncPolicy;
//   typedef message_filters::Synchronizer<MySyncPolicy> Sync;

//   message_filters::Subscriber<sensor_msgs::Image> wrist_rgb_sub_,
//       wrist_depth_sub_;
//   message_filters::Subscriber<sensor_msgs::JointState> joint_sub_;

//   ros::Publisher pub_wrist_rgb_, pub_wrist_depth_, pub_joint_, pub_ee_pose_;
//   boost::shared_ptr<Sync> sync_;

//   tf2_ros::Buffer tf_buffer_;
//   tf2_ros::TransformListener tf_listener_;
// };

// int main(int argc, char **argv) {
//   ros::init(argc, argv, "franka_state_camera_reader");

//   FrankaPureSync syncer;

//   // IMPORTANT : Utiliser 4 threads pour être large (1 pour images, 1 pour
//   // joints, 1 pour callbacks, 1 idle)
//   ros::AsyncSpinner spinner(4);
//   spinner.start();

//   ros::waitForShutdown();
//   return 0;
// }