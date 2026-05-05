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