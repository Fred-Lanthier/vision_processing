#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.hpp>
#include <realtime_tools/realtime_buffer.h>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>

namespace vision_processing {

// Velocity-commanded position servo for a group of joints.  The CBF publishes
// dq_safe directly to "command" and this controller integrates it at the
// hardware update rate.  Claiming PositionJointInterface is intentional: a
// zero command then holds the current reference against gravity, whereas the
// Gazebo VelocityJointInterface only regulates dq and lets the arm sag.
class JointGroupVelocityController
    : public controller_interface::Controller<
          hardware_interface::PositionJointInterface> {
 public:
  bool init(hardware_interface::PositionJointInterface* hardware,
            ros::NodeHandle& controller_nh) override {
    if (!controller_nh.getParam("joints", joint_names_) ||
        joint_names_.empty()) {
      ROS_ERROR("JointGroupVelocityController requires a non-empty 'joints' list");
      return false;
    }

    controller_nh.param("command_timeout", command_timeout_, 0.1);
    controller_nh.param("max_abs_velocity", max_abs_velocity_, 0.0);
    controller_nh.param("max_position_lead", max_position_lead_, 0.05);
    if (command_timeout_ <= 0.0) {
      ROS_ERROR("JointGroupVelocityController command_timeout must be positive");
      return false;
    }

    joint_handles_.reserve(joint_names_.size());
    try {
      for (const auto& joint_name : joint_names_) {
        joint_handles_.push_back(hardware->getHandle(joint_name));
      }
    } catch (const hardware_interface::HardwareInterfaceException& exception) {
      ROS_ERROR_STREAM("Failed to acquire position joint handle: "
                       << exception.what());
      return false;
    }

    Command initial;
    initial.velocity.assign(joint_names_.size(), 0.0);
    command_buffer_.writeFromNonRT(initial);
    position_reference_.assign(joint_names_.size(), 0.0);
    command_subscriber_ = controller_nh.subscribe(
        "command", 1, &JointGroupVelocityController::commandCallback, this);
    return true;
  }

  void starting(const ros::Time& time) override {
    Command command;
    command.velocity.assign(joint_names_.size(), 0.0);
    command.stamp = time;
    command_buffer_.initRT(command);
    for (std::size_t index = 0; index < joint_handles_.size(); ++index) {
      position_reference_[index] = joint_handles_[index].getPosition();
      joint_handles_[index].setCommand(position_reference_[index]);
    }
  }

  void update(const ros::Time& time, const ros::Duration& period) override {
    const Command* command = command_buffer_.readFromRT();
    const bool valid_size = command->velocity.size() == joint_handles_.size();
    const double age = (time - command->stamp).toSec();
    const bool fresh = command->stamp != ros::Time(0) && age >= 0.0 &&
                       age <= command_timeout_;

    for (std::size_t index = 0; index < joint_handles_.size(); ++index) {
      if (valid_size && fresh) {
        position_reference_[index] +=
            command->velocity[index] * period.toSec();

        // Do not let an unachieved reference accumulate far ahead of the
        // measured robot.  This prevents a delayed catch-up motion after the
        // CBF has held the arm back.
        if (max_position_lead_ > 0.0) {
          const double current_position = joint_handles_[index].getPosition();
          position_reference_[index] = std::max(
              current_position - max_position_lead_,
              std::min(current_position + max_position_lead_,
                       position_reference_[index]));
        }
      }
      // A stale command holds the last position reference instead of applying
      // zero effort/velocity, so initialization and shutdown remain gravity
      // supported.
      joint_handles_[index].setCommand(position_reference_[index]);
    }
  }

  void stopping(const ros::Time&) override {
    for (std::size_t index = 0; index < joint_handles_.size(); ++index) {
      position_reference_[index] = joint_handles_[index].getPosition();
      joint_handles_[index].setCommand(position_reference_[index]);
    }
  }

 private:
  struct Command {
    std::vector<double> velocity;
    ros::Time stamp;
  };

  void commandCallback(const std_msgs::Float64MultiArrayConstPtr& message) {
    if (message->data.size() != joint_handles_.size()) {
      ROS_ERROR_THROTTLE(
          1.0, "Velocity command has %zu values; expected %zu",
          message->data.size(), joint_handles_.size());
      return;
    }

    Command command;
    command.velocity.resize(joint_handles_.size());
    for (std::size_t index = 0; index < joint_handles_.size(); ++index) {
      double velocity = message->data[index];
      if (!std::isfinite(velocity)) {
        ROS_ERROR_THROTTLE(1.0, "Rejected non-finite velocity command");
        return;
      }
      if (max_abs_velocity_ > 0.0) {
        velocity = std::max(-max_abs_velocity_,
                            std::min(max_abs_velocity_, velocity));
      }
      command.velocity[index] = velocity;
    }
    command.stamp = ros::Time::now();
    command_buffer_.writeFromNonRT(command);
  }

  std::vector<std::string> joint_names_;
  std::vector<hardware_interface::JointHandle> joint_handles_;
  std::vector<double> position_reference_;
  realtime_tools::RealtimeBuffer<Command> command_buffer_;
  ros::Subscriber command_subscriber_;
  double command_timeout_{0.1};
  double max_abs_velocity_{0.0};
  double max_position_lead_{0.05};
};

}  // namespace vision_processing

PLUGINLIB_EXPORT_CLASS(
    vision_processing::JointGroupVelocityController,
    controller_interface::ControllerBase)
