#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <XmlRpcValue.h>
#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.hpp>
#include <realtime_tools/realtime_buffer.h>
#include <realtime_tools/realtime_publisher.h>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64.h>

#include <vision_processing/CertifiedCbfCommand.h>
#include <vision_processing/IssfEpsilonSample.h>
#include <vision_processing/certified_cbf_math.h>

namespace vision_processing {

// Hard-real-time consumer for the asynchronous GPU CBF.  The GPU worker may
// be late; update() never waits for CUDA or for a ROS callback.  It checks the
// latest command/certificate against the directly measured joint positions,
// applies the certified scalar brake, and integrates the resulting joint
// velocity into a gravity-holding position reference in the same manager tick.
class CertifiedCbfVelocityController
    : public controller_interface::Controller<
          hardware_interface::PositionJointInterface> {
 public:
  bool init(hardware_interface::PositionJointInterface* hardware,
            ros::NodeHandle& controller_nh) override {
    if (!controller_nh.getParam("joints", joint_names_) ||
        joint_names_.size() != certified_cbf::kJoints) {
      ROS_ERROR("CertifiedCbfVelocityController requires exactly seven joints");
      return false;
    }

    controller_nh.param("command_timeout", command_timeout_, 0.10);
    controller_nh.param("max_abs_velocity", max_abs_velocity_, 0.70);
    controller_nh.param("max_position_lead", max_position_lead_, 0.05);
    controller_nh.param("release_tau", release_tau_, 0.05);
    controller_nh.param("nominal_period", nominal_period_, 0.001);
    controller_nh.param("tick_decay", tick_decay_, 0.98);
    controller_nh.param("status_divisor", status_divisor_, 20);
    controller_nh.param("timing_divisor", timing_divisor_, 10);
    controller_nh.param(
        "epsilon_sample_divisor", epsilon_sample_divisor_, 10);
    controller_nh.param(
        "epsilon_normal_alpha_threshold",
        epsilon_normal_alpha_threshold_, 0.90);
    controller_nh.param(
        "epsilon_motion_threshold", epsilon_motion_threshold_, 1.0e-4);
    controller_nh.param("issf_veto_hold", issf_veto_hold_, 0.05);
    controller_nh.param("h_stop", parameters_.h_stop, 0.0);
    controller_nh.param("h_activate", parameters_.h_activate, 0.04);
    controller_nh.param("exp_kappa", parameters_.exp_kappa, 25.0);
    controller_nh.param("recovery_speed", parameters_.recovery_speed, 0.05);
    controller_nh.param("recovery_depth", parameters_.recovery_depth, 0.015);
    controller_nh.param("recovery_slack", parameters_.recovery_slack, 0.5);
    controller_nh.param("issf_epsilon", parameters_.issf_epsilon, 0.0);
    controller_nh.param("issf_rho", parameters_.issf_rho, 0.0);
    if (command_timeout_ <= 0.0 || nominal_period_ <= 0.0 ||
        max_position_lead_ < 0.0) {
      ROS_ERROR(
          "Certified CBF timing parameters must be positive and "
          "max_position_lead must be non-negative");
      return false;
    }
    tick_decay_ = std::max(0.0, std::min(1.0, tick_decay_));
    status_divisor_ = std::max(status_divisor_, 1);
    timing_divisor_ = std::max(timing_divisor_, 1);
    epsilon_sample_divisor_ = std::max(epsilon_sample_divisor_, 1);
    epsilon_normal_alpha_threshold_ = std::max(
        0.0, std::min(1.0, epsilon_normal_alpha_threshold_));
    epsilon_motion_threshold_ = std::max(epsilon_motion_threshold_, 0.0);
    if (!loadLipschitz(controller_nh)) {
      return false;
    }

    joint_handles_.reserve(joint_names_.size());
    try {
      for (const auto& joint_name : joint_names_) {
        joint_handles_.push_back(hardware->getHandle(joint_name));
      }
    } catch (const hardware_interface::HardwareInterfaceException& exception) {
      ROS_ERROR_STREAM("Failed to acquire certified CBF position joint handle: "
                       << exception.what());
      return false;
    }

    std::string command_topic = "/cbf_safety/certified_command";
    controller_nh.param("certified_command_topic", command_topic, command_topic);
    command_subscriber_ = controller_nh.subscribe(
        command_topic, 1,
        &CertifiedCbfVelocityController::commandCallback, this);
    if (issf_veto_hold_ > 0.0) {
      issf_subscriber_ = controller_nh.subscribe(
          "/cbf_safety/issf_monitor", 1,
          &CertifiedCbfVelocityController::issfCallback, this);
    }

    Packet initial;
    command_buffer_.writeFromNonRT(initial);
    status_publisher_.reset(
        new realtime_tools::RealtimePublisher<std_msgs::Float32MultiArray>(
            controller_nh, "/cbf_reflex/status", 1));
    status_publisher_->msg_.data.resize(6, 0.0f);
    compute_time_publisher_.reset(
        new realtime_tools::RealtimePublisher<std_msgs::Float64>(
            controller_nh, "/pipeline/timing/cbf_reflex_compute", 1));
    period_publisher_.reset(
        new realtime_tools::RealtimePublisher<std_msgs::Float64>(
            controller_nh, "/pipeline/timing/cbf_reflex_period", 1));
    epsilon_sample_publisher_.reset(
        new realtime_tools::RealtimePublisher<
            vision_processing::IssfEpsilonSample>(
            controller_nh, "/cbf_reflex/epsilon_sample", 1));

    ROS_INFO_STREAM(
        "Certified CBF controller ready: " << command_topic
        << " -> hardware at controller-manager rate; L="
        << lipschitzDescription()
        << ", h_activate=" << parameters_.h_activate
        << ", exp_kappa=" << parameters_.exp_kappa
        << ", ISSf=" << parameters_.issf_epsilon << "+"
        << parameters_.issf_rho << "*v"
        << ", epsilon calibration=" << epsilon_sample_divisor_
        << " ticks/window, velocity source=ros_control joint handle"
        << ", position lead=" << max_position_lead_ << " rad");
    return true;
  }

  void starting(const ros::Time& time) override {
    Packet initial;
    initial.receipt_stamp = time;
    command_buffer_.initRT(initial);
    for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
      position_reference_[joint] = joint_handles_[joint].getPosition();
      joint_handles_[joint].setCommand(position_reference_[joint]);
    }
    alpha_ = 1.0;
    horizon_ = nominal_period_;
    tick_ = 0;
    epsilon_reference_valid_ = false;
    epsilon_calibration_started_ = false;
    epsilon_normal_peak_ = EpsilonPeak{};
    epsilon_braking_peak_ = EpsilonPeak{};
  }

  void update(const ros::Time& time, const ros::Duration& period) override {
    const bool timing_due =
        (tick_ % static_cast<std::size_t>(timing_divisor_)) == 0;
    const ros::WallTime wall_start = timing_due
        ? ros::WallTime::now() : ros::WallTime();
    const double dt = std::max(period.toSec(), nominal_period_);
    horizon_ = std::max(dt, tick_decay_ * horizon_);
    const Packet* packet = command_buffer_.readFromRT();
    const double receipt_age = (time - packet->receipt_stamp).toSec();
    const bool fresh = packet->valid && receipt_age >= 0.0 &&
                       receipt_age <= command_timeout_;

    certified_cbf::Result certificate;
    double raw_alpha = 0.0;
    double solve_age = std::numeric_limits<double>::infinity();
    std::array<double, certified_cbf::kJoints> measured_q{};
    std::array<double, certified_cbf::kJoints> measured_velocity{};
    for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
      measured_q[joint] = joint_handles_[joint].getPosition();
      measured_velocity[joint] = joint_handles_[joint].getVelocity();
    }
    observeEpsilonSample(measured_velocity);

    if (fresh) {
      solve_age = std::max((time - packet->solve_stamp).toSec(), 0.0);
      std::array<double, certified_cbf::kMaxRows> effective_lipschitz{};
      fillEffectiveLipschitz(packet->row_count, effective_lipschitz);
      certificate = certified_cbf::computeScale(
          measured_q, packet->q0, packet->velocity,
          packet->h, packet->environment_hdot, packet->gradient,
          effective_lipschitz, packet->row_count, solve_age, horizon_,
          parameters_);
      raw_alpha = certificate.alpha;

      const double veto_until = issf_veto_until_.load(
          std::memory_order_relaxed);
      if (time.toSec() < veto_until && !isRecovery(*packet)) {
        raw_alpha = 0.0;
      }
    }

    // Brake immediately; release through the same first-order filter as the
    // Python reflex so a fresh packet cannot create a velocity step.
    if (raw_alpha < alpha_) {
      alpha_ = raw_alpha;
    } else {
      const double gamma = dt / (dt + std::max(release_tau_, 1.0e-6));
      alpha_ += gamma * (raw_alpha - alpha_);
    }
    alpha_ = std::max(0.0, std::min(1.0, alpha_));

    std::array<double, certified_cbf::kJoints> applied_velocity_command{};
    for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
      double applied_velocity = 0.0;
      if (fresh) {
        applied_velocity = alpha_ * packet->velocity[joint];
        if (max_abs_velocity_ > 0.0) {
          applied_velocity = std::max(
              -max_abs_velocity_,
              std::min(max_abs_velocity_, applied_velocity));
        }
        position_reference_[joint] += applied_velocity * period.toSec();

        // Prevent an unachieved position reference from accumulating far in
        // front of the measured robot.  A stale packet then holds at most this
        // bounded lead instead of continuing to integrate velocity.
        if (max_position_lead_ > 0.0) {
          position_reference_[joint] = std::max(
              measured_q[joint] - max_position_lead_,
              std::min(measured_q[joint] + max_position_lead_,
                       position_reference_[joint]));
        }
      }
      // PositionJointInterface intentionally supplies the gravity-holding
      // inner servo that made runPP1030 track the certified recovery command.
      // With a stale packet, retaining this reference safely holds the arm.
      applied_velocity_command[joint] = applied_velocity;
      joint_handles_[joint].setCommand(position_reference_[joint]);
    }
    storeEpsilonReference(
        *packet, measured_q, applied_velocity_command, alpha_, fresh);

    if ((tick_ % static_cast<std::size_t>(status_divisor_)) == 0 &&
        status_publisher_ && status_publisher_->trylock()) {
      auto& data = status_publisher_->msg_.data;
      data[0] = static_cast<float>(alpha_);
      data[1] = static_cast<float>(certificate.minimum_margin);
      data[2] = static_cast<float>(certificate.braking_rows);
      data[3] = static_cast<float>(solve_age);
      data[4] = static_cast<float>(packet->empirical_lipschitz);
      data[5] = static_cast<float>(horizon_);
      status_publisher_->unlockAndPublish();
    }
    if (timing_due) {
      if (compute_time_publisher_ && compute_time_publisher_->trylock()) {
        compute_time_publisher_->msg_.data =
            (ros::WallTime::now() - wall_start).toSec() * 1000.0;
        compute_time_publisher_->unlockAndPublish();
      }
      if (period_publisher_ && period_publisher_->trylock()) {
        period_publisher_->msg_.data = period.toSec() * 1000.0;
        period_publisher_->unlockAndPublish();
      }
    }
    publishEpsilonWindow();
    ++tick_;
  }

  void stopping(const ros::Time&) override {
    for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
      position_reference_[joint] = joint_handles_[joint].getPosition();
      joint_handles_[joint].setCommand(position_reference_[joint]);
    }
  }

 private:
  struct EpsilonPeak {
    bool valid{false};
    double adverse_projection{0.0};
  };

  struct Packet {
    bool valid{false};
    ros::Time solve_stamp{};
    ros::Time receipt_stamp{};
    std::uint32_t sequence{0};
    std::uint32_t selection_generation{0};
    std::size_t row_count{0};
    std::array<double, certified_cbf::kJoints> q0{};
    std::array<double, certified_cbf::kJoints> velocity{};
    std::array<double, certified_cbf::kMaxRows> h{};
    std::array<double, certified_cbf::kMaxRows> environment_hdot{};
    std::array<double,
               certified_cbf::kMaxRows * certified_cbf::kJoints> gradient{};
    double empirical_lipschitz{0.0};
  };

  // Identify the matched disturbance in q_dot = u + d without reconstructing
  // asynchronous ROS topics.  The joint handle reports the velocity at the
  // beginning of tick k, before this update writes u_k, so it is compared to
  // the command and gradients saved at tick k-1.  Do not low-pass only q_dot:
  // that would manufacture LPF(u)-u disturbance at every command transition.
  void observeEpsilonSample(
      const std::array<double, certified_cbf::kJoints>& measured_velocity) {
    if (!epsilon_reference_valid_) {
      return;
    }

    std::array<double, certified_cbf::kJoints> disturbance{};
    for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
      if (!std::isfinite(measured_velocity[joint])) {
        epsilon_reference_valid_ = false;
        return;
      }
      disturbance[joint] = measured_velocity[joint]
                         - epsilon_previous_applied_velocity_[joint];
    }

    EpsilonPeak& peak =
        epsilon_previous_alpha_ > epsilon_normal_alpha_threshold_
            ? epsilon_normal_peak_ : epsilon_braking_peak_;
    for (std::size_t row = 0; row < epsilon_reference_row_count_; ++row) {
      if (!epsilon_reference_active_[row]) {
        continue;
      }
      double gradient_squared = 0.0;
      double gradient_dot_disturbance = 0.0;
      for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
        const double gradient = epsilon_reference_gradient_[
            row * certified_cbf::kJoints + joint];
        gradient_squared += gradient * gradient;
        gradient_dot_disturbance += gradient * disturbance[joint];
      }
      if (gradient_squared <= 1.0e-12) {
        continue;
      }
      const double adverse_projection = std::max(
          0.0, -gradient_dot_disturbance / std::sqrt(gradient_squared));
      if (!peak.valid || adverse_projection > peak.adverse_projection) {
        peak.valid = true;
        peak.adverse_projection = adverse_projection;
      }
    }
  }

  void storeEpsilonReference(
      const Packet& packet,
      const std::array<double, certified_cbf::kJoints>& measured_q,
      const std::array<double, certified_cbf::kJoints>& applied_velocity,
      double alpha,
      bool fresh) {
    epsilon_previous_applied_velocity_ = applied_velocity;
    epsilon_previous_alpha_ = alpha;
    if (!epsilon_calibration_started_) {
      double certified_speed_squared = 0.0;
      if (fresh) {
        for (const double velocity : packet.velocity) {
          certified_speed_squared += velocity * velocity;
        }
      }
      if (std::sqrt(certified_speed_squared) <= epsilon_motion_threshold_) {
        epsilon_reference_valid_ = false;
        epsilon_reference_row_count_ = 0;
        return;
      }
      epsilon_calibration_started_ = true;
    }
    epsilon_reference_valid_ = fresh && packet.row_count > 0;
    epsilon_reference_row_count_ =
        epsilon_reference_valid_ ? packet.row_count : 0;
    if (!epsilon_reference_valid_) {
      return;
    }

    bool any_active = false;
    for (std::size_t row = 0; row < packet.row_count; ++row) {
      double extrapolated_h = packet.h[row];
      for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
        const std::size_t index = row * certified_cbf::kJoints + joint;
        const double gradient = packet.gradient[index];
        epsilon_reference_gradient_[index] = gradient;
        extrapolated_h += gradient * (measured_q[joint] - packet.q0[joint]);
      }
      epsilon_reference_active_[row] =
          extrapolated_h < parameters_.h_activate;
      any_active = any_active || epsilon_reference_active_[row];
    }
    epsilon_reference_valid_ = any_active;
  }

  void resetEpsilonWindow() {
    epsilon_normal_peak_ = EpsilonPeak{};
    epsilon_braking_peak_ = EpsilonPeak{};
  }

  void publishEpsilonWindow() {
    if (((tick_ + 1) % static_cast<std::size_t>(
             epsilon_sample_divisor_)) != 0) {
      return;
    }
    if (!epsilon_calibration_started_) {
      resetEpsilonWindow();
      return;
    }
    if (!epsilon_sample_publisher_ ||
        !epsilon_sample_publisher_->trylock()) {
      return;
    }

    auto& message = epsilon_sample_publisher_->msg_;
    message.has_normal = epsilon_normal_peak_.valid;
    message.normal_adverse_projection =
        epsilon_normal_peak_.adverse_projection;
    message.has_braking = epsilon_braking_peak_.valid;
    message.braking_adverse_projection =
        epsilon_braking_peak_.adverse_projection;
    epsilon_sample_publisher_->unlockAndPublish();
    resetEpsilonWindow();
  }

  bool loadLipschitz(ros::NodeHandle& controller_nh) {
    XmlRpc::XmlRpcValue value;
    if (!controller_nh.getParam("grad_lipschitz", value)) {
      lipschitz_values_[0] = 10.0;
      lipschitz_count_ = 1;
      return true;
    }
    std::vector<double> parsed;
    if (value.getType() == XmlRpc::XmlRpcValue::TypeInt) {
      parsed.push_back(static_cast<int>(value));
    } else if (value.getType() == XmlRpc::XmlRpcValue::TypeDouble) {
      parsed.push_back(static_cast<double>(value));
    } else if (value.getType() == XmlRpc::XmlRpcValue::TypeString) {
      std::string text = static_cast<std::string>(value);
      std::replace(text.begin(), text.end(), ',', ' ');
      std::istringstream stream(text);
      double number = 0.0;
      while (stream >> number) {
        parsed.push_back(number);
      }
    } else if (value.getType() == XmlRpc::XmlRpcValue::TypeArray) {
      for (int i = 0; i < value.size(); ++i) {
        if (value[i].getType() == XmlRpc::XmlRpcValue::TypeInt) {
          parsed.push_back(static_cast<int>(value[i]));
        } else if (value[i].getType() == XmlRpc::XmlRpcValue::TypeDouble) {
          parsed.push_back(static_cast<double>(value[i]));
        } else {
          ROS_ERROR("grad_lipschitz array must contain only numbers");
          return false;
        }
      }
    }
    if (parsed.empty() || parsed.size() > certified_cbf::kMaxRows) {
      ROS_ERROR("grad_lipschitz must contain between 1 and 16 values");
      return false;
    }
    lipschitz_count_ = parsed.size();
    for (std::size_t i = 0; i < parsed.size(); ++i) {
      if (!std::isfinite(parsed[i]) || parsed[i] < 0.0) {
        ROS_ERROR("grad_lipschitz values must be finite and non-negative");
        return false;
      }
      lipschitz_values_[i] = parsed[i];
    }
    return true;
  }

  void fillEffectiveLipschitz(
      std::size_t rows,
      std::array<double, certified_cbf::kMaxRows>& output) const {
    if (lipschitz_count_ == rows) {
      std::copy_n(lipschitz_values_.begin(), rows, output.begin());
      return;
    }
    const double conservative = *std::max_element(
        lipschitz_values_.begin(),
        lipschitz_values_.begin() + lipschitz_count_);
    output.fill(conservative);
  }

  std::string lipschitzDescription() const {
    std::ostringstream stream;
    for (std::size_t i = 0; i < lipschitz_count_; ++i) {
      if (i) stream << ',';
      stream << lipschitz_values_[i];
    }
    return stream.str();
  }

  bool isRecovery(const Packet& packet) const {
    bool has_violating_row = false;
    for (std::size_t row = 0; row < packet.row_count; ++row) {
      if (packet.h[row] >= 0.0) {
        continue;
      }
      has_violating_row = true;
      double rate = 0.0;
      for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
        rate += packet.gradient[row * certified_cbf::kJoints + joint]
              * packet.velocity[joint];
      }
      if (rate < 0.0) {
        return false;
      }
    }
    return has_violating_row;
  }

  void commandCallback(
      const vision_processing::CertifiedCbfCommandConstPtr& message) {
    const std::size_t rows = message->row_count;
    if (rows == 0 || rows > certified_cbf::kMaxRows ||
        message->barrier_value.size() != rows ||
        message->environment_hdot.size() != rows ||
        message->barrier_gradient.size() != rows * certified_cbf::kJoints) {
      ROS_ERROR_THROTTLE(1.0, "Rejected malformed certified CBF packet");
      return;
    }

    Packet packet;
    packet.valid = true;
    packet.solve_stamp = message->header.stamp;
    packet.receipt_stamp = ros::Time::now();
    packet.sequence = message->header.seq;
    packet.selection_generation = message->selection_generation;
    packet.row_count = rows;
    for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
      packet.q0[joint] = message->solve_joint_position[joint];
      packet.velocity[joint] = message->safe_joint_velocity[joint];
      if (!std::isfinite(packet.q0[joint]) ||
          !std::isfinite(packet.velocity[joint])) {
        ROS_ERROR_THROTTLE(1.0, "Rejected non-finite certified CBF packet");
        return;
      }
    }
    for (std::size_t row = 0; row < rows; ++row) {
      packet.h[row] = message->barrier_value[row];
      packet.environment_hdot[row] = message->environment_hdot[row];
      if (!std::isfinite(packet.h[row]) ||
          !std::isfinite(packet.environment_hdot[row])) {
        ROS_ERROR_THROTTLE(1.0, "Rejected non-finite certified CBF row");
        return;
      }
      for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
        const double value = message->barrier_gradient[
            row * certified_cbf::kJoints + joint];
        if (!std::isfinite(value)) {
          ROS_ERROR_THROTTLE(1.0, "Rejected non-finite CBF gradient");
          return;
        }
        packet.gradient[row * certified_cbf::kJoints + joint] = value;
      }
    }

    // Assumption monitor, deliberately computed outside update().  Pairs
    // spanning a critical-point selection change are not curvature samples.
    if (have_previous_packet_ && previous_packet_.row_count == rows &&
        previous_packet_.selection_generation == packet.selection_generation) {
      double dq_squared = 0.0;
      for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
        const double delta = packet.q0[joint] - previous_packet_.q0[joint];
        dq_squared += delta * delta;
      }
      const double dq = std::sqrt(dq_squared);
      if (dq > 1.0e-4) {
        for (std::size_t row = 0; row < rows; ++row) {
          if (packet.h[row] >= parameters_.h_activate ||
              previous_packet_.h[row] >= parameters_.h_activate) {
            continue;
          }
          double dg_squared = 0.0;
          for (std::size_t joint = 0; joint < certified_cbf::kJoints; ++joint) {
            const std::size_t index = row * certified_cbf::kJoints + joint;
            const double delta = packet.gradient[index]
                               - previous_packet_.gradient[index];
            dg_squared += delta * delta;
          }
          packet.empirical_lipschitz = std::max(
              packet.empirical_lipschitz, std::sqrt(dg_squared) / dq);
        }
      }
    }
    previous_packet_ = packet;
    have_previous_packet_ = true;
    command_buffer_.writeFromNonRT(packet);
  }

  void issfCallback(const std_msgs::Float32MultiArrayConstPtr& message) {
    if (message->data.size() >= 4 && message->data[3] > 0.5f) {
      issf_veto_until_.store(
          ros::Time::now().toSec() + issf_veto_hold_,
          std::memory_order_relaxed);
    }
  }

  std::vector<std::string> joint_names_;
  std::vector<hardware_interface::JointHandle> joint_handles_;
  realtime_tools::RealtimeBuffer<Packet> command_buffer_;
  ros::Subscriber command_subscriber_;
  ros::Subscriber issf_subscriber_;
  std::unique_ptr<
      realtime_tools::RealtimePublisher<std_msgs::Float32MultiArray>>
      status_publisher_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::Float64>>
      compute_time_publisher_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::Float64>>
      period_publisher_;
  std::unique_ptr<realtime_tools::RealtimePublisher<
      vision_processing::IssfEpsilonSample>> epsilon_sample_publisher_;

  certified_cbf::Parameters parameters_;
  std::array<double, certified_cbf::kMaxRows> lipschitz_values_{};
  std::size_t lipschitz_count_{0};
  Packet previous_packet_{};
  bool have_previous_packet_{false};
  std::atomic<double> issf_veto_until_{0.0};
  double command_timeout_{0.10};
  double max_abs_velocity_{0.70};
  double max_position_lead_{0.05};
  double release_tau_{0.05};
  double nominal_period_{0.001};
  double tick_decay_{0.98};
  double issf_veto_hold_{0.05};
  double alpha_{1.0};
  double horizon_{0.001};
  double epsilon_normal_alpha_threshold_{0.90};
  double epsilon_motion_threshold_{1.0e-4};
  int status_divisor_{20};
  int timing_divisor_{10};
  int epsilon_sample_divisor_{10};
  std::size_t tick_{0};
  std::array<double, certified_cbf::kJoints> position_reference_{};

  bool epsilon_reference_valid_{false};
  bool epsilon_calibration_started_{false};
  std::size_t epsilon_reference_row_count_{0};
  std::array<double, certified_cbf::kJoints>
      epsilon_previous_applied_velocity_{};
  std::array<double,
             certified_cbf::kMaxRows * certified_cbf::kJoints>
      epsilon_reference_gradient_{};
  std::array<bool, certified_cbf::kMaxRows> epsilon_reference_active_{};
  double epsilon_previous_alpha_{1.0};
  EpsilonPeak epsilon_normal_peak_{};
  EpsilonPeak epsilon_braking_peak_{};
};

}  // namespace vision_processing

PLUGINLIB_EXPORT_CLASS(
    vision_processing::CertifiedCbfVelocityController,
    controller_interface::ControllerBase)
