#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vision_processing {
namespace certified_cbf {

constexpr std::size_t kJoints = 7;
constexpr std::size_t kMaxRows = 16;

struct Parameters {
  double h_stop{0.0};
  double h_activate{0.04};
  double exp_kappa{0.0};
  double recovery_speed{0.05};
  double recovery_depth{0.015};
  double recovery_slack{0.5};
  double issf_epsilon{0.0};
  double issf_rho{0.0};
};

struct Result {
  double alpha{1.0};
  double minimum_margin{std::numeric_limits<double>::infinity()};
  std::size_t braking_rows{0};
  bool feasible{true};
};

struct IntervalResult {
  double alpha{0.0};
  bool feasible{false};
};

// Largest common alpha in [0,1] satisfying
//     a_i + b_i alpha - c_i alpha^2 >= 0
// for every row.  This is the scalar certified-reflex problem; the feasible
// set of each concave quadratic is an interval.
inline IntervalResult intersectIntervals(const double* a, const double* b,
                                         const double* c, std::size_t count,
                                         double tolerance = 1.0e-12) {
  double common_lo = 0.0;
  double common_hi = 1.0;
  for (std::size_t i = 0; i < count; ++i) {
    double lo = 0.0;
    double hi = 1.0;
    if (c[i] > tolerance) {
      const double discriminant = b[i] * b[i] + 4.0 * c[i] * a[i];
      if (discriminant < 0.0) {
        return {};
      }
      const double root = std::sqrt(std::max(discriminant, 0.0));
      lo = std::max(0.0, (b[i] - root) / (2.0 * c[i]));
      hi = std::min(1.0, (b[i] + root) / (2.0 * c[i]));
    } else if (b[i] > tolerance) {
      lo = std::max(0.0, -a[i] / b[i]);
    } else if (b[i] < -tolerance) {
      hi = std::min(1.0, a[i] / (-b[i]));
    } else if (a[i] < -tolerance) {
      return {};
    }
    if (lo > hi + tolerance) {
      return {};
    }
    common_lo = std::max(common_lo, lo);
    common_hi = std::min(common_hi, hi);
  }
  if (common_hi < common_lo - tolerance) {
    return {};
  }
  return {std::max(0.0, std::min(1.0, common_hi)), true};
}

// CPU-only certified brake used by the real-time controller.  q is read
// directly from the hardware interface, so unlike the rospy implementation
// there is no /joint_states sampling gap or odometric tail to estimate.
inline Result computeScale(
    const std::array<double, kJoints>& q,
    const std::array<double, kJoints>& q0,
    const std::array<double, kJoints>& velocity,
    const std::array<double, kMaxRows>& h,
    const std::array<double, kMaxRows>& environment_hdot,
    const std::array<double, kMaxRows * kJoints>& gradient,
    const std::array<double, kMaxRows>& lipschitz,
    std::size_t row_count, double solve_age, double horizon,
    const Parameters& parameters) {
  Result result;
  row_count = std::min(row_count, kMaxRows);
  if (row_count == 0) {
    return result;
  }

  std::array<double, kJoints> drift{};
  double drift_norm_squared = 0.0;
  double velocity_norm_squared = 0.0;
  double drift_velocity = 0.0;
  for (std::size_t j = 0; j < kJoints; ++j) {
    drift[j] = q[j] - q0[j];
    drift_norm_squared += drift[j] * drift[j];
    velocity_norm_squared += velocity[j] * velocity[j];
    drift_velocity += drift[j] * velocity[j];
  }
  const double velocity_norm = std::sqrt(velocity_norm_squared);
  solve_age = std::max(solve_age, 0.0);
  horizon = std::max(horizon, 1.0e-6);

  std::array<double, kMaxRows> active_a{};
  std::array<double, kMaxRows> active_b{};
  std::array<double, kMaxRows> active_c{};
  std::size_t active_count = 0;
  bool full_command_feasible = true;

  for (std::size_t row = 0; row < row_count; ++row) {
    double gradient_drift = 0.0;
    double gradient_velocity = 0.0;
    double gradient_norm_squared = 0.0;
    for (std::size_t j = 0; j < kJoints; ++j) {
      const double g = gradient[row * kJoints + j];
      gradient_drift += g * drift[j];
      gradient_velocity += g * velocity[j];
      gradient_norm_squared += g * g;
    }
    const double L = std::max(lipschitz[row], 0.0);
    double h_now = h[row] + gradient_drift
                 - 0.5 * L * drift_norm_squared;
    const double gradient_norm = std::sqrt(gradient_norm_squared);
    const double issf_hold = gradient_norm * parameters.issf_epsilon * horizon;
    double b = (gradient_velocity - L * drift_velocity) * horizon
             - gradient_norm * parameters.issf_rho
               * velocity_norm * horizon;
    const double c = 0.5 * L * velocity_norm_squared * horizon * horizon;

    if (parameters.exp_kappa > 0.0) {
      if (h_now >= parameters.h_activate) {
        continue;
      }
      const double depth = std::max(parameters.recovery_depth, 1.0e-6);
      const double recovery_demand = parameters.recovery_slack
          * parameters.recovery_speed
          * std::min(std::max(-h_now, 0.0) / depth, 1.0);
      const double allowed_rate = h_now >= 0.0
          ? parameters.exp_kappa * h_now : -recovery_demand;
      const double a = allowed_rate * horizon
                     - environment_hdot[row] * (solve_age + horizon)
                     - issf_hold;
      const double margin = a + b - c;
      result.minimum_margin = std::min(result.minimum_margin, margin);
      if (margin < 0.0) {
        ++result.braking_rows;
        full_command_feasible = false;
      }
      active_a[active_count] = a;
      active_b[active_count] = b;
      active_c[active_count] = c;
      ++active_count;
      continue;
    }

    double a = h_now
             - environment_hdot[row] * (solve_age + horizon)
             - issf_hold - parameters.h_stop;
    const bool recovery = a <= 0.0 && gradient_velocity >= 0.0;
    if (h_now >= parameters.h_activate || recovery) {
      continue;
    }
    const double margin = a + b - c;
    result.minimum_margin = std::min(
        result.minimum_margin, margin + parameters.h_stop);
    if (margin >= 0.0) {
      continue;
    }
    ++result.braking_rows;
    full_command_feasible = false;
    if (a <= 0.0) {
      result.alpha = 0.0;
      return result;
    }
    double alpha = 1.0;
    if (c < 1.0e-12) {
      alpha = b < 0.0 ? a / std::max(-b, 1.0e-12) : 1.0;
    } else {
      const double discriminant = std::max(b * b + 4.0 * c * a, 0.0);
      alpha = (b + std::sqrt(discriminant)) / (2.0 * c);
    }
    result.alpha = std::min(result.alpha,
                            std::max(0.0, std::min(1.0, alpha)));
  }

  if (parameters.exp_kappa > 0.0 && active_count > 0
      && !full_command_feasible) {
    const IntervalResult intersection = intersectIntervals(
        active_a.data(), active_b.data(), active_c.data(), active_count);
    result.alpha = intersection.feasible ? intersection.alpha : 0.0;
    result.feasible = intersection.feasible;
  }
  return result;
}

}  // namespace certified_cbf
}  // namespace vision_processing
