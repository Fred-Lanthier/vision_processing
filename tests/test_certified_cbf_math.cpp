#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "vision_processing/certified_cbf_math.h"

namespace cc = vision_processing::certified_cbf;

TEST(CertifiedCbfIntervals, MatchesLinearAndQuadraticCases) {
  const double a[] = {0.25, -0.25};
  const double b[] = {-1.0, 1.0};
  const double c[] = {0.0, 0.0};
  const cc::IntervalResult result = cc::intersectIntervals(a, b, c, 2);
  ASSERT_TRUE(result.feasible);
  EXPECT_NEAR(result.alpha, 0.25, 1.0e-12);

  const double qa[] = {0.1875};
  const double qb[] = {0.0};
  const double qc[] = {1.0};
  const cc::IntervalResult quadratic = cc::intersectIntervals(qa, qb, qc, 1);
  ASSERT_TRUE(quadratic.feasible);
  EXPECT_NEAR(quadratic.alpha, std::sqrt(0.1875), 1.0e-12);
}

TEST(CertifiedCbfIntervals, DetectsConflictingRecoveryAndBrake) {
  const double a[] = {-0.8, 0.2};
  const double b[] = {1.0, -1.0};
  const double c[] = {0.0, 0.0};
  const cc::IntervalResult result = cc::intersectIntervals(a, b, c, 2);
  EXPECT_FALSE(result.feasible);
  EXPECT_DOUBLE_EQ(result.alpha, 0.0);
}

TEST(CertifiedCbfScale, ExponentialConditionBrakesAtExpectedScale) {
  std::array<double, cc::kJoints> q{};
  std::array<double, cc::kJoints> q0{};
  std::array<double, cc::kJoints> velocity{};
  std::array<double, cc::kMaxRows> h{};
  std::array<double, cc::kMaxRows> env{};
  std::array<double, cc::kMaxRows * cc::kJoints> gradient{};
  std::array<double, cc::kMaxRows> lipschitz{};
  velocity[0] = -1.0;
  gradient[0] = 1.0;
  h[0] = 0.001;

  cc::Parameters parameters;
  parameters.exp_kappa = 25.0;
  parameters.h_activate = 0.04;
  const cc::Result result = cc::computeScale(
      q, q0, velocity, h, env, gradient, lipschitz,
      1, 0.0, 0.001, parameters);
  EXPECT_TRUE(result.feasible);
  EXPECT_EQ(result.braking_rows, 1u);
  EXPECT_NEAR(result.alpha, 0.025, 1.0e-12);
}

TEST(CertifiedCbfScale, FarRowsStayInactive) {
  std::array<double, cc::kJoints> q{};
  std::array<double, cc::kJoints> q0{};
  std::array<double, cc::kJoints> velocity{};
  std::array<double, cc::kMaxRows> h{};
  std::array<double, cc::kMaxRows> env{};
  std::array<double, cc::kMaxRows * cc::kJoints> gradient{};
  std::array<double, cc::kMaxRows> lipschitz{};
  h[0] = 1.0;
  velocity[0] = -1.0;
  gradient[0] = 1.0;
  cc::Parameters parameters;
  parameters.exp_kappa = 25.0;
  const cc::Result result = cc::computeScale(
      q, q0, velocity, h, env, gradient, lipschitz,
      1, 10.0, 0.001, parameters);
  EXPECT_DOUBLE_EQ(result.alpha, 1.0);
  EXPECT_EQ(result.braking_rows, 0u);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
