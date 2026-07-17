#!/usr/bin/env bash
# Generate the xArm7 pick-and-place robot_description for GAZEBO, with the
# panda_finger_joint2 <mimic> tag stripped.
#
# Why: under gazebo_ros_control/DefaultRobotHWSim the URDF <mimic> on the finger
# joint prevents an effort handle from being registered for panda_finger_joint2,
# so franka_gazebo/FrankaGripperSim (which claims BOTH finger handles) fails to
# start and the fingers stay limp/uncontrolled. FrankaGripperSim already
# simulates the finger coupling in software (its IDLE state makes finger2 track
# finger1), so the URDF mimic is unnecessary here and must be removed for the
# fingers to be driven and to close. Under FrankaHWSim (panda) this is handled
# internally, which is why the panda sim does not need this.
#
# Used ONLY for the Gazebo robot_description. The planner/CBF load the xacro
# directly (the mimic there is harmless — they use joint positions only), so the
# kinematic model they see is unchanged.
#
# Args are forwarded to xacro (e.g. arm_id:=panda), exactly like the plain
# `xacro <file> arm_id:=panda` call it replaces.
set -euo pipefail

PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
XACRO_FILE="$PKG_DIR/urdf/xarm7_pickplace.xacro"

# xacro warnings go to stderr; only the URDF reaches stdout. pipefail makes a
# xacro failure abort the whole pipe instead of emitting a truncated URDF.
xacro "$XACRO_FILE" "$@" | sed '/<mimic[^>]*finger_joint1/d'
