#!/bin/bash
# Variables temporaires pour ce projet seulement
export ROS_MASTER_URI=http://132.207.24.13:11311
export ROS_IP=132.207.24.13
echo "Variables temporaires configurées pour ce projet"
echo "ROS_MASTER_URI=$ROS_MASTER_URI"
echo "ROS_IP=$ROS_IP"
roslaunch Process_image vision_server.launch