from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import xacro
import rospy
import numpy as np


def compute_T_child_parent_xacro(xacro_file, child_link, parent_link):
    doc = xacro.process_file(xacro_file)
    robot = ET.fromstring(doc.toxml())

    target_joint = None
    for joint in robot.findall('joint'):
        parent = joint.find('parent')
        child = joint.find('child')
        if (parent is not None and child is not None
                and parent.get('link') == parent_link
                and child.get('link') == child_link):
            target_joint = joint
            break

    if target_joint is None:
        rospy.logerr(f"Joint {parent_link} -> {child_link} non trouvé dans le URDF.")
        return None

    origin = target_joint.find('origin')
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin is not None:
        xyz = [float(value) for value in origin.get('xyz', '0 0 0').split()]
        rpy = [float(value) for value in origin.get('rpy', '0 0 0').split()]

    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T
