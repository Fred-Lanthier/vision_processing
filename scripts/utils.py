from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF
import xacro
import rospy
import numpy as np

def compute_T_child_parent_xacro(xacro_file, child_link, parent_link):
        
        doc = xacro.process_file(xacro_file)
        urdf_string = doc.toxml()
        
        # Parse the URDF
        robot = URDF.from_xml_string(urdf_string)

        # Trouver le joint connectant les deux links
        target_joint = None
        for joint in robot.joints:
            # On cherche le joint qui relie spécifiquement ton parent à ton enfant
            if joint.parent == parent_link and joint.child == child_link:
                target_joint = joint
                break
        
        if target_joint is None:
            rospy.logerr(f"Joint {parent_link} -> {child_link} non trouvé dans le URDF.")
            return None
        
        xyz = target_joint.origin.xyz
        rpy = target_joint.origin.rpy
        
        rotation = R.from_euler('xyz', rpy).as_matrix()
        translation = np.array(xyz)

        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation

        return T