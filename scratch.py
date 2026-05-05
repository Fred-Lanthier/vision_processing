import numpy as np
import time
import pybullet as pb
import rospkg

pb.connect(pb.DIRECT)
rospack = rospkg.RosPack()
urdf_path = rospack.get_path('vision_processing') + "/third_party/RDF/collision_avoidance_example/xarm7_urdf/xarm7_FT_EE.urdf"
robot_id = pb.loadURDF(urdf_path, useFixedBase=True)

from vision_processing import fast_ik_module
ee_name = "link_eef"
try:
    ik_cpp = fast_ik_module.FastIK(urdf_path, ee_name)
    print("FastIK loaded!")
except Exception as e:
    print(e)
