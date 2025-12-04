import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import rospkg
import json
import os
import sys


class RealTime3DDP:
    def __init__(self):
        rospack = rospkg.RosPack()
        model_path = os.path.join(rospack.get_path("vision_processing"), "dp3_policy_best_robust.ckpt")
        self.model = torch.load(model_path)
        
    def predict(self, state):
        