#!/home/flanthier/Github/src/vision_processing/venv_sam3/bin/python3
# -*- coding: utf-8 -*-
# venv python (3.10) pinned: muggled_sam / SAM3 use 3.10+ syntax. ROS packages
# (rospy/tf2) reach this interpreter via the PYTHONPATH roslaunch sets, exactly
# like the installed casf/cbf nodes that also run on the venv python.
"""
sam3_sam2_node_pp.py
====================
PICK-AND-PLACE perception: detect + track TWO objects from the STATIC camera and
publish a 2D mask for each:
  * the RED CUBE  (pick target)  -> /vision/sam2_mask_cube
  * the BROWN BOX (place target) -> /vision/sam2_mask_box

`_pp` copy of sam3_sam2_node.py (feeding tracks ONE target on the wrist cam).
Same SAM3-init + SAM2-track machinery, generalized to a list of objects. The
feeding node is left untouched. Each mask is consumed by its own
point_cloud_projector instance to produce /perception/target_cube and
/perception/target_box for the conditioning node.
"""
import rospy
import rospkg
import sys
import os
import cv2
import numpy as np
import torch
import threading
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from vision_processing.srv import Sam3Segment, Sam3SegmentRequest

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('vision_processing')
muggled_sam_path = os.path.join(pkg_path, "third_party", "muggled_sam")
if muggled_sam_path not in sys.path:
    sys.path.insert(0, muggled_sam_path)
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

try:
    from muggled_sam.make_sam import make_sam_from_state_dict
    from muggled_sam.demo_helpers.video_data_storage import SAMVideoObjectResults
except ImportError as e:
    print(f"❌ Erreur Import Muggled SAM : {e}")
    sys.exit(1)


class TrackedObject:
    """One SAM3-initialized, SAM2-tracked object with its own mask publisher."""
    def __init__(self, name, prompt, mask_topic):
        self.name = name
        self.prompt = prompt
        self.mem = None
        self.pub = rospy.Publisher(mask_topic, Image, queue_size=1)


class Sam3Sam2NodePP:
    def __init__(self):
        rospy.init_node('sam3_sam2_node_pp')

        self.sam2_model_path = rospy.get_param(
            "~sam2_model_path", "/home/flanthier/segment-anything-2/checkpoints/sam2.1_hiera_tiny.pt")
        self.cube_prompt = rospy.get_param("~cube_prompt", "red cube")
        self.box_prompt = rospy.get_param("~box_prompt", "brown box")
        self.tracking_rate_hz = float(rospy.get_param("~tracking_rate_hz", 5.0))
        self.sam3_init_retry_period = max(0.1, float(rospy.get_param("~sam3_init_retry_period", 8.0)))
        self.sam2_max_side_length = rospy.get_param("~sam2_max_side_length", None)
        self.camera_frame = rospy.get_param("~camera_frame", "camera_static_optical_frame")

        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.imgenc_config_dict = {"max_side_length": self.sam2_max_side_length, "use_square_sizing": True}

        rospy.loginfo(f"⏳ Chargement SAM2 Tiny depuis : {self.sam2_model_path}")
        try:
            _, self.sammodel = make_sam_from_state_dict(self.sam2_model_path)
            self.sammodel.to(device=self.device, dtype=self.dtype)
            self.sammodel.eval()
        except Exception as e:
            rospy.logerr(f"❌ Impossible de charger SAM2 : {e}")
            sys.exit(1)

        # The two objects to detect + track.
        self.objects = [
            TrackedObject("cube", self.cube_prompt, "/vision/sam2_mask_cube"),
            TrackedObject("box",  self.box_prompt,  "/vision/sam2_mask_box"),
        ]

        self.frame_idx = 0
        self.next_init_time = 0.0
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        rospy.loginfo("✅ Nœud DUAL-TARGET SAM3+SAM2 (cube + box) prêt.")
        self.ai_thread = threading.Thread(target=self.ai_worker)
        self.ai_thread.daemon = True
        self.ai_thread.start()
        self.run_loop()

    def init_with_sam3(self, rgb_cv):
        service_name = '/sam3/segment'
        try:
            rospy.wait_for_service(service_name, timeout=90.0)
            sam3_segment = rospy.ServiceProxy(service_name, Sam3Segment)
        except rospy.ROSException:
            rospy.logerr(f"❌ Service {service_name} indisponible après 90s.")
            return False

        with torch.inference_mode():
            init_encoded_img, _, _ = self.sammodel.encode_image(rgb_cv, **self.imgenc_config_dict)

        for obj in self.objects:
            if obj.mem is not None:
                continue
            req = Sam3SegmentRequest(rgb_image=self.bridge.cv2_to_imgmsg(rgb_cv, encoding="bgr8"),
                                     text_prompt=obj.prompt, confidence_threshold=0.1)
            resp = sam3_segment(req)
            if resp.success:
                mask_t = torch.from_numpy(self.bridge.imgmsg_to_cv2(resp.mask, "mono8") > 0
                                          ).to(self.device).float().unsqueeze(0).unsqueeze(0)
                with torch.inference_mode():
                    init_mem, init_ptr = self.sammodel.initialize_from_mask(init_encoded_img, mask_t > 0)
                obj.mem = SAMVideoObjectResults.create().store_prompt_result(self.frame_idx, init_mem, init_ptr)
                rospy.loginfo(f"🎯 '{obj.prompt}' ({obj.name}) initialisé.")
            else:
                rospy.logwarn(f"⚠️ Échec SAM3 {obj.name} ('{obj.prompt}'): {resp.message}")

        return all(o.mem is not None for o in self.objects)

    def _publish_mask(self, obj, mask_bool, shape, stamp):
        img = (mask_bool * 255).astype(np.uint8) if mask_bool is not None else np.zeros(shape, dtype=np.uint8)
        msg = self.bridge.cv2_to_imgmsg(img, encoding="mono8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        obj.pub.publish(msg)

    def ai_worker(self):
        rate = rospy.Rate(max(self.tracking_rate_hz, 0.1))
        while not rospy.is_shutdown():
            with self.frame_lock:
                frame_data = self.latest_frame
                self.latest_frame = None
            if frame_data is None:
                rate.sleep(); continue
            rgb_cv, frame_time = frame_data
            shape = (rgb_cv.shape[0], rgb_cv.shape[1])

            if any(o.mem is None for o in self.objects):
                if time.time() >= self.next_init_time:
                    try:
                        ok = self.init_with_sam3(rgb_cv)
                    except Exception as e:
                        rospy.logerr_throttle(5.0, f"SAM3 init error: {e}")
                        ok = False
                    if not ok:
                        self.next_init_time = time.time() + self.sam3_init_retry_period
                for o in self.objects:
                    if o.mem is None:
                        self._publish_mask(o, None, shape, frame_time)

            try:
                with torch.inference_mode():
                    encoded_imgs_list, _, _ = self.sammodel.encode_image(rgb_cv, **self.imgenc_config_dict)
                for obj in self.objects:
                    if obj.mem is None:
                        continue
                    with torch.inference_mode():
                        score, best_idx, preds, mem_enc, obj_ptr = self.sammodel.step_video_masking(
                            encoded_imgs_list, **obj.mem.to_dict())
                    if score.item() > 0:
                        obj.mem.store_frame_result(self.frame_idx, mem_enc, obj_ptr)
                        mask_f = cv2.resize(preds[0, best_idx].cpu().float().numpy().squeeze(),
                                            (rgb_cv.shape[1], rgb_cv.shape[0]))
                        self._publish_mask(obj, (mask_f > 0), shape, frame_time)
                    else:
                        obj.mem = None
                        self.next_init_time = 0.0
                        self._publish_mask(obj, None, shape, frame_time)
                self.frame_idx += 1
            except Exception as e:
                rospy.logerr_throttle(1.0, f"Erreur AI: {e}")
            rate.sleep()

    def ros_image_callback(self, rgb_msg):
        try:
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            with self.frame_lock:
                self.latest_frame = (rgb_cv.copy(), rgb_msg.header.stamp)
        except Exception as e:
            rospy.logerr_throttle(5, f"Décodage image: {e}")

    def run_loop(self):
        rgb_topic = rospy.get_param("~rgb_topic", "/camera_static/color/image_raw")
        rospy.loginfo(f"🛰️ Souscription à {rgb_topic}")
        rospy.Subscriber(rgb_topic, Image, self.ros_image_callback, queue_size=1)
        rospy.spin()


if __name__ == '__main__':
    Sam3Sam2NodePP()
