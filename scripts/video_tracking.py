import glob
import os
from sam3.model_builder import build_sam3_video_predictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam3.visualization_utils import (
    load_frame,
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

try:
    import rospkg
    ROSPKG_AVAILABLE = True
except ImportError:
    ROSPKG_AVAILABLE = False
    print("⚠️ rospkg not found. Using fallback paths.")


plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12

def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def abs_to_rel_coords(coords, IMG_WIDTH, IMG_HEIGHT, coord_type="point"):
    """Convert absolute coordinates to relative coordinates (0-1 range)

    Args:
        coords: List of coordinates
        coord_type: 'point' for [x, y] or 'box' for [x, y, w, h]
    """
    if coord_type == "point":
        return [[x / IMG_WIDTH, y / IMG_HEIGHT] for x, y in coords]
    elif coord_type == "box":
        return [
            [x / IMG_WIDTH, y / IMG_HEIGHT, w / IMG_WIDTH, h / IMG_HEIGHT]
            for x, y, w, h in coords
        ]
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}")

def loading_video(video_path):
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.png"))
    video_frames_for_vis.sort(
        # Sort by frame number considering the file name static_rgb_step_0XXX.png where XXX is the frame number
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
    )
    return video_frames_for_vis

def main():
    if ROSPKG_AVAILABLE:
        try:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('vision_processing')
        except Exception as e:
            print(f"⚠️ Error using rospkg: {e}")
            package_path = "/home/flanthier/Github/src/vision_processing"
    else:
        package_path = "/home/flanthier/Github/src/vision_processing"
        
    print(f"📂 Package path: {package_path}")
    
    # Input directory: images_trajectory
    images_dir = os.path.join(package_path, 'scripts', 'images_trajectory')
    output_dir = os.path.join(package_path, 'scripts', 'results_trajectory')

    # Load video frames
    video_frames_for_vis = loading_video(images_dir)
    print(f"Video frames loaded: {len(video_frames_for_vis)}")
    # Load SAM3 Video Predictor
    gpus_to_use = range(torch.cuda.device_count())
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    # Open an inference session on this video
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=images_dir,
        )
    )
    session_id = response["session_id"]

    _ = predictor.handle_request(
        request=dict(
            type="reset_session",
            session_id=session_id,
        )
    )

    prompt_text_str = "robot"
    frame_idx = 0  # add a text prompt on frame 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=frame_idx,
            text=prompt_text_str,
        )
    )
    out = response["outputs"]
    
    plt.close("all")
    visualize_formatted_frame_output(
        frame_idx,
        video_frames_for_vis,
        outputs_list=[prepare_masks_for_visualization({frame_idx: out})],
        titles=["SAM 3 Dense Tracking outputs"],
        figsize=(6, 4),
    )   
    
    # now we propagate the outputs from frame 0 to the end of the video and collect all outputs
    outputs_per_frame = propagate_in_video(predictor, session_id)

    # finally, we reformat the outputs for visualization and plot the outputs every 60 frames
    outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

    vis_frame_stride = 60
    plt.close("all")
    for frame_idx in range(0, len(outputs_per_frame), vis_frame_stride):
        print(f"Processing frame {frame_idx}")
        visualize_formatted_frame_output(
            frame_idx,
            video_frames_for_vis,
            outputs_list=[outputs_per_frame],
            titles=["SAM 3 Dense Tracking outputs"],
            figsize=(6, 4),
        )

if __name__ == "__main__":
    main()