import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# EDIT: A.HABER
import io 

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# EDIT: A. HABER
# THESE ARE THE MODEL CHECKPOINTS 
# these are different models that you can play with

# sam2_hiera_large.pt
# sam2_hiera_base_plus.pt
# sam2_hiera_small.pt
# sam2_hiera_tiny.pt
sam2_checkpoint = "C:\\codes\\segment-anything-2\\checkpoints\\sam2_hiera_large.pt"


# THESE are yaml files corresponding to the checkpoints, you have to match 
# them with the corresponding checkpoint file
#sam2_hiera_l.yaml
#sam2_hiera_b+.yaml
#sam2_hiera_s.yaml
#sam2_hiera_t.yaml
model_cfg = "sam2_hiera_l.yaml"

# create the predictor object
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# mask function 
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# function for showing points
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  

# EDIT: A. Haber
# here, the path should be windows path and not the linux path 
# in windows " C:\codes\sam2\segment-anything-2\video"
# is in Python "C:\\codes\\sam2\\segment-anything-2\\output
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "C:\\codes\\segment-anything-2\\video"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# EDIT: A. HABER: show the first frame, identify the object pixels
# writ
# take a look the first video frame (or change), and find the object points 
# at least two of them.
print(frame_names)

frame_idx = 0
plt.figure(figsize=(12, 8))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
plt.show()

print("Press Enter to continue...")
input()
print("Continuing...")




inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# EDIT: A. HABER 
# here enter the point coordinates [x,y] as rows of a numpy array

points = np.array([[126, 264], [200, 114]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1, 1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.show()

print("Press Enter to continue...")
input()
print("Continuing...")

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")

# EDIT: A. Haber
# define the figure outside of the loop
fig = plt.figure(figsize=(6, 4))
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    
    plt.title(f"frame {out_frame_idx}")
    im=plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])),animated=True)
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    # EDIT: HABER 
    # here, just save the files with an increasing index in the folder called output
    plt.savefig(f'output/s{out_frame_idx}.png')
    #plt.show()

    


