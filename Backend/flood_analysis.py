"""
flood_analysis.py - Pixel-level flood change detection
"""
import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any

def run_flood_pixel_analysis(before_path: str, after_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Run pixel-level change detection analysis between before and after images.
    
    Args:
        before_path: Path to before image
        after_path: Path to after image
        output_dir: Directory to save analysis outputs
        
    Returns:
        Dictionary containing analysis results and output paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    before_img = cv2.imread(before_path)
    after_img = cv2.imread(after_path)
    
    if before_img is None or after_img is None:
        return {"error": "Could not load images"}
    
    # Resize if needed
    if before_img.shape != after_img.shape:
        after_img = cv2.resize(after_img, (before_img.shape[1], before_img.shape[0]))
    
    # Compute simple difference
    diff = cv2.absdiff(before_img, after_img)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get changed pixels
    _, change_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    changed_pixels = np.sum(change_mask > 0)
    total_pixels = change_mask.shape[0] * change_mask.shape[1]
    
    results = {
        "total_pixels": int(total_pixels),
        "changed_pixels": int(changed_pixels),
        "change_percentage": float(changed_pixels / total_pixels * 100) if total_pixels > 0 else 0,
    }
    
    # Save summary
    df = pd.DataFrame([results])
    csv_path = os.path.join(output_dir, "pixel_change_summary.csv")
    df.to_csv(csv_path, index=False)
    
    # Save change mask visualization
    change_vis_path = os.path.join(output_dir, "change_mask.png")
    cv2.imwrite(change_vis_path, change_mask)
    
    return {
        **results,
        "csv_path": csv_path,
        "change_mask_path": change_vis_path,
    }

# Legacy script code below (for backward compatibility)
# -------------------------------------------------------

# Listing files in /data to see available images and masks
import os, textwrap, json, pathlib, sys
base = "/data"
files = []
for root, dirs, filenames in os.walk(base):
    for f in filenames:
        files.append(os.path.join(root, f))
output = {"count": len(files), "files": files[:200]}
print(json.dumps(output, indent=2))
# STDOUT/STDERR
# {
#  "count": 2,
#  "files": [
#  "/data/Screenshot 2025-10-30 at 10.15.02 PM.png",
#  "/data/Screenshot 2025-10-30 at 10.15.12 PM.png"
#  ]
# }
# Run an analysis pipeline on the two provided RGB images.
# - Align images (ECC)
# - Histogram match
# - Compute ExG and G-R proxies
# - Compute difference maps and simple change masks
# - Save visual outputs and a CSV summary of candidate affected areas (pixel counts)
# - Create a Jupyter notebook (.ipynb) that includes the SAM inference snippet (non-runnable here)
# and the steps we executed, for reproducibility.
import os, json, nbformat, math
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import numpy as np, cv2
from skimage import exposure
import matplotlib.pyplot as plt
plt.ioff()
# file paths
before_fp = "D:/Flood Damage Identification and Quantification Mobile Application/Backend/data/before.png"

after_fp = "D:/Flood Damage Identification and Quantification Mobile Application/Backend/data/after.png"

out_dir = "D:/Flood Damage Identification and Quantification Mobile Application/Backend/data/flood_pipeline_outputs"


os.makedirs(out_dir, exist_ok=True)
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def histogram_match(src, ref):
    matched = np.zeros_like(src, dtype=np.uint8)
    for ch in range(3):
        matched[..., ch] = exposure.match_histograms(src[..., ch], ref[..., ch])
    return matched
def align_images_ecc(ref_img, moving_img, warp_mode=cv2.MOTION_AFFINE, num_iter=2000, eps=1e-6):
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    mov_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, eps)
    try:
        cc, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria, None, 1)
    except cv2.error as e:
        print("ECC failed:", e)
        return moving_img, None
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(moving_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(moving_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, warp_matrix
# read
img_before = read_rgb(before_fp)
img_after = read_rgb(after_fp)
# align after to before
aligned_after, warp = align_images_ecc(img_before, img_after, warp_mode=cv2.MOTION_AFFINE, num_iter=3000)
aligned_after = histogram_match(aligned_after, img_before)
# compute proxies
def compute_exg(img):
    R = img[...,0].astype(np.float32)
    G = img[...,1].astype(np.float32)
    B = img[...,2].astype(np.float32)
    exg = 2*G - R - B
    return exg
def compute_g_minus_r(img):
    return img[...,1].astype(np.float32) - img[...,0].astype(np.float32)
def brightness(img):
    return img.mean(axis=2).astype(np.float32)
exg_before = compute_exg(img_before)
exg_after = compute_exg(aligned_after)
gmr_before = compute_g_minus_r(img_before)
gmr_after = compute_g_minus_r(aligned_after)
bright_before = brightness(img_before)
bright_after = brightness(aligned_after)
# difference maps
exg_diff = exg_before - exg_after # positive -> vegetation loss
gmr_diff = gmr_after - gmr_before # positive -> greener (or water if dark)
bright_diff = bright_after - bright_before
# simple masks with suggested thresholds (tweakable)
EXG_LOSS_T = 20.0 # ExG drop > threshold -> veg loss candidate
GMR_WATER_T = 10.0 # G-R increase > threshold
BRIGHT_DARK = 100.0 # brightness less than this suggests water
veg_loss_mask = (exg_diff > EXG_LOSS_T)
water_candidate_mask = ((gmr_after > GMR_WATER_T) & (bright_after < BRIGHT_DARK))
# Save images for visualization
def save_vis(img, path):
    import imageio
    imageio.imwrite(path, img.astype(np.uint8))
# create viz overlays
import matplotlib.pyplot as plt
def overlay_mask_on_rgb(rgb, mask, color=(255,0,0), alpha=0.4):
    overlay = rgb.copy().astype(np.float32)
    overlay[mask] = overlay[mask]*(1-alpha) + np.array(color)*(alpha)
    return np.clip(overlay,0,255).astype(np.uint8)
overlay_veg_loss = overlay_mask_on_rgb(aligned_after, veg_loss_mask, color=(255,0,0), alpha=0.35)
overlay_water = overlay_mask_on_rgb(aligned_after, water_candidate_mask, color=(0,0,255), alpha=0.35)
save_vis(img_before, os.path.join(out_dir, "before_rgb.png"))
save_vis(aligned_after, os.path.join(out_dir, "after_rgb_aligned_matched.png"))
save_vis((255*( (exg_before - exg_before.min())/(exg_before.max()-exg_before.min()+1e-9) )).astype(np.uint8), os.path.join(out_dir, "exg_before_norm.png"))
save_vis((255*( (exg_after - exg_after.min())/(exg_after.max()-exg_after.min()+1e-9) )).astype(np.uint8), os.path.join(out_dir, "exg_after_norm.png"))
save_vis((255*( (exg_diff - exg_diff.min())/(exg_diff.max()-exg_diff.min()+1e-9) )).astype(np.uint8), os.path.join(out_dir, "exg_diff_norm.png"))
save_vis(overlay_veg_loss, os.path.join(out_dir, "overlay_veg_loss.png"))
save_vis(overlay_water, os.path.join(out_dir, "overlay_water_candidate.png"))
# compute summary metrics (pixel counts and area assuming 1 pixel -> unknown area; report pixels)
summary = {
    "total_pixels": int(img_before.shape[0]*img_before.shape[1]),
    "veg_loss_pixels": int(np.sum(veg_loss_mask)),
    "water_candidate_pixels": int(np.sum(water_candidate_mask)),
    "veg_loss_pct": float(np.sum(veg_loss_mask))/float(img_before.shape[0]*img_before.shape[1])*100.0,
    "water_candidate_pct": float(np.sum(water_candidate_mask))/float(img_before.shape[0]*img_before.shape[1])*100.0
}
import pandas as pd
df = pd.DataFrame([summary])
csv_out = os.path.join(out_dir, "pixel_change_summary.csv")
df.to_csv(csv_out, index=False)
# create a simple PNG montage for quick view
fig, axes = plt.subplots(2,3, figsize=(12,8))
axes = axes.flatten()
axes[0].imshow(img_before); axes[0].set_title("Before RGB"); axes[0].axis("off")
axes[1].imshow(aligned_after); axes[1].set_title("After RGB (aligned & matched)"); axes[1].axis("off")
axes[2].imshow(overlay_veg_loss); axes[2].set_title("Veg loss candidates (red)"); axes[2].axis("off")
axes[3].imshow(overlay_water); axes[3].set_title("Water candidates (blue)"); axes[3].axis("off")
axes[4].imshow((exg_diff - exg_diff.min())/(exg_diff.max()-exg_diff.min()+1e-9)); axes[4].set_title("ExG diff (norm)"); axes[4].axis("off")
axes[5].axis("off")
montage_fp = os.path.join(out_dir, "montage.png")
plt.tight_layout(); plt.savefig(montage_fp, dpi=150); plt.close()
# Now create a Jupyter notebook that includes:
# - instructions and SAM inference snippet (explanation why not run here)
# - the code we executed (alignment + proxies + masks)
nb = new_notebook()
nb.cells = []
nb.cells.append(new_markdown_cell("# Flood change detection notebook\nThis notebook contains:\n\n- A SAM inference snippet (instructions & code) — NOTE: model weights and GPU are required; not run here.\n- Steps to align images, compute RGB proxies (ExG, G-R), and produce candidate change masks.\n\nFiles produced by this run are saved in `/data/flood_pipeline_outputs/`.\n"))
# SAM inference snippet (user must download model and set path)
sam_snippet = r'''# --- SAM inference snippet (requires segment-anything model & weights) ---
# pip install git+https://github.com/facebookresearch/segment-anything.git
# You must download the SAM model checkpoint (e.g., sam_vit_h_4b8939.pth) and set SAM_WEIGHTS to its path.
# Running SAM requires a GPU for reasonable speed.
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
from PIL import Image
import numpy as np
SAM_WEIGHTS = "/path/to/sam_vit_h_4b8939.pth" # <-- download and place
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_h" # change to the weights type you downloaded
sam = sam_model_registry[model_type](checkpoint=SAM_WEIGHTS)
sam.to(device=DEVICE)
# Example automatic mask generation for the after image
image_pil = Image.open("after.png").convert("RGB")
image_np = np.array(image_pil)
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95)
masks = mask_generator.generate(image_np) # list of mask dicts
# Save masks individually as pngs or compose an indexed mask.
'''
nb.cells.append(new_code_cell(sam_snippet))
# Add the processing code (simplified) as runnable cells
proc_code = f"""# Alignment and proxy computation (executed on server)\n\nfrom PIL import Image\nimg_before = Image.open(r'{before_fp}').convert('RGB')\nimg_after = Image.open(r'{after_fp}').convert('RGB')\nprint('Images loaded: sizes', img_before.size, img_after.size)\n\n# The detailed code is saved in the pipeline outputs. Please run the full pipeline script for adjustments.\n"""
nb.cells.append(new_code_cell(proc_code))
# Add note about thresholds and tuning
nb.cells.append(new_markdown_cell("## Threshold tuning\nSuggested starting thresholds for RGB-only imagery (tweak per-scene):\n\n- EXG vegetation loss threshold: 20\n- G-R water candidate threshold: 10\n- Brightness threshold for water detection: 100\n\nUse the visual outputs (`montage.png`, `overlay_veg_loss.png`, `overlay_water_candidate.png`) to iteratively adjust these values."))
# save notebook
nb_path = "/data/flood_pipeline_notebook.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
# report outputs
out = {
    "outputs_folder": out_dir,
    "files_written": os.listdir(out_dir),
    "summary": summary,
    "notebook": nb_path,
    "note": "SAM inference snippet included in notebook. Actual SAM inference not run here because model weights are not present in the environment."
}
print(json.dumps(out, indent=2))
# Retry: smaller ECC iterations and direct processing; will produce outputs in /data/flood_pipeline_outputs
import os, cv2, numpy as np, imageio
from skimage import exposure
import matplotlib.pyplot as plt
plt.ioff()
before_fp = "before.png"
after_fp = "after.png"
out_dir = "flood_pipeline_outputs"
os.makedirs(out_dir, exist_ok=True)
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
def histogram_match(src, ref):
    matched = np.zeros_like(src, dtype=np.uint8)
    for ch in range(3):
        matched[..., ch] = exposure.match_histograms(src[..., ch], ref[..., ch])
    return matched
def align_images_ecc(ref_img, moving_img, warp_mode=cv2.MOTION_AFFINE, num_iter=500, eps=1e-4):
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    mov_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, eps)
    try:
        cc, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria, None, 1)
    except cv2.error as e:
        print("ECC failed:", e)
        return moving_img, None
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(moving_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(moving_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, warp_matrix
img_before = read_rgb(before_fp)
img_after = read_rgb(after_fp)
aligned_after, warp = align_images_ecc(img_before, img_after, warp_mode=cv2.MOTION_AFFINE, num_iter=800, eps=1e-5)
if warp is None:
    aligned_after = img_after.copy()
aligned_after = histogram_match(aligned_after, img_before)
def compute_exg(img):
    R = img[...,0].astype(np.float32)
    G = img[...,1].astype(np.float32)
    B = img[...,2].astype(np.float32)
    return 2*G - R - B
def compute_g_minus_r(img):
    return img[...,1].astype(np.float32) - img[...,0].astype(np.float32)
def brightness(img):
    return img.mean(axis=2).astype(np.float32)
exg_before = compute_exg(img_before)
exg_after = compute_exg(aligned_after)
gmr_before = compute_g_minus_r(img_before)
gmr_after = compute_g_minus_r(aligned_after)
bright_before = brightness(img_before)
bright_after = brightness(aligned_after)
exg_diff = exg_before - exg_after
gmr_diff = gmr_after - gmr_before
EXG_LOSS_T = 20.0
GMR_WATER_T = 10.0
BRIGHT_DARK = 100.0
veg_loss_mask = (exg_diff > EXG_LOSS_T)
water_candidate_mask = ((gmr_after > GMR_WATER_T) & (bright_after < BRIGHT_DARK))
def overlay_mask_on_rgb(rgb, mask, color=(255,0,0), alpha=0.35):
    overlay = rgb.copy().astype(np.float32)
    overlay[mask] = overlay[mask]*(1-alpha) + np.array(color)*(alpha)
    return np.clip(overlay,0,255).astype(np.uint8)
overlay_veg_loss = overlay_mask_on_rgb(aligned_after, veg_loss_mask, color=(255,0,0), alpha=0.35)
overlay_water = overlay_mask_on_rgb(aligned_after, water_candidate_mask, color=(0,0,255), alpha=0.35)
imageio.imwrite(os.path.join(out_dir, "before_rgb.png"), img_before)
imageio.imwrite(os.path.join(out_dir, "after_rgb_aligned.png"), aligned_after)
imageio.imwrite(os.path.join(out_dir, "overlay_veg_loss.png"), overlay_veg_loss)
imageio.imwrite(os.path.join(out_dir, "overlay_water_candidate.png"), overlay_water)
# simple stats
h,w = img_before.shape[:2]
summary = {
    "total_pixels": int(h*w),
    "veg_loss_pixels": int(veg_loss_mask.sum()),
    "water_candidate_pixels": int(water_candidate_mask.sum()),
    "veg_loss_pct": float(veg_loss_mask.sum())/float(h*w)*100.0,
    "water_candidate_pct": float(water_candidate_mask.sum())/float(h*w)*100.0
}
import pandas as pd
df = pd.DataFrame([summary])
csv_out = os.path.join(out_dir, "pixel_change_summary.csv")
df.to_csv(csv_out, index=False)
# montage
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1,3, figsize=(15,5))
axes[0].imshow(img_before); axes[0].set_title("Before"); axes[0].axis('off')
axes[1].imshow(aligned_after); axes[1].set_title("After aligned"); axes[1].axis('off')
axes[2].imshow(overlay_veg_loss); axes[2].set_title("Veg loss (red)"); axes[2].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "quick_montage.png"), dpi=150)
plt.close()
print("Wrote outputs to:", out_dir)
print(summary)
# ------
# Try again with downsampling to reduce compute time.
import os, cv2, numpy as np, imageio
from skimage import exposure
import matplotlib.pyplot as plt
plt.ioff()
before_fp = "D:/Flood Damage Identification and Quantification Mobile Application/Backend/data/before.png"
after_fp = "D:/Flood Damage Identification and Quantification Mobile Application/Backend/data/after.png"


out_dir = "D:/Flood Damage Identification and Quantification Mobile Application/Backend/data/flood_pipeline_outputs"
os.makedirs(out_dir, exist_ok=True)
def read_rgb_small(path, max_dim=1000):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h,w = img.shape[:2]
    scale = 1.0
    if max(h,w) > max_dim:
        scale = max_dim/float(max(h,w))
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale
def histogram_match(src, ref):
    matched = np.zeros_like(src, dtype=np.uint8)
    for ch in range(3):
        matched[..., ch] = exposure.match_histograms(src[..., ch], ref[..., ch])
    return matched
def align_images_ecc(ref_img, moving_img, warp_mode=cv2.MOTION_AFFINE, num_iter=200, eps=1e-3):
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    mov_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3,3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2,3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, eps)
    try:
        cc, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria, None, 1)
    except cv2.error as e:
        print("ECC failed:", e)
        return moving_img, None
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(moving_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(moving_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, warp_matrix
img_before, scale_b = read_rgb_small(before_fp, max_dim=900)
img_after, scale_a = read_rgb_small(after_fp, max_dim=900)
print("loaded shapes:", img_before.shape, img_after.shape, "scales:", scale_b, scale_a)
aligned_after, warp = align_images_ecc(img_before, img_after, warp_mode=cv2.MOTION_AFFINE, num_iter=200, eps=1e-3)
if warp is None:
    aligned_after = img_after.copy()
aligned_after = histogram_match(aligned_after, img_before)
def compute_exg(img):
    R = img[...,0].astype(np.float32)
    G = img[...,1].astype(np.float32)
    B = img[...,2].astype(np.float32)
    return 2*G - R - B
def compute_g_minus_r(img):
    return img[...,1].astype(np.float32) - img[...,0].astype(np.float32)
def brightness(img):
    return img.mean(axis=2).astype(np.float32)
exg_before = compute_exg(img_before)
exg_after = compute_exg(aligned_after)
gmr_before = compute_g_minus_r(img_before)
gmr_after = compute_g_minus_r(aligned_after)
bright_after = brightness(aligned_after)
exg_diff = exg_before - exg_after
gmr_diff = gmr_after - gmr_before
EXG_LOSS_T = 20.0
GMR_WATER_T = 10.0
BRIGHT_DARK = 100.0
veg_loss_mask = (exg_diff > EXG_LOSS_T)
water_candidate_mask = ((gmr_after > GMR_WATER_T) & (bright_after < BRIGHT_DARK))
def overlay_mask_on_rgb(rgb, mask, color=(255,0,0), alpha=0.35):
    overlay = rgb.copy().astype(np.float32)
    overlay[mask] = overlay[mask]*(1-alpha) + np.array(color)*(alpha)
    return np.clip(overlay,0,255).astype(np.uint8)
overlay_veg_loss = overlay_mask_on_rgb(aligned_after, veg_loss_mask, color=(255,0,0), alpha=0.35)
overlay_water = overlay_mask_on_rgb(aligned_after, water_candidate_mask, color=(0,0,255), alpha=0.35)
import imageio
imageio.imwrite(os.path.join(out_dir, "before_rgb_small.png"), img_before)
imageio.imwrite(os.path.join(out_dir, "after_rgb_aligned_small.png"), aligned_after)
imageio.imwrite(os.path.join(out_dir, "overlay_veg_loss_small.png"), overlay_veg_loss)
imageio.imwrite(os.path.join(out_dir, "overlay_water_candidate_small.png"), overlay_water)
h,w = img_before.shape[:2]
summary = {
    "total_pixels": int(h*w),
    "veg_loss_pixels": int(veg_loss_mask.sum()),
    "water_candidate_pixels": int(water_candidate_mask.sum()),
    "veg_loss_pct": float(veg_loss_mask.sum())/float(h*w)*100.0,
    "water_candidate_pct": float(water_candidate_mask.sum())/float(h*w)*100.0
}
import pandas as pd
df = pd.DataFrame([summary])
csv_out = os.path.join(out_dir, "pixel_change_summary_small.csv")
df.to_csv(csv_out, index=False)
print("saved outputs to", out_dir)
print(summary)
# STDOUT/STDERR
# loaded shapes: (465, 900, 3) (492, 900, 3) scales: 0.7765314926660914 0.7785467128027682
# saved outputs to /data/flood_pipeline_outputs
# {'total_pixels': 418500, 'veg_loss_pixels': 125906, 'water_candidate_pixels': 254030}
# Create a small Jupyter notebook with SAM snippet and notes, save to /data
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
nb = new_notebook()
nb.cells = []
nb.cells.append(new_markdown_cell("# Flood change pipeline notebook\nThis notebook contains:\n- SAM inference snippet (requires weights and GPU)\n- Notes on thresholds and how to run the provided pipeline script\n- Links to output images produced by the automated run in `/data/flood_pipeline_outputs/`\n"))
sam_snippet = r"""# SAM inference snippet (requires model weights)
# pip install git+https://github.com/facebookresearch/segment-anything.git
# Download SAM checkpoint (e.g. sam_vit_h_4b8939.pth) and set SAM_WEIGHTS path.
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch
SAM_WEIGHTS = "/path/to/sam_vit_h_4b8939.pth" # update
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=SAM_WEIGHTS)
sam.to(device=DEVICE)
# Generate masks:
# image_np = np.array(Image.open('after.png').convert('RGB'))
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image_np)
# Save masks accordingly (one PNG per mask or an indexed mask)
"""
nb.cells.append(new_code_cell(sam_snippet))
nb.cells.append(new_markdown_cell("## Files produced by automated run\n- /data/flood_pipeline_outputs/before_rgb_small.png\n- /data/flood_pipeline_outputs/after_rgb_aligned_small.png\n- /data/flood_pipeline_outputs/overlay_veg_loss_small.png\n- /data/flood_pipeline_outputs/overlay_water_candidate_small.png\n- /data/flood_pipeline_outputs/pixel_change_summary_small.csv\n\nAdjust thresholds in the notebook and re-run the pipeline cells to tune for your scenes."))
nb_path = "/data/flood_pipeline_notebook.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print("Notebook written to", nb_path)
# STDOUT/STDERR
# Notebook written to /data/flood_pipeline_notebook.ipynb