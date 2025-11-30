# Run an analysis pipeline on the two provided RGB images. 
# - Align images (ECC) 
# - Histogram match 
# - Compute ExG and G-R proxies 
# - Compute difference maps and simple change masks 
# - Save visual outputs and a CSV summary of candidate affected areas (pixel counts) # - Create a Jupyter notebook (.ipynb) that includes the SAM inference snippet (non-runnable here) 
# and the steps we executed, for reproducibility. 
 
import glob
import os, json, nbformat, math 
import numpy as np, cv2 
from skimage import exposure 
import matplotlib.pyplot as plt 
import pandas as pd
import imageio 
import random

plt.ioff() 



# np.random.seed(3)

# def build_sam_overlay(image_np, masks, alpha=0.4):
#     overlay = image_np.copy()

#     for ann in masks:
#         m = ann["segmentation"].astype(np.uint8)

#         # Random color
#         color = np.array([
#             random.randint(0, 255),
#             random.randint(0, 255),
#             random.randint(0, 255)
#         ], dtype=np.uint8)

#         # Apply transparent fill
#         overlay[m == 1] = (
#             (1 - alpha) * overlay[m == 1] + alpha * color
#         ).astype(np.uint8)

#         # Draw border with the same color as fill
#         contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(overlay, contours, -1, color.tolist(), 2)

#     return overlay


# overlay_before = build_sam_overlay(image_np_before, masks_before)
# imageio.imwrite(os.path.join(out_dir, "before_sam_overlay.png"), overlay_before)

# overlay_after = build_sam_overlay(image_np_after, masks_after)
# imageio.imwrite(os.path.join(out_dir, "after_sam_overlay.png"), overlay_after)




from rasterio.features import shapes 
from shapely.geometry import shape, mapping, Polygon 
import geopandas as gpd 
from scipy.optimize import linear_sum_assignment
from rasterio.transform import from_origin


def read_rgb_image(path):
    """Read an RGB image into numpy float32 array (H,W,3) scaled 0-255.""" 
    img = cv2.imread(os.path, cv2.IMREAD_COLOR) 
    if img is None: 
        raise FileNotFoundError(f"Image not found: {os.path}") 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    return img.astype(np.uint8) 


def load_binary_masks_from_folder(folder, img_shape): 
    """ 
    Load binary masks (png) from a folder. Return labeled mask (H,W) where each object has a 
    unique integer id, 
    and a dict mapping id -> filename. 
    """ 
    if not os.path.isdir(folder): 
        raise FileNotFoundError(f"Mask folder not found: {folder}") 
    mask_files = sorted(glob.glob(os.path.join(folder, "*.png"))) 
    
    if len(mask_files) == 0: 
        raise FileNotFoundError(f"No PNG masks found in {folder}") 
    accum = np.zeros(img_shape[:2], dtype=np.int32) 
    id_map = {} 
    current_id = 1 
    for mf in mask_files: 
        m = cv2.imread(mf, cv2.IMREAD_UNCHANGED) 
        if m is None: 
            continue 
        # convert to binary if multi-channel or colored 
        if m.ndim == 3: 
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) 
        _, bw = cv2.threshold(m, 127, 1, cv2.THRESH_BINARY) 
        if bw.sum() < MIN_MASK_AREA_PIXELS: 
            continue 
        # avoid overlaps: only assign pixels not yet labeled 
        new_pixels = (bw == 1) & (accum == 0) 
        accum[new_pixels] = current_id 
        id_map[current_id] = os.path.basename(mf) 
        current_id += 1 
    return accum, id_map 

def load_indexed_mask(path): 
    """Load an indexed/color-coded mask where unique integer values represent instance ids.""" 
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED) 
    if m is None: 
        raise FileNotFoundError(path) 
    if m.ndim == 3: 
        # convert rgb colors to single label via unique color mapping 
        # pack colors into ints 
        packed = (m[...,0].astype(np.uint32) << 16) | (m[...,1].astype(np.uint32) << 8) | (m[...,2].astype(np.uint32)) 
        unique_vals = np.unique(packed) 
        label_img = np.zeros(packed.shape, dtype=np.int32) 
        idx = 1 
        for v in unique_vals: 
            if v == 0:  # treat black as background 
                continue 
            label_img[packed == v] = idx 
            idx += 1 
        return label_img 
    else: 
        return m.astype(np.int32) 
    
 

def raster_labels_to_polygons(label_img, transform=None, crs=None): 
    """ 
    Convert label raster to GeoDataFrame of polygons with properties: 
    'id', 'area_pixels' 
    """ 
    records = [] 
    labels = np.unique(label_img) 
    labels = labels[labels != 0] 
    for lab in labels: 
        mask = (label_img == lab).astype(np.uint8) 
        # optionally remove tiny bits 
        if mask.sum() < MIN_MASK_AREA_PIXELS: 
            continue 
        # use rasterio.features.shapes to extract polygons 
        for geom, val in shapes(mask, mask=mask.astype(np.uint8), transform=transform): 
            poly = shape(geom) 
            if poly.area == 0: 
                continue 
            records.append({'id': int(lab), 'geometry': poly, 'area_pixels': int(mask.sum())}) 
            break  # one polygon per label (first) 
    if len(records) == 0: 
        return gpd.GeoDataFrame(columns=['id','geometry','area_pixels'], geometry='geometry', crs=crs) 
    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=crs) 
    return gdf 

def compute_rgb_proxies(img, mask): 
    """ 
    Compute simple RGB-based proxies for vegetation and water per-mask: - ExG = 2G - R - B (excess green) -> vegetation proxy - G_minus_R = G - R -> water proxy when combined with darkness 
    Returns mean_exg, mean_g_minus_r, mean_brightness 
    """ 
    R = img[...,0].astype(np.float32) 
    G = img[...,1].astype(np.float32) 
    B = img[...,2].astype(np.float32) 
    exg = 2*G - R - B 
    g_minus_r = G - R 
    brightness = (R + G + B)/3.0 
    masked = mask.astype(bool) 
    if masked.sum() == 0: 
        return {'exg': 0.0, 'g_r': 0.0, 'brightness': 0.0} 
    return { 
    'exg': float(np.mean(exg[masked])), 
    'g_r': float(np.mean(g_minus_r[masked])), 
    'brightness': float(np.mean(brightness[masked])) 
    } 


def polygon_iou(poly1, poly2): 
    """Compute IoU for shapely polygons. If disjoint, IoU=0.""" 
    if not poly1.is_valid or not poly2.is_valid: 
        return 0.0 
    inter = poly1.intersection(poly2).area 
    union = poly1.union(poly2).area 
    if union == 0: 
        return 0.0 
    return inter/union


SAM_MASKS_BEFORE_DIR = "Backend/data/sam_masks_before"   # folder with binary PNG masks for 'before' 
SAM_MASKS_AFTER_DIR  = "Backend/data/sam_masks_after"    # folder with binary PNG masks for 'after' 
# Alternatively, if you have a single indexed mask: set SAM_INDEXED_BEFORE = "before_indexed_mask.png", etc. 
SAM_INDEXED_BEFORE = None 
SAM_INDEXED_AFTER = None 


# Spatial thresholds 
MIN_MASK_AREA_PIXELS = 50    # drop tiny masks (noise) 
IOU_MATCH_THRESHOLD = 0.2    # IoU threshold below which it's considered new/deleted 
IOU_SAME_OBJECT = 0.5        # IoU >= this => same object 

# Output 
OUT_SHP = "Backend/data/objects_change_report.geojson" 
OUT_CSV = "Backend/data/objects_change_report.csv" 


# Semantic thresholds (RGB proxies; adjust for your imagery) 
EXG_VEG_THRESHOLD = 20.0     # Excess-G > threshold -> vegetation (units: scaled 0-255) 
WATER_G_R_DIFF = 10.0        # G - R > threshold & brightness low -> water proxy 
DARKNESS_BRIGHTNESS = 90.0   # mean RGB < threshold considered 'dark' (water candidate)


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


# Save images for visualization 
def save_vis(img, path): 
    imageio.imwrite(path, img.astype(np.uint8)) 

# create viz overlays 
def overlay_mask_on_rgb(rgb, mask, color=(255,0,0), alpha=0.4):     
    overlay = rgb.copy().astype(np.float32) 
    overlay[mask] = overlay[mask]*(1-alpha) + np.array(color)*(alpha)     
    return np.clip(overlay,0,255).astype(np.uint8)


def pipeline():
     
    # file paths 
    before_fp = "Backend/data/before.png" 
    after_fp  = "Backend/data/after.png" 
    out_dir = "Backend/data/flood_pipeline_outputs" 

    os.makedirs(out_dir, exist_ok=True) 
    
    
    
    # read 
    img_before = read_rgb(before_fp) 
    img_after = read_rgb(after_fp) 
    
    # align after to before 
    aligned_after, warp = align_images_ecc(img_before, img_after, warp_mode=cv2.MOTION_AFFINE, num_iter=3000,eps=1e-6) 
    aligned_after = histogram_match(aligned_after, img_before) 
    
     
    
    exg_before = compute_exg(img_before) 
    exg_after  = compute_exg(aligned_after) 
    gmr_before = compute_g_minus_r(img_before) 
    gmr_after  = compute_g_minus_r(aligned_after) 
    bright_before = brightness(img_before) 
    bright_after  = brightness(aligned_after) 
    
    # difference maps 
    exg_diff = exg_before - exg_after  # positive -> vegetation loss gmr_diff = gmr_after - gmr_before  # positive -> greener (or water if dark) bright_diff = bright_after - bright_before 
    
    # simple masks with suggested thresholds (tweakable) 
    EXG_LOSS_T = 20.0   # ExG drop > threshold -> veg loss candidate 
    GMR_WATER_T = 10.0  # G-R increase > threshold 
    BRIGHT_DARK = 100.0 # brightness less than this suggests water 

    veg_loss_mask = (exg_diff > EXG_LOSS_T) 
    water_candidate_mask = ((gmr_after > GMR_WATER_T) & (bright_after < BRIGHT_DARK)) 
    
 
    
    overlay_veg_loss = overlay_mask_on_rgb(aligned_after, veg_loss_mask, color=(255,0,0), alpha=0.35) 
    overlay_water = overlay_mask_on_rgb(aligned_after, water_candidate_mask, color=(0,0,255), alpha=0.35) 
    
    save_vis(img_before, os.path.join(out_dir, "before_rgb.png")) 
    save_vis(aligned_after, os.path.join(out_dir, "after_rgb_aligned_matched.png")) 
    save_vis((255*( (exg_before - exg_before.min())/(exg_before.max()-exg_before.min()+1e-9) )).astype(np.uint8), os.path.join(out_dir, "exg_before_norm.png")) 
    save_vis((255*( (exg_after - exg_after.min())/(exg_after.max()-exg_after.min()+1e-9) )).astype(np.uint8), os.path.join(out_dir, "exg_after_norm.png")) 
    save_vis((255*( (exg_diff - exg_diff.min())/(exg_diff.max()-exg_diff.min()+1e-9))).astype(np.uint8), os.path.join(out_dir, "exg_diff_norm.png")) 
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
    
    df = pd.DataFrame([summary]) 
    csv_out = os.path.join(out_dir, "pixel_change_summary.csv") 
    df.to_csv(csv_out, index=False) 
    
    # create a simple PNG montage for quick view 
    fig, axes = plt.subplots(2,3, figsize=(12,8)) 
    axes = axes.flatten() 
    axes[0].imshow(img_before); 
    axes[0].set_title("Before RGB"); 
    axes[0].axis("off") 
    axes[1].imshow(aligned_after); 
    axes[1].set_title("After RGB (aligned & matched)"); 
    axes[1].axis("off") 
    axes[2].imshow(overlay_veg_loss); 
    axes[2].set_title("Veg loss candidates (red)"); 
    axes[2].axis("off") 
    axes[3].imshow(overlay_water); 
    axes[3].set_title("Water candidates (blue)"); 
    axes[3].axis("off") 
    axes[4].imshow((exg_diff - exg_diff.min())/(exg_diff.max()-exg_diff.min()+1e-9)); 
    axes[4].set_title("ExG diff (norm)"); 
    axes[4].axis("off") 
    axes[5].axis("off") 
    montage_fp = os.path.join(out_dir, "montage.png") 
    plt.tight_layout(); 
    plt.savefig(montage_fp, dpi=150); 
    plt.close() 
    

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor 
    import torch 
    from PIL import Image 
    
    
    SAM_WEIGHTS = "Backend/sam_vit_h_4b8939.pth"  # <-- download and place 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
    print("Using device:", DEVICE)
    model_type = "vit_h"  # change to the weights type you downloaded 
    sam = sam_model_registry[model_type](checkpoint=SAM_WEIGHTS) 
    sam.to(device=DEVICE) 
    
    #automatic mask generation for the after aligned image 
    image_np_after = np.array(Image.open(after_fp).convert("RGB") ) 
    mask_generator_after = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95) 
    masks_after = mask_generator_after.generate(image_np_after)  # list of mask dicts 


    # Save masks individually as pngs or compose an indexed mask. 
    print(json.dumps(out_dir, indent=2)) 

    #automatic mask generation for the before image 
    image_np_before = np.array(Image.open(before_fp).convert("RGB") ) 
    mask_generator_before = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95) 
    masks_before = mask_generator_before.generate(image_np_before)  # list of mask dicts 
    # Save masks individually as pngs or compose an indexed mask. 
    print(json.dumps(out_dir, indent=2)) 

    fliped_masks_before = []
    for mask_dict in masks_before:
        flipped_mask = cv2.flip(mask_dict['segmentation'].astype(np.uint8) * 255, 0)  # vertical flip
        fliped_masks_before.append(flipped_mask)


    warped_masks_after = []
    for mask_dict in masks_after:
        mask = mask_dict['segmentation'].astype(np.uint8) * 255
        warped_mask = cv2.warpAffine(mask, warp, (image_np_before.shape[1], image_np_before.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        warped_mask = cv2.flip(warped_mask, 0)  # vertical flip
        warped_masks_after.append(warped_mask)

    # Output directories
    out_dir_after = "Backend/data/sam_masks_after"
    out_dir_before = "Backend/data/sam_masks_before"
    os.makedirs(out_dir_after, exist_ok=True)
    os.makedirs(out_dir_before, exist_ok=True)


    # Function to save masks that are already numpy arrays
    def save_masks(masks, out_dir):
        for i, mask in enumerate(masks):
            mask_fp = os.path.join(out_dir, f"mask_{i:03d}.png")
            imageio.imwrite(mask_fp, mask)

    # Save warped after masks
    save_masks(warped_masks_after, out_dir_after)
    print(f"Saved {len(warped_masks_after)} masks to {out_dir_after}")

    # Save before masks
    save_masks(fliped_masks_before, out_dir_before)
    print(f"Saved {len(masks_before)} masks to {out_dir_before}")



    # 3) Load SAM masks (either folders of binary masks or indexed) 
    print("Loading SAM masks...") 
    if SAM_INDEXED_BEFORE: 
        labels_before = load_indexed_mask(SAM_INDEXED_BEFORE) 
    else: 
        labels_before, idmap_before = load_binary_masks_from_folder(SAM_MASKS_BEFORE_DIR, img_before.shape) 
    if SAM_INDEXED_AFTER: 
        labels_after = load_indexed_mask(SAM_INDEXED_AFTER) 
    else: 
    # Note: after masks must be warped into the 'before' frame to match images if SAM was created in after image coords 
    # If your SAM masks are already aligned to the images and images used above were aligned accordingly, proceed: 
        labels_after, idmap_after = load_binary_masks_from_folder(SAM_MASKS_AFTER_DIR, img_before.shape)

    # 4) Raster labels -> polygons (no geotransform in simple image coordinates) 
    print("Vectorizing masks...") 

    # Suppose labels_before and labels_after are NumPy arrays
    height_before, width_before = labels_before.shape
    height_after, width_after   = labels_after.shape

    # Create dummy transforms for both
    transform_before = from_origin(0, height_before, 1, 1)
    transform_after  = from_origin(0, height_after, 1, 1)

    # Convert to polygons
    gdf_before = raster_labels_to_polygons(labels_before, transform=transform_before, crs=None)
    gdf_after  = raster_labels_to_polygons(labels_after,  transform=transform_after,  crs=None)

    # 5) Compute spectral proxies for each polygon 
    print("Computing RGB proxies...") 
    def per_row_compute_proxies(gdf, img): 
        exg_list, gr_list, bright_list = [], [], [] 
        masks = [] 
        h,w = img.shape[:2] 
        for idx,row in gdf.iterrows(): 
            # Rasterize polygon to mask 
            rr = np.zeros((h,w), dtype=np.uint8) 
            # simple rasterization using cv2.fillPoly (convert to pixel coords) 
            coords = np.array(list(row.geometry.exterior.coords)).astype(np.int32) 
            try: 
                cv2.fillPoly(rr, [coords], 1) 
            except Exception: 
                # fallback: bounding box mask 
                minx, miny, maxx, maxy = row.geometry.bounds 
                rr[int(miny):int(maxy)+1, int(minx):int(maxx)+1] = 1 
            proxies = compute_rgb_proxies(img, rr) 
            exg_list.append(proxies['exg']) 
            gr_list.append(proxies['g_r']) 
            bright_list.append(proxies['brightness']) 
            masks.append(rr) 
        gdf = gdf.copy() 
        gdf['exg'] = exg_list 
        gdf['g_r'] = gr_list 
        gdf['brightness'] = bright_list 
        gdf['mask_pixels'] = [int(m.sum()) for m in masks] 
        return gdf
     
    gdf_before = per_row_compute_proxies(gdf_before, img_before) 
    gdf_after  = per_row_compute_proxies(gdf_after, aligned_after) 

    def save_image_with_polygons(img, gdf, out_fp):
                """Save image with random-colored polygon fills using NumPy."""
                height, width = img.shape[:2]

                fig = plt.figure(frameon=False)
                fig.set_size_inches(width / 100.0, height / 100.0)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)

                ax.imshow(img)

                if len(gdf) > 0:
                    # NumPy: N polygons × 4 RGBA values
                    # Random colors, but consistent alpha = 0.4
                    facecolors = np.hstack([
                        np.random.rand(len(gdf), 3),        # RGB
                        np.full((len(gdf), 1), 0.4)         # Alpha
                    ])

                    gdf.plot(
                        ax=ax,
                        facecolor=facecolors,
                        edgecolor='black',
                        linewidth=0.8
                    )

                fig.savefig(out_fp, dpi=100)
                plt.close(fig)

    # Usage:
    save_image_with_polygons(img_before, gdf_before, "Backend/data/before_with_polygons.png")
    save_image_with_polygons(aligned_after, gdf_after, "Backend/data/after_with_polygons.png")

    # 6) Semantic labeling (simple rules) 
    def semantic_label(row): 
        if row['exg'] >= EXG_VEG_THRESHOLD: 
            return "VEGETATION" 
        if (row['g_r'] >= WATER_G_R_DIFF) and (row['brightness'] <= DARKNESS_BRIGHTNESS): 
            return "WATER" 
        # fallback: dark + low exg might be water, bright & low exg -> BARE_SOIL 
        if row['brightness'] < DARKNESS_BRIGHTNESS: 
            return "POSSIBLE_WATER" 
        return "BARE_SOIL" 

    gdf_before['class'] = gdf_before.apply(semantic_label, axis=1) 
    gdf_after['class'] = gdf_after.apply(semantic_label, axis=1) 

    # 7) Compute IoU matrix between before & after 
    print("Computing IoU matrix and bipartite matching...") 
    nB = len(gdf_before) 
    nA = len(gdf_after) 
    iou_mat = np.zeros((nB, nA), dtype=np.float32) 
    for i,rb in gdf_before.iterrows(): 
        for j,ra in gdf_after.iterrows(): 
            iou_mat[i,j] = polygon_iou(rb.geometry, ra.geometry) 
    # Hungarian maximum matching: but we want to maximize IoU -> minimize negative IoU 
    row_ind, col_ind = linear_sum_assignment(-iou_mat) 

    # 8) Compile results 
    results = [] 
    matched_after_idx = set() 
    for r,c in zip(row_ind, col_ind): 
        iou = float(iou_mat[r,c]) 
        before = gdf_before.iloc[r] 
        after = gdf_after.iloc[c] 
        matched_after_idx.add(c) 
        same_obj = (iou >= IOU_SAME_OBJECT) 
        status = "matched_same" if same_obj else ("possible_change" if iou >= 
        IOU_MATCH_THRESHOLD else "weak_match") 
        # compute area change in pixels (we stored mask_pixels) 
        area_before = before['mask_pixels'] 
        area_after = after['mask_pixels'] 
        pct_change = (area_after - area_before)/area_before if area_before > 0 else None 
        # determine semantic change label 
        sem_change = f"{before['class']} -> {after['class']}" 
        results.append({ 
            'before_id': int(before['id']), 
            'after_id': int(after['id']), 
            'iou': iou, 
            'status': status, 
            'area_before_px': int(area_before), 
            'area_after_px': int(area_after), 
            'pct_change': float(pct_change) if pct_change is not None else None, 
            'semantic_change': sem_change, 
            'before_exg': float(before['exg']), 
            'after_exg': float(after['exg']), 
            'before_g_r': float(before['g_r']), 
            'after_g_r': float(after['g_r']) 
            }) 
        
    # any after objects not matched -> new objects 
    for j,ra in gdf_after.reset_index().iterrows(): 
        if j not in matched_after_idx: 
            results.append({ 
                'before_id': None, 
                'after_id': int(ra['id']), 
                'iou': 0.0, 
                'status': 'new_after', 
                'area_before_px': 0, 
                'area_after_px': int(ra['mask_pixels']), 
                'pct_change': None, 
                'semantic_change': f"NONE -> {ra['class']}", 
                'before_exg': None, 
                'after_exg': float(ra['exg']), 
                'before_g_r': None, 
                'after_g_r': float(ra['g_r']) 
                }) 
    # objects in before not matched to after (deleted) 
    matched_before_idx = set(row_ind)
    for i,rb in gdf_before.reset_index().iterrows(): 
        if i not in matched_before_idx: 
            results.append({ 
            'before_id': int(rb['id']), 
            'after_id': None, 
            'iou': 0.0, 
            'status': 'deleted_after', 
            'area_before_px': int(rb['mask_pixels']), 
            'area_after_px': 0, 
            'pct_change': None, 
            'semantic_change': f"{rb['class']} -> NONE", 
            'before_exg': float(rb['exg']), 
            'after_exg': None, 
            'before_g_r': float(rb['g_r']), 
            'after_g_r': None 
            }) 
    df = pd.DataFrame(results) 
    df.to_csv(OUT_CSV, index=False) 
    print(f"Wrote CSV report: {OUT_CSV}") 


    # 9) Optionally export GeoJSON with change attributes (join geometries) 
    # We'll include before geometry when available; else after geometry 
    features = [] 
    for row in results: 
        geom = None 
        if row['before_id'] is not None: 
            sel = gdf_before[gdf_before['id'] == row['before_id']] 
            if len(sel) > 0: 
                geom = sel.iloc[0].geometry 
        if geom is None and row['after_id'] is not None: 
            sel = gdf_after[gdf_after['id'] == row['after_id']] 
            if len(sel) > 0: 
                geom = sel.iloc[0].geometry 
        if geom is None: 
            continue 
        feat = { 
        'id': f"{row['before_id']}_{row['after_id']}", 
        'geometry': geom, 
        **row 
        } 
        features.append(feat) 
    if len(features) > 0: 
        out_gdf = gpd.GeoDataFrame(features, geometry='geometry') 
        out_gdf.to_file(OUT_SHP, driver='GeoJSON') 
        print(f"Wrote GeoJSON: {OUT_SHP}") 
        print("Done.")

    from skimage import measure
    
    def mask_to_polygons(binary_mask, min_area=50):
        polygons = []
        contours = measure.find_contours(binary_mask.astype(float), 0.5)

        for c in contours:
            poly = Polygon([(p[1], p[0]) for p in c])  # (x,y)
            if poly.area > min_area:
                polygons.append(poly)

        return gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
    
    gdf_veg_loss = mask_to_polygons(veg_loss_mask)
    gdf_water    = mask_to_polygons(water_candidate_mask)

    #gdf_sam = gpd.GeoDataFrame(masks_after, geometry=[Polygon(m['segments']) for m in masks_after])

    veg_loss_in_sam = gpd.overlay(gdf_after, gdf_veg_loss, how="intersection")
    water_in_sam    = gpd.overlay(gdf_after, gdf_water, how="intersection")

    def save_image_with_polygons(img, gdf_list, color_list, out_fp):
        height, width = img.shape[:2]
        fig = plt.figure(frameon=False)
        fig.set_size_inches(width / 100.0, height / 100.0)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(img)

        for gdf, color in zip(gdf_list, color_list):
            gdf.plot(ax=ax, facecolor='none', edgecolor=color, linewidth=1)

        fig.savefig(out_fp, dpi=100)
        plt.close(fig)
    
    save_image_with_polygons(
        aligned_after,
        [veg_loss_in_sam, water_in_sam],
        ["red", "blue"],
        "Backend/data/after_with_changes.png"
    )






# Only run pipeline if this file is executed directly, not when imported
if __name__ == "__main__":
    pipeline()



# # Retry: smaller ECC iterations and direct processing; will produce outputs in /mnt/data/flood_pipeline_outputs 

# plt.ioff() 

# img_before = read_rgb(before_fp) 
# img_after = read_rgb(after_fp) 
 
# aligned_after, warp = align_images_ecc(img_before, img_after, warp_mode=cv2.MOTION_AFFINE, num_iter=800, eps=1e-5) 
# if warp is None:     
#     aligned_after = img_after.copy() 
 
# aligned_after = histogram_match(aligned_after, img_before) 
 

# exg_before = compute_exg(img_before) 
# exg_after = compute_exg(aligned_after) 
# gmr_before = compute_g_minus_r(img_before) 
# gmr_after = compute_g_minus_r(aligned_after) 
# bright_before = brightness(img_before) 
# bright_after = brightness(aligned_after) 
 
# exg_diff = exg_before - exg_after 
# gmr_diff = gmr_after - gmr_before 
 
# EXG_LOSS_T = 20.0 
# GMR_WATER_T = 10.0 
# BRIGHT_DARK = 100.0 
 
# veg_loss_mask = (exg_diff > EXG_LOSS_T) 
# water_candidate_mask = ((gmr_after > GMR_WATER_T) & (bright_after < BRIGHT_DARK)) 
 
 
# overlay_veg_loss = overlay_mask_on_rgb(aligned_after, veg_loss_mask, color=(255,0,0), alpha=0.35) 
# overlay_water = overlay_mask_on_rgb(aligned_after, water_candidate_mask, color=(0,0,255), alpha=0.35) 
 
# imageio.imwrite(os.path.join(out_dir, "before_rgb.png"), img_before) 
# imageio.imwrite(os.path.join(out_dir, "after_rgb_aligned.png"), aligned_after) 
# imageio.imwrite(os.path.join(out_dir, "overlay_veg_loss.png"), overlay_veg_loss) 
# imageio.imwrite(os.path.join(out_dir, "overlay_water_candidate.png"), overlay_water) 
 
# # simple stats 
# h,w = img_before.shape[:2] 
# summary = { 
#     "total_pixels": int(h*w), 
#     "veg_loss_pixels": int(veg_loss_mask.sum()), 
#     "water_candidate_pixels": int(water_candidate_mask.sum()), 
#     "veg_loss_pct": float(veg_loss_mask.sum())/float(h*w)*100.0, 
#     "water_candidate_pct": float(water_candidate_mask.sum())/float(h*w)*100.0 
# } 


# df = pd.DataFrame([summary]) 
# csv_out = os.path.join(out_dir, "pixel_change_summary.csv") 
# df.to_csv(csv_out, index=False) 
 
# # montage 
 
# fig, axes = plt.subplots(1,3, figsize=(15,5)) 
# axes[0].imshow(img_before); axes[0].set_title("Before"); axes[0].axis('off') 
# axes[1].imshow(aligned_after); axes[1].set_title("After aligned"); axes[1].axis('off') 
# axes[2].imshow(overlay_veg_loss); axes[2].set_title("Veg loss (red)"); axes[2].axis('off') 
# plt.tight_layout() 
# plt.savefig(os.path.join(out_dir, "quick_montage.png"), dpi=150) 
# plt.close() 
 
# print("Wrote outputs to:", out_dir) 
# print(summary) 
 
 
# # Try again with downsampling to reduce compute time. 

# plt.ioff() 

# def read_rgb_small(path, max_dim=1000): 
#     img = cv2.imread(path, cv2.IMREAD_COLOR)     
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     
#     h,w = img.shape[:2]     
#     scale = 1.0     
#     if max(h,w) > max_dim:         
#         scale = max_dim/float(max(h,w)) 
#         img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)     
#         return img, scale 
 

 
# img_before, scale_b = read_rgb_small(before_fp, max_dim=900) 
# img_after, scale_a = read_rgb_small(after_fp, max_dim=900) 
# print("loaded shapes:", img_before.shape, img_after.shape, "scales:", scale_b, scale_a) 
 
# aligned_after, warp = align_images_ecc(img_before, img_after, warp_mode=cv2.MOTION_AFFINE, num_iter=200, eps=1e-3) 
# if warp is None:     
#     aligned_after = img_after.copy() 
# aligned_after = histogram_match(aligned_after, img_before) 
 

 
# exg_before = compute_exg(img_before) 
# exg_after = compute_exg(aligned_after) 
# gmr_before = compute_g_minus_r(img_before) 
# gmr_after = compute_g_minus_r(aligned_after) 
# bright_after = brightness(aligned_after) 
 
# exg_diff = exg_before - exg_after 
# gmr_diff = gmr_after - gmr_before 
 
# EXG_LOSS_T = 20.0 
# GMR_WATER_T = 10.0 
# BRIGHT_DARK = 100.0 
 
# veg_loss_mask = (exg_diff > EXG_LOSS_T) 
# water_candidate_mask = ((gmr_after > GMR_WATER_T) & (bright_after < BRIGHT_DARK)) 

 
# overlay_veg_loss = overlay_mask_on_rgb(aligned_after, veg_loss_mask, color=(255,0,0), alpha=0.35) 
# overlay_water = overlay_mask_on_rgb(aligned_after, water_candidate_mask, color=(0,0,255), alpha=0.35) 


# imageio.imwrite(os.path.join(out_dir, "before_rgb_small.png"), img_before) 
# imageio.imwrite(os.path.join(out_dir, "after_rgb_aligned_small.png"), aligned_after) 
# imageio.imwrite(os.path.join(out_dir, "overlay_veg_loss_small.png"), overlay_veg_loss) 
# imageio.imwrite(os.path.join(out_dir, "overlay_water_candidate_small.png"), overlay_water) 
 
# h,w = img_before.shape[:2] 
# summary = { 
#     "total_pixels": int(h*w), 
#     "veg_loss_pixels": int(veg_loss_mask.sum()), 
#     "water_candidate_pixels": int(water_candidate_mask.sum()), 
#     "veg_loss_pct": float(veg_loss_mask.sum())/float(h*w)*100.0, 
#     "water_candidate_pct": float(water_candidate_mask.sum())/float(h*w)*100.0 
# } 


# df = pd.DataFrame([summary]) 
# csv_out = os.path.join(out_dir, "pixel_change_summary_small.csv") 
# df.to_csv(csv_out, index=False) 
 
# print("saved outputs to", out_dir) 
# print(summary) 
 
'''
STDOUT/STDERR loaded shapes: (465, 900, 3) (492, 900, 3) scales: 0.7765314926660914 
0.7785467128027682 
saved outputs to /mnt/data/flood_pipeline_outputs 
{'total_pixels': 418500, 'veg_loss_pixels': 125906, 'water_candidate_pixels': 
254030 
'''

