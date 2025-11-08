# Optional: install dependencies before running, e.g.:
# pip install opencv-python pillow numpy shapely rasterio geopandas scikit-image scipy pandas
# If you plan to run SAM inference later:
# pip install torch torchvision  # plus segment-anything package per your SAM install
#!/usr/bin/env python3
"""
flood_change_pipeline.py
Pipeline skeleton:
 - load RGB before/after images
 - co-register (ECC)
 - optional histogram matching
 - ingest SAM masks (binary PNG masks per-object OR a single indexed mask)
 - raster -> vector (shapely polygons)
 - compute per-mask spectral proxies (ExG for vegetation, G-R normalized for water proxy)
 - match masks across dates by IoU using Hungarian algorithm
 - classify semantic change rules and write CSV

Adjust thresholds in PARAMETERS section.
"""

import os
import glob
import numpy as np
import cv2
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping, Polygon
import geopandas as gpd
from scipy.optimize import linear_sum_assignment
import pandas as pd
from skimage import exposure
from skimage.measure import label
import warnings
# --------------------------
# PARAMETERS (tweak these)
# --------------------------
"""
Runtime-configurable inputs are now passed as parameters to main().
Defaults below are only used if main() is called without arguments.
"""
# --------------------------
# INPUT/CONFIG (defaults)
# --------------------------
BEFORE_IMG_DEFAULT = "before.png"
AFTER_IMG_DEFAULT = "after.png"
SAM_MASKS_BEFORE_DIR_DEFAULT = "sam_masks_before"  # folder with binary PNG masks for 'before' (relative)
SAM_MASKS_AFTER_DIR_DEFAULT = "sam_masks_after"   # folder with binary PNG masks for 'after' (relative)
# Alternatively, if you have a single indexed mask: set SAM_INDEXED_* to a path
SAM_INDEXED_BEFORE_DEFAULT = None
SAM_INDEXED_AFTER_DEFAULT = None
# Co-registration parameters
ECC_WARP_MODE = cv2.MOTION_AFFINE # cv2.MOTION_TRANSLATION, AFFINE, HOMOGRAPHY
ECC_NUM_ITER = 5000
ECC_TERMINATION_EPS = 1e-6
# Radiometric/histogram matching
DO_HIST_MATCH = True
# Semantic thresholds (RGB proxies; adjust for your imagery)
# Semantic thresholds (RGB proxies; adjust for your imagery)
EXG_VEG_THRESHOLD = 20.0  # Excess-G > threshold -> vegetation (units: scaled 0-255)
WATER_G_R_DIFF = 10.0  # G - R > threshold & brightness low -> water proxy
DARKNESS_BRIGHTNESS = 90.0  # mean RGB < threshold considered 'dark' (water candidate)
# Spatial thresholds
MIN_MASK_AREA_PIXELS = 500 # drop tiny masks (noise)
IOU_MATCH_THRESHOLD = 0.2 # IoU threshold below which it's considered new/deleted
IOU_SAME_OBJECT = 0.5 # IoU >= this => same object
# Output
OUT_SHP_NAME = "objects_change_report.geojson"
OUT_CSV_NAME = "objects_change_report.csv"

# --------------------------
# Utility functions
# --------------------------
def read_rgb_image(path):
	"""Read an RGB image into numpy uint8 array (H,W,3) scaled 0-255."""
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	if img is None:
		raise FileNotFoundError(f"Image not found: {path}")
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img.astype(np.uint8)
def histogram_match(src, ref):
	"""Match histogram of src to ref per-channel using skimage exposure."""
	matched = np.zeros_like(src, dtype=np.uint8)
	for ch in range(3):
		matched[..., ch] = exposure.match_histograms(src[..., ch], ref[..., ch])
	return matched
def align_images_ecc(ref_img, moving_img, warp_mode=ECC_WARP_MODE,
					 num_iter=ECC_NUM_ITER, eps=ECC_TERMINATION_EPS):
	"""
	Use OpenCV ECC to align moving_img to ref_img.
	Inputs: uint8 RGB arrays
	Returns aligned moving_img and the warp matrix
	"""
	# convert to gray
	ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
	mov_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)
	if warp_mode == cv2.MOTION_HOMOGRAPHY:
		warp_matrix = np.eye(3, 3, dtype=np.float32)
	else:
		warp_matrix = np.eye(2, 3, dtype=np.float32)
	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, eps)
	try:
		cc, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode,
											   criteria, None, 1)
	except cv2.error as e:
		warnings.warn("ECC alignment failed: " + str(e))
		return moving_img, warp_matrix  # return unaligned

	# warp the moving image to the ref frame
	if warp_mode == cv2.MOTION_HOMOGRAPHY:
		aligned = cv2.warpPerspective(moving_img, warp_matrix,
									  (ref_img.shape[1], ref_img.shape[0]),
									  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	else:
		aligned = cv2.warpAffine(moving_img, warp_matrix,
								 (ref_img.shape[1], ref_img.shape[0]),
								 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
	return aligned, warp_matrix

def load_binary_masks_from_folder(folder, img_shape):
	"""
	Load binary masks (png) from a folder. Return labeled mask (H,W) where each object has a
	unique integer id, and a dict mapping id -> filename.
	"""
	if not folder or not os.path.isdir(folder):
		# Graceful fallback: return empty mask image
		return np.zeros(img_shape[:2], dtype=np.int32), {}
	mask_files = sorted(glob.glob(os.path.join(folder, "*.png")))
	if len(mask_files) == 0:
		return np.zeros(img_shape[:2], dtype=np.int32), {}
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
		packed = ((m[..., 0].astype(np.uint32) << 16) |
				  (m[..., 1].astype(np.uint32) << 8) |
				  (m[..., 2].astype(np.uint32)))
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
	from affine import Affine
	if transform is None:
		transform = Affine.translation(0, 0) * Affine.scale(1, 1)
	

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
		return gpd.GeoDataFrame(columns=['id', 'geometry', 'area_pixels'], geometry='geometry', crs=crs)
	gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=crs)
	return gdf
def compute_rgb_proxies(img, mask):
	"""
	Compute simple RGB-based proxies for vegetation and water per-mask:
	- ExG = 2G - R - B (excess green) -> vegetation proxy
	- G_minus_R = G - R -> water proxy when combined with darkness
	Returns mean_exg, mean_g_minus_r, mean_brightness
	"""
	R = img[..., 0].astype(np.float32)
	G = img[..., 1].astype(np.float32)
	B = img[..., 2].astype(np.float32)
	exg = 2 * G - R - B
	g_minus_r = G - R
	brightness = (R + G + B) / 3.0
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
	return inter / union
# --------------------------
# Main pipeline
# --------------------------
def main(
	before_img_path: str | None = None,
	after_img_path: str | None = None,
	output_dir: str | None = None,
	sam_masks_before_dir: str | None = None,
	sam_masks_after_dir: str | None = None,
	sam_indexed_before: str | None = None,
	sam_indexed_after: str | None = None,
):
	"""
	Run the flood change pipeline.

	Parameters:
	- before_img_path: path to "before" RGB image
	- after_img_path: path to "after" RGB image
	- output_dir: directory to write outputs into (CSV/GeoJSON and any intermediates)
	- sam_masks_before_dir: directory containing binary PNG masks for before image
	- sam_masks_after_dir: directory containing binary PNG masks for after image
	- sam_indexed_before: path to single indexed mask for before (optional alternative to folder)
	- sam_indexed_after: path to single indexed mask for after (optional alternative to folder)

	Returns: dict with keys: csv_path, geojson_path
	"""
	# Resolve parameters against defaults
	before_img_path = before_img_path or BEFORE_IMG_DEFAULT
	after_img_path = after_img_path or AFTER_IMG_DEFAULT
	output_dir = output_dir or os.getcwd()
	os.makedirs(output_dir, exist_ok=True)

	# Resolve mask configs
	sam_masks_before_dir = sam_masks_before_dir or SAM_MASKS_BEFORE_DIR_DEFAULT
	sam_masks_after_dir = sam_masks_after_dir or SAM_MASKS_AFTER_DIR_DEFAULT
	sam_indexed_before = sam_indexed_before or SAM_INDEXED_BEFORE_DEFAULT
	sam_indexed_after = sam_indexed_after or SAM_INDEXED_AFTER_DEFAULT

	# Precompute output paths
	out_csv = os.path.join(output_dir, OUT_CSV_NAME)
	out_geojson = None

	# 1) Read images
	img_before = read_rgb_image(before_img_path)
	img_after = read_rgb_image(after_img_path)
	

	# 2) Align images (align 'after' to 'before')
	print("Aligning images (ECC)...")
	aligned_after, warp = align_images_ecc(img_before, img_after)
	if DO_HIST_MATCH:
		print("Histogram matching 'after' -> 'before'...")
		aligned_after = histogram_match(aligned_after, img_before)

	# 3) Load SAM masks (either folders of binary masks or indexed)
	print("Loading SAM masks...")
	if sam_indexed_before:
		labels_before = load_indexed_mask(sam_indexed_before)
		idmap_before = {}
	else:
		# If a relative folder is provided, interpret relative to current working dir
		masks_before_dir = sam_masks_before_dir
		if masks_before_dir and not os.path.isabs(masks_before_dir):
			masks_before_dir = os.path.abspath(masks_before_dir)
		labels_before, idmap_before = load_binary_masks_from_folder(masks_before_dir, img_before.shape)

	if sam_indexed_after:
		labels_after = load_indexed_mask(sam_indexed_after)
		idmap_after = {}
	else:
		# Note: after masks must be warped into the 'before' frame to match images if SAM was
		# created in after image coords
		# If your SAM masks are already aligned to the images and images used above were aligned
		# accordingly, proceed:
		masks_after_dir = sam_masks_after_dir
		if masks_after_dir and not os.path.isabs(masks_after_dir):
			masks_after_dir = os.path.abspath(masks_after_dir)
		# Load masks for 'after'
		labels_after_raw, idmap_after = load_binary_masks_from_folder(masks_after_dir, img_after.shape)

		# Warp the combined label mask to 'before' frame using ECC warp
		print("Warping 'after' masks to align with 'before' image...")
		if ECC_WARP_MODE == cv2.MOTION_HOMOGRAPHY:
			labels_after = cv2.warpPerspective(labels_after_raw.astype(np.uint8), warp,
											(img_before.shape[1], img_before.shape[0]),
											flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
		else:
			labels_after = cv2.warpAffine(labels_after_raw.astype(np.uint8), warp,
										(img_before.shape[1], img_before.shape[0]),
										flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)

	# 4) Raster labels -> polygons (no geotransform in simple image coordinates)
	print("Vectorizing masks...")
	gdf_before = raster_labels_to_polygons(labels_before, transform=None, crs=None)
	gdf_after = raster_labels_to_polygons(labels_after, transform=None, crs=None)

	# 5) Compute spectral proxies for each polygon
	print("Computing RGB proxies...")

	def per_row_compute_proxies(gdf, img):
		exg_list, gr_list, bright_list = [], [], []
		masks = []
		h, w = img.shape[:2]
		for idx, row in gdf.iterrows():
			# Rasterize polygon to mask
			rr = np.zeros((h, w), dtype=np.uint8)
			# simple rasterization using cv2.fillPoly (convert to pixel coords)
			try:
				coords = np.array(list(row.geometry.exterior.coords)).astype(np.int32)
				cv2.fillPoly(rr, [coords], 1)
			except Exception:
				# fallback: bounding box mask
				minx, miny, maxx, maxy = row.geometry.bounds
				rr[int(miny):int(maxy) + 1, int(minx):int(maxx) + 1] = 1
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
	gdf_after = per_row_compute_proxies(gdf_after, aligned_after)

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
	if nB == 0 or nA == 0:
		print("No objects found in one or both dates. Writing empty report.")
		# Write empty CSV with headers
		empty_cols = [
			'before_id','after_id','iou','status','area_before_px','area_after_px',
			'pct_change','semantic_change','before_exg','after_exg','before_g_r','after_g_r'
		]
		pd.DataFrame(columns=empty_cols).to_csv(out_csv, index=False)
		return {"csv_path": out_csv, "geojson_path": None}
	iou_mat = np.zeros((nB, nA), dtype=np.float32)
	for i, rb in gdf_before.iterrows():
		for j, ra in gdf_after.iterrows():
			iou_mat[i, j] = polygon_iou(rb.geometry, ra.geometry)

	# Hungarian maximum matching: but we want to maximize IoU -> minimize negative IoU
	row_ind, col_ind = linear_sum_assignment(-iou_mat)

	# 8) Compile results
	results = []
	matched_after_idx = set()
	for r, c in zip(row_ind, col_ind):
		iou = float(iou_mat[r, c])
		before = gdf_before.iloc[r]
		after = gdf_after.iloc[c]
		matched_after_idx.add(c)
		same_obj = (iou >= IOU_SAME_OBJECT)
		status = "matched_same" if same_obj else ("possible_change" if iou >= IOU_MATCH_THRESHOLD else "weak_match")
		# compute area change in pixels (we stored mask_pixels)
		area_before = before['mask_pixels']
		area_after = after['mask_pixels']
		pct_change = (area_after - area_before) / area_before if area_before > 0 else None
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
	for j, ra in gdf_after.reset_index().iterrows():
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
	for i, rb in gdf_before.reset_index().iterrows():
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
	df.to_csv(out_csv, index=False)
	print(f"Wrote CSV report: {out_csv}")

	# 9 Optionally export GeoJSON with change attributes (join geometries)
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
		out_geojson = os.path.join(output_dir, OUT_SHP_NAME)
		out_gdf.to_file(out_geojson, driver='GeoJSON')
		print(f"Wrote GeoJSON: {out_geojson}")
	print("Done.")

	return {"csv_path": out_csv, "geojson_path": out_geojson}


if __name__ == "__main__":
	main()