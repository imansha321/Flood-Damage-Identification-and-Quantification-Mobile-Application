from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os, shutil, uuid, traceback, json, re
import numpy as np
import cv2
import pandas as pd

app = FastAPI()

# Enable CORS for Expo web (localhost:8081) and mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://localhost:19006", "http://192.168.222.80:8081", "http://192.168.222.80:19006", "*"],  # "*" allows all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files after UPLOAD_DIR is defined below.

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")


@app.get("/")
async def root():
    return {"message": "Flood Damage Identification and Quantification API"}

@app.post("/analyze/")
async def analyze_flood(
    before: UploadFile | None = File(None),
    after: UploadFile | None = File(None),
    pixel_resolution_m: float | None = Query(None, description="Ground sampling distance in meters per pixel (optional)")
):
    """Analyze flood impact between two images.
    Creates a per-request working directory to avoid collisions.
    Returns paths to generated CSV and GeoJSON (if available)."""
    # Basic validation
    if before is None or after is None or not getattr(before, "filename", None) or not getattr(after, "filename", None):
        raise HTTPException(status_code=400, detail="Both before and after images are required")

    # Create unique working directory
    request_id = str(uuid.uuid4())
    work_dir = os.path.join(UPLOAD_DIR, request_id)
    os.makedirs(work_dir, exist_ok=True)

    before_path = os.path.join(work_dir, "before.png")
    after_path = os.path.join(work_dir, "after.png")

    try:
        # Save uploaded images
        with open(before_path, "wb") as f:
            shutil.copyfileobj(before.file, f)
        with open(after_path, "wb") as f:
            shutil.copyfileobj(after.file, f)

        # Try to auto-detect pixel resolution if not provided
        detected_res = None
        try:
            # 1) PNG pHYs chunk (physical pixel dimensions)
            try:
                from PIL import Image
                with Image.open(before_path) as img:
                    # pHYs returns (x_dpi, y_dpi, unit) where unit=1 is pixels/meter
                    if 'dpi' in img.info:
                        dpi_x, dpi_y = img.info['dpi']
                        # DPI is dots per inch; convert to meters per pixel
                        # 1 inch = 0.0254 meters
                        # resolution = 0.0254 / DPI
                        detected_res = 0.0254 / float(dpi_x)
                        print(f"[INFO] Detected resolution from PNG DPI: {detected_res:.6f} m/px (DPI: {dpi_x})")
            except Exception as pil_err:
                print(f"[WARN] PNG metadata read failed: {pil_err}")
            
            # 2) World file next to before image: .pgw/.wld (PNG world file uses .pgw)
            if detected_res is None:
                base_no_ext, _ = os.path.splitext(before_path)
                for ext in ('.pgw', '.wld'):
                    wf = base_no_ext + ext
                    if os.path.isfile(wf):
                        with open(wf, 'r') as f:
                            lines = [l.strip() for l in f.readlines() if l.strip()]
                        # World file first line is pixel size in x units/pixel
                        if lines:
                            try:
                                detected_res = abs(float(lines[0]))
                                print(f"[INFO] Detected resolution from world file: {detected_res} m/px")
                                break
                            except Exception:
                                pass
            
            # 3) Filename pattern e.g., before_0p5mpp.png or before_0.5mpp.png
            if detected_res is None:
                m = re.search(r"([0-9]+(?:[\.,][0-9]+)?)\s*mpp|mperpixel|mppx|m_per_px", before.filename or '', flags=re.IGNORECASE)
                if m:
                    detected_res = float(m.group(1).replace(',', '.'))
                    print(f"[INFO] Detected resolution from filename: {detected_res} m/px")
        except Exception as _auto_err:
            print(f"[WARN] auto-res detection failed: {_auto_err}")

        # Determine effective resolution
        effective_res = pixel_resolution_m or detected_res
        
        # Run full pipeline through algorithm.py (handles SAM, alignment, masking, and change detection)
        try:
            # Import and setup paths for algorithm.py pipeline
            import sys
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            
            # Import algorithm functions
            from algorithm import (
                read_rgb, align_images_ecc, histogram_match, compute_exg, compute_g_minus_r,
                brightness, overlay_mask_on_rgb, load_binary_masks_from_folder,
                raster_labels_to_polygons, compute_rgb_proxies, polygon_iou,
                MIN_MASK_AREA_PIXELS, IOU_MATCH_THRESHOLD, IOU_SAME_OBJECT,
                EXG_VEG_THRESHOLD, WATER_G_R_DIFF, DARKNESS_BRIGHTNESS
            )
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            import torch
            from PIL import Image
            from rasterio.features import shapes
            from shapely.geometry import shape
            import geopandas as gpd
            from scipy.optimize import linear_sum_assignment
            from rasterio.transform import from_origin
            import imageio
            
        except Exception as imp_err:
            raise HTTPException(status_code=500, detail=f"Pipeline imports failed: {imp_err}")

        try:
            # Determine SAM checkpoint path
            checkpoint = os.getenv("SAM_CHECKPOINT_PATH", os.path.join(backend_dir, "sam_vit_h_4b8939.pth"))
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint}")
            
            # Run the complete pipeline from algorithm.py
            print("[INFO] Starting flood analysis pipeline...")
            
            # 1) Load and align images
            print("[INFO] Loading images...")
            img_before = read_rgb(before_path)
            img_after = read_rgb(after_path)
            
            print("[INFO] Aligning images...")
            aligned_after, warp = align_images_ecc(img_before, img_after, warp_mode=cv2.MOTION_AFFINE, num_iter=3000, eps=1e-6)
            aligned_after = histogram_match(aligned_after, img_before)
            
            # 2) Run SAM segmentation
            print("[INFO] Running SAM segmentation...")
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            model_type = os.getenv("SAM_MODEL_TYPE", "vit_h")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=DEVICE)
            
            # Generate masks for both images
            image_np_before = np.array(Image.open(before_path).convert("RGB"))
            image_np_after = np.array(Image.open(after_path).convert("RGB"))
            
            mask_generator = SamAutomaticMaskGenerator(
                sam, points_per_side=32, pred_iou_thresh=0.88, stability_score_thresh=0.95
            )
            
            masks_before = mask_generator.generate(image_np_before)
            masks_after = mask_generator.generate(image_np_after)
            
            # Clean up SAM model from GPU memory immediately after inference
            del sam
            del mask_generator
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("[INFO] Released SAM model from GPU memory")
            
            # 3) Process and save masks
            print(f"[INFO] Generated {len(masks_before)} masks for before, {len(masks_after)} for after")
            
            out_dir_before = os.path.join(work_dir, "sam_masks_before")
            out_dir_after = os.path.join(work_dir, "sam_masks_after")
            os.makedirs(out_dir_before, exist_ok=True)
            os.makedirs(out_dir_after, exist_ok=True)
            
            # Flip and save before masks
            for i, mask_dict in enumerate(masks_before):
                flipped_mask = cv2.flip(mask_dict['segmentation'].astype(np.uint8) * 255, 0)
                mask_fp = os.path.join(out_dir_before, f"mask_{i:03d}.png")
                imageio.imwrite(mask_fp, flipped_mask)
            
            # Warp, flip and save after masks
            for i, mask_dict in enumerate(masks_after):
                mask = mask_dict['segmentation'].astype(np.uint8) * 255
                warped_mask = cv2.warpAffine(mask, warp, (image_np_before.shape[1], image_np_before.shape[0]),
                                            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
                warped_mask = cv2.flip(warped_mask, 0)
                mask_fp = os.path.join(out_dir_after, f"mask_{i:03d}.png")
                imageio.imwrite(mask_fp, warped_mask)
            
            # 4) Run change detection pipeline
            print("[INFO] Running change detection analysis...")
            labels_before, idmap_before = load_binary_masks_from_folder(out_dir_before, img_before.shape)
            labels_after, idmap_after = load_binary_masks_from_folder(out_dir_after, img_before.shape)
            
            # Vectorize masks
            height_before, width_before = labels_before.shape
            height_after, width_after = labels_after.shape
            transform_before = from_origin(0, height_before, 1, 1)
            transform_after = from_origin(0, height_after, 1, 1)
            
            gdf_before = raster_labels_to_polygons(labels_before, transform=transform_before, crs=None)
            gdf_after = raster_labels_to_polygons(labels_after, transform=transform_after, crs=None)
            
            # Compute RGB proxies
            def per_row_compute_proxies(gdf, img):
                exg_list, gr_list, bright_list, masks = [], [], [], []
                h, w = img.shape[:2]
                for idx, row in gdf.iterrows():
                    rr = np.zeros((h, w), dtype=np.uint8)
                    coords = np.array(list(row.geometry.exterior.coords)).astype(np.int32)
                    try:
                        cv2.fillPoly(rr, [coords], 1)
                    except Exception:
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
            gdf_after = per_row_compute_proxies(gdf_after, aligned_after)
            
            # Calculate pixel-level change detection
            print("[INFO] Computing pixel-level change detection...")
            
            # Compute vegetation and water change masks
            exg_before = compute_exg(img_before)
            exg_after = compute_exg(aligned_after)
            gmr_before = compute_g_minus_r(img_before)
            gmr_after = compute_g_minus_r(aligned_after)
            bright_before = brightness(img_before)
            bright_after = brightness(aligned_after)
            
            # Difference maps
            exg_diff = exg_before - exg_after  # positive -> vegetation loss
            gmr_diff = gmr_after - gmr_before  # positive -> greener or water
            bright_diff = bright_after - bright_before
            
            # Thresholds for pixel change detection
            EXG_LOSS_T = 20.0
            GMR_WATER_T = 10.0
            BRIGHT_DARK = 100.0
            
            # Create change masks
            veg_loss_mask = (exg_diff > EXG_LOSS_T)
            water_candidate_mask = ((gmr_after > GMR_WATER_T) & (bright_after < BRIGHT_DARK))
            
            # Calculate pixel change statistics
            total_pixels = img_before.shape[0] * img_before.shape[1]
            veg_loss_pixels = int(np.sum(veg_loss_mask))
            water_candidate_pixels = int(np.sum(water_candidate_mask))
            veg_loss_pct = (veg_loss_pixels / total_pixels) * 100.0
            water_candidate_pct = (water_candidate_pixels / total_pixels) * 100.0
            
            # Create combined change mask visualization
            change_mask_rgb = aligned_after.copy()
            change_mask_rgb = overlay_mask_on_rgb(change_mask_rgb, veg_loss_mask, color=(255, 0, 0), alpha=0.4)
            change_mask_rgb = overlay_mask_on_rgb(change_mask_rgb, water_candidate_mask, color=(0, 0, 255), alpha=0.4)
            
            change_mask_path = os.path.join(work_dir, "pixel_change_mask.png")
            imageio.imwrite(change_mask_path, change_mask_rgb)
            
            # Save pixel change summary CSV
            pixel_summary = {
                "total_pixels": total_pixels,
                "veg_loss_pixels": veg_loss_pixels,
                "water_candidate_pixels": water_candidate_pixels,
                "veg_loss_pct": veg_loss_pct,
                "water_candidate_pct": water_candidate_pct,
                "pixel_resolution_m": effective_res
            }
            
            pixel_csv_path = os.path.join(work_dir, "pixel_change_summary.csv")
            pd.DataFrame([pixel_summary]).to_csv(pixel_csv_path, index=False)
            print(f"[INFO] Pixel change: {veg_loss_pixels} veg loss, {water_candidate_pixels} water candidate")
            
            # Generate overlay images with polygons
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
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
            
            before_overlay_path = os.path.join(work_dir, "before_with_polygons.png")
            after_overlay_path = os.path.join(work_dir, "after_with_polygons.png")
            save_image_with_polygons(img_before, gdf_before, before_overlay_path)
            save_image_with_polygons(aligned_after, gdf_after, after_overlay_path)
            print(f"[INFO] Saved overlay images: {before_overlay_path}, {after_overlay_path}")
            
            # Semantic labeling
            def semantic_label_fn(row):
                if row['exg'] >= EXG_VEG_THRESHOLD:
                    return "VEGETATION"
                if (row['g_r'] >= WATER_G_R_DIFF) and (row['brightness'] <= DARKNESS_BRIGHTNESS):
                    return "WATER"
                if row['brightness'] < DARKNESS_BRIGHTNESS:
                    return "POSSIBLE_WATER"
                return "BARE_SOIL"
            
            gdf_before['class'] = gdf_before.apply(semantic_label_fn, axis=1)
            gdf_after['class'] = gdf_after.apply(semantic_label_fn, axis=1)
            
            # IoU matching
            nB, nA = len(gdf_before), len(gdf_after)
            results_list = []
            
            if nB > 0 and nA > 0:
                iou_mat = np.zeros((nB, nA), dtype=np.float32)
                for i, rb in gdf_before.iterrows():
                    for j, ra in gdf_after.iterrows():
                        iou_mat[i, j] = polygon_iou(rb.geometry, ra.geometry)
                
                row_ind, col_ind = linear_sum_assignment(-iou_mat)
                matched_after_idx = set()
                
                for r, c in zip(row_ind, col_ind):
                    iou = float(iou_mat[r, c])
                    before = gdf_before.iloc[r]
                    after = gdf_after.iloc[c]
                    matched_after_idx.add(c)
                    same_obj = (iou >= IOU_SAME_OBJECT)
                    status = "matched_same" if same_obj else ("possible_change" if iou >= IOU_MATCH_THRESHOLD else "weak_match")
                    area_before = before['mask_pixels']
                    area_after = after['mask_pixels']
                    pct_change = (area_after - area_before) / area_before if area_before > 0 else None
                    sem_change = f"{before['class']} -> {after['class']}"
                    
                    results_list.append({
                        'before_id': int(before['id']), 'after_id': int(after['id']),
                        'iou': iou, 'status': status,
                        'area_before_px': int(area_before), 'area_after_px': int(area_after),
                        'pct_change': float(pct_change) if pct_change is not None else None,
                        'semantic_change': sem_change,
                        'before_exg': float(before['exg']), 'after_exg': float(after['exg']),
                        'before_g_r': float(before['g_r']), 'after_g_r': float(after['g_r'])
                    })
                
                # New objects
                for j, ra in gdf_after.reset_index().iterrows():
                    if j not in matched_after_idx:
                        results_list.append({
                            'before_id': None, 'after_id': int(ra['id']),
                            'iou': 0.0, 'status': 'new_after',
                            'area_before_px': 0, 'area_after_px': int(ra['mask_pixels']),
                            'pct_change': None, 'semantic_change': f"NONE -> {ra['class']}",
                            'before_exg': None, 'after_exg': float(ra['exg']),
                            'before_g_r': None, 'after_g_r': float(ra['g_r'])
                        })
                
                # Deleted objects
                matched_before_idx = set(row_ind)
                for i, rb in gdf_before.reset_index().iterrows():
                    if i not in matched_before_idx:
                        results_list.append({
                            'before_id': int(rb['id']), 'after_id': None,
                            'iou': 0.0, 'status': 'deleted_after',
                            'area_before_px': int(rb['mask_pixels']), 'area_after_px': 0,
                            'pct_change': None, 'semantic_change': f"{rb['class']} -> NONE",
                            'before_exg': float(rb['exg']), 'after_exg': None,
                            'before_g_r': float(rb['g_r']), 'after_g_r': None
                        })
            
            # Save CSV report
            df_results = pd.DataFrame(results_list)
            csv_path = os.path.join(work_dir, "objects_change_report.csv")
            df_results.to_csv(csv_path, index=False)
            print(f"[INFO] Wrote CSV report: {csv_path}")
            
            # Save GeoJSON
            features = []
            for row in results_list:
                geom = None
                if row['before_id'] is not None:
                    sel = gdf_before[gdf_before['id'] == row['before_id']]
                    if len(sel) > 0:
                        geom = sel.iloc[0].geometry
                if geom is None and row['after_id'] is not None:
                    sel = gdf_after[gdf_after['id'] == row['after_id']]
                    if len(sel) > 0:
                        geom = sel.iloc[0].geometry
                if geom is not None:
                    features.append({'id': f"{row['before_id']}_{row['after_id']}", 'geometry': geom, **row})
            
            geojson_path = None
            if len(features) > 0:
                out_gdf = gpd.GeoDataFrame(features, geometry='geometry')
                geojson_path = os.path.join(work_dir, "objects_change_report.geojson")
                out_gdf.to_file(geojson_path, driver='GeoJSON')
                print(f"[INFO] Wrote GeoJSON: {geojson_path}")
            
            # Calculate summary metrics
            total_pixels = img_before.shape[0] * img_before.shape[1]
            changed_pixels = sum([r['area_after_px'] - r['area_before_px'] for r in results_list if r['status'] == 'possible_change'])
            
            # Calculate pixel-level area changes if resolution is known
            pixel_area_info = {}
            if effective_res:
                pixel_area_m2 = effective_res ** 2
                pixel_area_info = {
                    "veg_loss_m2": veg_loss_pixels * pixel_area_m2,
                    "water_candidate_m2": water_candidate_pixels * pixel_area_m2,
                    "veg_loss_ha": (veg_loss_pixels * pixel_area_m2) / 10000,
                    "water_candidate_ha": (water_candidate_pixels * pixel_area_m2) / 10000
                }
            
            results = {
                "pipeline": {"csv_path": csv_path, "geojson_path": geojson_path},
                "sam": {"before_masks": len(masks_before), "after_masks": len(masks_after)},
                "metrics": {
                    "total_objects_before": nB,
                    "total_objects_after": nA,
                    "matched_objects": len([r for r in results_list if r['status'] in ['matched_same', 'possible_change']]),
                    "new_objects": len([r for r in results_list if r['status'] == 'new_after']),
                    "deleted_objects": len([r for r in results_list if r['status'] == 'deleted_after']),
                    "total_pixels": total_pixels,
                    "pixel_resolution_m": effective_res
                },
                "overlay": {
                    "before_overlay": before_overlay_path,
                    "after_overlay": after_overlay_path
                },
                "pixel_analysis": {
                    "change_mask_path": change_mask_path,
                    "csv_path": pixel_csv_path,
                    "veg_loss_pixels": veg_loss_pixels,
                    "water_candidate_pixels": water_candidate_pixels,
                    "veg_loss_pct": veg_loss_pct,
                    "water_candidate_pct": water_candidate_pct,
                    **pixel_area_info
                }
            }
            
            print("[INFO] Analysis complete")
            
            # Final GPU memory cleanup
            if 'torch' in sys.modules:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("[INFO] Final GPU memory cleanup completed")
            
        except FileNotFoundError as fe:
            raise HTTPException(status_code=400, detail=f"Required file not found: {fe}")
        except Exception as se:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {se}")
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # Convert absolute paths to relative URLs for static file access
    def make_relative_url(abs_path):
        if abs_path and os.path.isfile(abs_path):
            # Get absolute path of UPLOAD_DIR to properly compute relative path
            upload_dir_abs = os.path.abspath(UPLOAD_DIR)
            # Convert absolute path to relative from UPLOAD_DIR
            rel_path = os.path.relpath(abs_path, upload_dir_abs)
            # Convert backslashes to forward slashes for URLs
            return f"/files/{rel_path.replace(os.sep, '/')}"
        return None

    overlay_info = results.get("overlay", {})
    if overlay_info:
        overlay_info = {
            **overlay_info,
            "before_overlay_url": make_relative_url(overlay_info.get("before_overlay")),
            "after_overlay_url": make_relative_url(overlay_info.get("after_overlay")),
        }

    pixel_analysis = results.get("pixel_analysis", {})
    if pixel_analysis and "change_mask_path" in pixel_analysis:
        pixel_analysis = {
            **pixel_analysis,
            "change_mask_url": make_relative_url(pixel_analysis.get("change_mask_path")),
        }

    return {
        "message": "Analysis complete",
        "request_id": request_id,
        "csv_url": make_relative_url(results.get("pipeline", {}).get("csv_path")),
        "geojson_url": make_relative_url(results.get("pipeline", {}).get("geojson_path")),
        "overlay_info": overlay_info,
        "pixel_analysis": pixel_analysis,
        "metrics": results.get("metrics"),
        "sam": {
            "before_masks": results.get("sam", {}).get("before_masks"),
            "after_masks": results.get("sam", {}).get("after_masks"),
        }
    }
@app.get("/results/{request_id}")
async def get_results_summary(request_id: str):
    work_dir = os.path.join(UPLOAD_DIR, request_id)
    if not os.path.isdir(work_dir):
        raise HTTPException(status_code=404, detail="Request ID not found")
    summary_fp = os.path.join(work_dir, "objects_change_report.csv")
    overlay_fp = os.path.join(work_dir, "change_detection_overlay.png")
    geojson_fp = os.path.join(work_dir, "objects_change_report.geojson")
    resp = {
        'request_id': request_id,
        'csv_exists': os.path.isfile(summary_fp),
        'geojson_exists': os.path.isfile(geojson_fp),
        'overlay_exists': os.path.isfile(overlay_fp),
        'files_base_url': f"/files/{request_id}/"
    }
    return resp

@app.get("/results/{request_id}/geojson")
async def download_geojson(request_id: str):
    fp = os.path.join(UPLOAD_DIR, request_id, "objects_change_report.geojson")
    if not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="GeoJSON not found")
    return FileResponse(fp, media_type="application/geo+json", filename="objects_change_report.geojson")

@app.get("/results/{request_id}/overlay")
async def download_overlay(request_id: str):
    fp = os.path.join(UPLOAD_DIR, request_id, "change_detection_overlay.png")
    if not os.path.isfile(fp):
        raise HTTPException(status_code=404, detail="Overlay not found")
    return FileResponse(fp, media_type="image/png", filename="change_detection_overlay.png")

@app.get("/results/{request_id}/polygons")
async def polygons_json(request_id: str):
    geojson_fp = os.path.join(UPLOAD_DIR, request_id, "objects_change_report.geojson")
    before_fp = os.path.join(UPLOAD_DIR, request_id, "before.png")
    if not os.path.isfile(geojson_fp):
        raise HTTPException(status_code=404, detail="GeoJSON not found")
    try:
        import geopandas as gpd
        import cv2
        gdf = gpd.read_file(geojson_fp)
        img = cv2.imread(before_fp)
        h, w = (img.shape[0], img.shape[1]) if img is not None else (None, None)
        polys = []
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            exterior = list(geom.exterior.coords) if geom.exterior else []
            coords = [{'x': float(x), 'y': float(y)} for x, y in exterior]
            polys.append({
                'id': row.get('id'),
                'semantic_change': row.get('semantic_change'),
                'area_before_px': row.get('area_before_px'),
                'area_after_px': row.get('area_after_px'),
                'area_before_m2': row.get('area_before_m2'),
                'area_after_m2': row.get('area_after_m2'),
                'area_before_ha': row.get('area_before_ha'),
                'area_after_ha': row.get('area_after_ha'),
                'iou': row.get('iou'),
                'status': row.get('status'),
                'coordinates': coords
            })
        return JSONResponse({'request_id': request_id, 'image_width': w, 'image_height': h, 'polygons': polys})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse polygons: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
