# ============================================================
# SAM Mask Generator for Before & After Images
# ============================================================

import os
import cv2
from typing import Optional
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    sam_model_registry = None
    SamAutomaticMaskGenerator = None

# -------------------------------
# USER SETTINGS
# -------------------------------
DEFAULT_MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_l")  # can be vit_b, vit_l, vit_h
DEFAULT_CHECKPOINT = os.getenv("SAM_CHECKPOINT_PATH", os.path.join(os.path.dirname(__file__), "sam_vit_l_0b3195.pth"))

# -------------------------------
# LOAD MODEL
# -------------------------------
def _load_sam(model_type: str, checkpoint: str):
    if sam_model_registry is None:
        raise RuntimeError("segment_anything package not installed. Please install it to generate masks.")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    print(f"[INFO] Loading SAM model '{model_type}' from {checkpoint} ...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    try:
        sam.to(device)
    except Exception:
        sam.to("cpu")
        device = "cpu"
    print(f"[INFO] SAM model loaded on {device}.")
    return sam

# Function to generate masks for a given image
def generate_sam_masks(sam, image_path: str, output_folder: str,
                       points_per_side: int = 32,
                       pred_iou_thresh: float = 0.88,
                       stability_score_thresh: float = 0.95) -> int:
    print(f"\n[INFO] Processing image: {image_path}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize mask generator with tuned parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh
    )

    # Generate masks
    masks = mask_generator.generate(image_rgb)
    print(f"[INFO] Generated {len(masks)} masks.")

    # Save masks as binary PNGs
    for i, mask in enumerate(masks):
        filename = os.path.join(output_folder, f"mask_{i:03d}.png")
        mask_img = (mask["segmentation"].astype("uint8") * 255)
        cv2.imwrite(filename, mask_img)

    print(f"[INFO] Saved all masks to: {output_folder}")
    return len(masks)

def generate_before_after_masks(
    before_image_path: str,
    after_image_path: str,
    output_before_dir: str,
    output_after_dir: str,
    model_type: Optional[str] = None,
    checkpoint: Optional[str] = None,
) -> dict:
    """Generate SAM masks for before & after images; returns dict with counts.

    If segment-anything or checkpoint is missing, raises RuntimeError/FileNotFoundError.
    """
    model_type = model_type or DEFAULT_MODEL_TYPE
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    sam = _load_sam(model_type, checkpoint)
    before_count = generate_sam_masks(sam, before_image_path, output_before_dir)
    after_count = generate_sam_masks(sam, after_image_path, output_after_dir)
    return {
        "model_type": model_type,
        "checkpoint": checkpoint,
        "before_masks": before_count,
        "after_masks": after_count,
        "before_dir": output_before_dir,
        "after_dir": output_after_dir,
    }

def run_full_pipeline(
    before_image_path: str,
    after_image_path: str,
    output_dir: str,
    model_type: Optional[str] = None,
    checkpoint: Optional[str] = None,
    pixel_resolution_m: Optional[float] = None,
) -> dict:
    """
    Run the complete flood analysis pipeline:
    1. SAM mask generation
    2. Flood change pipeline (object matching & semantic analysis)
    3. GeoJSON overlay visualization
    4. Pixel-level flood analysis
    
    Returns comprehensive results dictionary.
    """
    import traceback
    
    # Step 1: Generate SAM masks
    print("\n[STEP 1/4] Running SAM mask generation...")
    mask_before_dir = os.path.join(output_dir, "sam_masks_before")
    mask_after_dir = os.path.join(output_dir, "sam_masks_after")
    os.makedirs(mask_before_dir, exist_ok=True)
    os.makedirs(mask_after_dir, exist_ok=True)
    
    try:
        sam_info = generate_before_after_masks(
            before_image_path=before_image_path,
            after_image_path=after_image_path,
            output_before_dir=mask_before_dir,
            output_after_dir=mask_after_dir,
            model_type=model_type,
            checkpoint=checkpoint,
        )
        print(f"[INFO] SAM generated {sam_info['before_masks']} before masks and {sam_info['after_masks']} after masks")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"SAM mask generation failed: {e}")
    
    # Step 2: Run flood change pipeline
    print("\n[STEP 2/4] Running flood change pipeline...")
    try:
        from flood_change_pipeline import main as run_pipeline
        pipeline_results = run_pipeline(
            before_img_path=before_image_path,
            after_img_path=after_image_path,
            output_dir=output_dir,
            sam_masks_before_dir=mask_before_dir,
            sam_masks_after_dir=mask_after_dir,
        )
        print(f"[INFO] Pipeline complete. CSV: {pipeline_results.get('csv_path')}, GeoJSON: {pipeline_results.get('geojson_path')}")
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Flood change pipeline failed: {e}")
    
    # Step 3: Generate GeoJSON overlay visualization
    print("\n[STEP 3/4] Generating GeoJSON overlay visualization...")
    overlay_info = None
    try:
        from geoJSON import overlay_geojson_on_images
        if pipeline_results.get("geojson_path") and os.path.isfile(pipeline_results["geojson_path"]):
            overlay_info = overlay_geojson_on_images(
                before_img_path=before_image_path,
                after_img_path=after_image_path,
                geojson_path=pipeline_results["geojson_path"],
                output_dir=output_dir,
                alpha=0.4
            )
            print(f"[INFO] Overlay visualization saved: {overlay_info.get('before_overlay')}, {overlay_info.get('after_overlay')}")
    except Exception as e:
        print(f"[WARN] GeoJSON overlay generation failed (non-fatal): {e}")
        traceback.print_exc()
    
    # Step 4: Run pixel-level flood analysis
    print("\n[STEP 4/4] Running pixel-level flood analysis...")
    pixel_analysis = None
    try:
        from flood_analysis import run_flood_pixel_analysis
        pixel_analysis = run_flood_pixel_analysis(
            before_image_path,
            after_image_path,
            os.path.join(output_dir, "pixel_analysis")
        )
        print(f"[INFO] Pixel analysis complete")
    except Exception as e:
        print(f"[WARN] Pixel analysis failed (non-fatal): {e}")
        traceback.print_exc()
    
    # Aggregate metrics from CSV with optional pixel resolution
    print("\n[INFO] Aggregating metrics...")
    metrics = {}
    csv_path = pipeline_results.get("csv_path")
    try:
        import pandas as pd
        if csv_path and os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            pixel_area_m2 = (pixel_resolution_m ** 2) if pixel_resolution_m else None
            
            def is_veg_loss(row):
                return isinstance(row['semantic_change'], str) and row['semantic_change'].startswith('VEGETATION') and '-> VEGETATION' not in row['semantic_change']
            
            def is_new_water(row):
                sc = str(row['semantic_change'])
                return ('-> WATER' in sc or '-> POSSIBLE_WATER' in sc) and not sc.startswith('WATER') and not sc.startswith('POSSIBLE_WATER')
            
            def is_built_structure_affected(row):
                sc = str(row['semantic_change'])
                # Built structures are typically BARE_SOIL in our classification
                # Affected if changed from BARE_SOIL to WATER or if still exists but has water nearby
                return ('BARE_SOIL -> WATER' in sc or 'BARE_SOIL -> POSSIBLE_WATER' in sc)
            
            veg_loss_px = int(df[df.apply(is_veg_loss, axis=1)]['area_before_px'].sum()) if 'area_before_px' in df.columns else 0
            new_water_px = int(df[df.apply(is_new_water, axis=1)]['area_after_px'].sum()) if 'area_after_px' in df.columns else 0
            built_affected_px = int(df[df.apply(is_built_structure_affected, axis=1)]['area_after_px'].sum()) if 'area_after_px' in df.columns else 0
            
            # Total flooded area includes all new water
            total_flooded_px = new_water_px
            
            metrics = {
                'total_objects_before': int(df['before_id'].notna().sum()) if 'before_id' in df.columns else 0,
                'total_objects_after': int(df['after_id'].notna().sum()) if 'after_id' in df.columns else 0,
                'vegetation_loss_pixels': veg_loss_px,
                'new_water_pixels': new_water_px,
                'built_structures_affected_pixels': built_affected_px,
                'total_flooded_pixels': total_flooded_px,
                'pixel_resolution_m': pixel_resolution_m,
                'vegetation_loss_area_m2': veg_loss_px * pixel_area_m2 if pixel_area_m2 else None,
                'new_water_area_m2': new_water_px * pixel_area_m2 if pixel_area_m2 else None,
                'built_structures_affected_area_m2': built_affected_px * pixel_area_m2 if pixel_area_m2 else None,
                'total_flooded_area_m2': total_flooded_px * pixel_area_m2 if pixel_area_m2 else None,
                'vegetation_loss_area_ha': (veg_loss_px * pixel_area_m2 / 10000.0) if pixel_area_m2 else None,
                'new_water_area_ha': (new_water_px * pixel_area_m2 / 10000.0) if pixel_area_m2 else None,
                'built_structures_affected_area_ha': (built_affected_px * pixel_area_m2 / 10000.0) if pixel_area_m2 else None,
                'total_flooded_area_ha': (total_flooded_px * pixel_area_m2 / 10000.0) if pixel_area_m2 else None,
            }
            
            # Enrich GeoJSON with area attributes if available
            if pixel_resolution_m and pipeline_results.get('geojson_path') and os.path.isfile(pipeline_results['geojson_path']):
                try:
                    import geopandas as gpd
                    gj = gpd.read_file(pipeline_results['geojson_path'])
                    if 'area_before_px' in gj.columns:
                        gj['area_before_m2'] = gj['area_before_px'].fillna(0) * pixel_area_m2
                        gj['area_before_ha'] = gj['area_before_m2'] / 10000.0
                    if 'area_after_px' in gj.columns:
                        gj['area_after_m2'] = gj['area_after_px'].fillna(0) * pixel_area_m2
                        gj['area_after_ha'] = gj['area_after_m2'] / 10000.0
                    gj.to_file(pipeline_results['geojson_path'], driver='GeoJSON')
                    print(f"[INFO] GeoJSON enriched with area measurements")
                except Exception as gje:
                    print(f"[WARN] Failed to enrich GeoJSON with area fields: {gje}")
    except Exception as met_err:
        print(f"[WARN] Metrics aggregation failed: {met_err}")
        traceback.print_exc()
    
    print("\n[SUCCESS] Full pipeline completed successfully!")
    
    return {
        "message": "Full pipeline complete",
        "sam": sam_info,
        "pipeline": pipeline_results,
        "overlay": overlay_info,
        "pixel_analysis": pixel_analysis,
        "metrics": metrics,
    }

# -------------------------------
# RUN FOR BOTH IMAGES
# -------------------------------
if __name__ == "__main__":
    # Example CLI usage with defaults
    before_image_path = os.path.join(os.path.dirname(__file__), "before.png")
    after_image_path = os.path.join(os.path.dirname(__file__), "after.png")
    output_dir = os.path.join(os.path.dirname(__file__), "pipeline_output")
    
    try:
        results = run_full_pipeline(
            before_image_path=before_image_path,
            after_image_path=after_image_path,
            output_dir=output_dir,
        )
        print("\n[INFO] Full pipeline results:", results)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

