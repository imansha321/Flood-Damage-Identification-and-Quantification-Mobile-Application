import geopandas as gpd
import cv2
import numpy as np
import warnings
import os
from typing import Optional, Dict, Any

def align_images_ecc(ref_img, moving_img, warp_mode=cv2.MOTION_HOMOGRAPHY,
                     num_iter=5000, eps=1e-7):
    """Align moving_img to ref_img using ECC."""
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    mov_gray = cv2.cvtColor(moving_img, cv2.COLOR_RGB2GRAY)

    if ref_gray.shape != mov_gray.shape:
        mov_gray = cv2.resize(mov_gray, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_LINEAR)
        moving_img = cv2.resize(moving_img, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_LINEAR)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, num_iter, eps)

    try:
        _, warp_matrix = cv2.findTransformECC(ref_gray, mov_gray, warp_matrix, warp_mode, criteria, None, 1)
    except cv2.error as e:
        warnings.warn("ECC alignment failed: " + str(e))
        return moving_img, warp_matrix

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(moving_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        aligned = cv2.warpAffine(moving_img, warp_matrix, (ref_img.shape[1], ref_img.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return aligned, warp_matrix

def _build_color_map(gdf: gpd.GeoDataFrame) -> Optional[Dict[str, np.ndarray]]:
    """Build a color map for semantic_change values with intuitive color coding."""
    if 'semantic_change' not in gdf.columns:
        return None
    
    # Define semantic colors for better interpretation
    color_rules = {
        # Water-related (Blue tones)
        'WATER': (0, 100, 255),  # Blue
        'POSSIBLE_WATER': (100, 150, 255),  # Light blue
        
        # Vegetation-related (Green tones)
        'VEGETATION': (0, 200, 50),  # Green
        
        # Bare soil / Built structures (Orange/Brown tones)
        'BARE_SOIL': (200, 150, 100),  # Tan/Brown
        
        # No change
        'NONE': (128, 128, 128),  # Gray
    }
    
    color_map = {}
    unique_changes = gdf['semantic_change'].unique()
    
    for change in unique_changes:
        change_str = str(change)
        
        # Prioritize water detection (flooding)
        if '-> WATER' in change_str or '-> POSSIBLE_WATER' in change_str:
            color_map[change] = np.array([0, 100, 255], dtype=np.uint8)  # Blue for flooded
        
        # Vegetation loss
        elif 'VEGETATION ->' in change_str and 'VEGETATION' not in change_str.split('->')[1]:
            color_map[change] = np.array([50, 200, 0], dtype=np.uint8)  # Green for veg loss
        
        # Built structures affected (bare soil to water)
        elif 'BARE_SOIL -> WATER' in change_str or 'BARE_SOIL -> POSSIBLE_WATER' in change_str:
            color_map[change] = np.array([255, 140, 0], dtype=np.uint8)  # Orange for built structures
        
        # Vegetation to water (combined category)
        elif 'VEGETATION -> WATER' in change_str or 'VEGETATION -> POSSIBLE_WATER' in change_str:
            color_map[change] = np.array([0, 150, 255], dtype=np.uint8)  # Cyan for veg-to-water
        
        # No significant change
        elif '->' not in change_str or (change_str.split('->')[0].strip() == change_str.split('->')[1].strip()):
            color_map[change] = np.array([180, 180, 180], dtype=np.uint8)  # Light gray
        
        # Default fallback
        else:
            # Generate a semi-random color based on hash
            hash_val = hash(change) % 256
            color_map[change] = np.array([hash_val, (hash_val * 2) % 256, (hash_val * 3) % 256], dtype=np.uint8)
    
    return color_map

def _draw_polygons_on_image(image: np.ndarray, gdf: gpd.GeoDataFrame, color_map=None, alpha=0.4) -> np.ndarray:
    """Draw GeoJSON polygons on an image with transparency."""
    img_overlay = image.copy()
    for _, row in gdf.iterrows():
        geom = row['geometry']
        if hasattr(geom, 'exterior'):
            coords = np.array(geom.exterior.coords).astype(np.int32)
            if color_map is not None and 'semantic_change' in row:
                color = color_map.get(row['semantic_change'], np.array([0, 0, 255], dtype=np.uint8))
            else:
                color = np.array([0, 0, 255], dtype=np.uint8)
            overlay = img_overlay.copy()
            cv2.fillPoly(overlay, [coords], color.tolist())
            cv2.addWeighted(overlay, alpha, img_overlay, 1 - alpha, 0, img_overlay)
            cv2.polylines(img_overlay, [coords], isClosed=True, color=(0, 0, 0), thickness=2)
    return img_overlay

def overlay_geojson_on_images(before_img_path: str, after_img_path: str, geojson_path: str, output_dir: str, alpha: float = 0.4) -> Dict[str, Any]:
    """Overlay GeoJSON polygons on before and aligned-after images.
    Saves PNGs into output_dir and returns their paths and basic info.
    """
    if not os.path.isfile(geojson_path):
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
    
    if not os.path.isfile(before_img_path):
        raise FileNotFoundError(f"Before image not found: {before_img_path}")
    
    if not os.path.isfile(after_img_path):
        raise FileNotFoundError(f"After image not found: {after_img_path}")

    # Read images with error checking
    img_before = cv2.imread(before_img_path)
    if img_before is None:
        raise ValueError(f"Failed to load before image: {before_img_path}")
    img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
    
    img_after = cv2.imread(after_img_path)
    if img_after is None:
        raise ValueError(f"Failed to load after image: {after_img_path}")
    img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
    
    aligned_after, _ = align_images_ecc(img_before, img_after)

    gdf = gpd.read_file(geojson_path)
    color_map = _build_color_map(gdf)

    before_overlay = _draw_polygons_on_image(img_before, gdf, color_map=color_map, alpha=alpha)
    after_overlay = _draw_polygons_on_image(aligned_after, gdf, color_map=color_map, alpha=alpha)

    os.makedirs(output_dir, exist_ok=True)
    out_before = os.path.join(output_dir, "before_with_polygons.png")
    out_after = os.path.join(output_dir, "after_with_polygons.png")
    cv2.imwrite(out_before, cv2.cvtColor(before_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_after, cv2.cvtColor(after_overlay, cv2.COLOR_RGB2BGR))

    h, w = img_before.shape[:2]
    return {
        "before_overlay": out_before,
        "after_overlay": out_after,
        "image_width": int(w),
        "image_height": int(h),
        "polygon_count": int(len(gdf)),
        "change_types": list(gdf['semantic_change'].unique()) if 'semantic_change' in gdf.columns else []
    }


# Legacy function for backward compatibility with existing server.py calls
def generate_change_overlay(before_img_path: str, after_img_path: str, geojson_path: str, output_path: Optional[str] = None) -> Optional[str]:
    """Generate polygon overlays and return the 'after' overlay path for backward compatibility."""
    if not geojson_path or not os.path.isfile(geojson_path):
        return None
    output_dir = os.path.dirname(output_path) if output_path else os.path.dirname(geojson_path)
    try:
        result = overlay_geojson_on_images(before_img_path, after_img_path, geojson_path, output_dir)
        # Return after_overlay path for compatibility
        return result.get("after_overlay")
    except Exception as e:
        warnings.warn(f"generate_change_overlay failed: {e}")
        return None


if __name__ == "__main__":
    base = os.path.dirname(__file__) or "."
    before = os.path.join(base, "before.png")
    after = os.path.join(base, "after.png")
    gj = os.path.join(base, "objects_change_report.geojson")
    out = os.path.join(base, "geojson_overlays")
    os.makedirs(out, exist_ok=True)
    try:
        info = overlay_geojson_on_images(before, after, gj, out)
        print("Overlay info:", info)
    except Exception as e:
        print("GeoJSON overlay failed:", e)
