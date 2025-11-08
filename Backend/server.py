from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os, shutil, uuid, traceback, json, re

app = FastAPI()

# Enable CORS for Expo web (localhost:8081) and mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8081", "http://localhost:19006", "*"],  # "*" allows all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files after UPLOAD_DIR is defined below.

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")
@app.post("/analyze-debug/")
async def analyze_debug(request: Request):
    """Debug endpoint to inspect incoming multipart form keys and file metadata."""
    form = await request.form()
    keys = list(form.keys())
    files = {}
    for k, v in form.items():
        try:
            files[k] = {
                "is_uploadfile": hasattr(v, "filename"),
                "filename": getattr(v, "filename", None),
                "content_type": getattr(v, "content_type", None),
                "type": str(type(v)),
            }
        except Exception as _:
            files[k] = {"error": "inspect_failed"}
    return {"keys": keys, "files": files}


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
        
        # Determine checkpoint path (env override -> default in Backend)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint = os.getenv("SAM_CHECKPOINT_PATH", os.path.join(base_dir, "sam_vit_l_0b3195.pth"))
        
        # Run full pipeline through sam.py (SAM -> flood_change_pipeline -> geoJSON -> flood_analysis)
        try:
            from sam import run_full_pipeline
        except Exception as imp_err:
            raise HTTPException(status_code=500, detail=f"SAM pipeline not available: {imp_err}")

        try:
            results = run_full_pipeline(
                before_image_path=before_path,
                after_image_path=after_path,
                output_dir=work_dir,
                model_type=os.getenv("SAM_MODEL_TYPE", "vit_l"),
                checkpoint=checkpoint,
                pixel_resolution_m=effective_res,
            )
        except FileNotFoundError as fe:
            raise HTTPException(status_code=400, detail=f"SAM checkpoint missing. Set SAM_CHECKPOINT_PATH env or place the file at {checkpoint}. Error: {fe}")
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
    uvicorn.run(app, host="localhost", port=8000)
