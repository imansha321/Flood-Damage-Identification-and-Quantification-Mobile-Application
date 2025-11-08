# Backend API

FastAPI server for Flood Damage Identification and Quantification.

## Endpoints

- POST /analyze/
  - multipart/form-data with two files:
    - before: image (png/jpg)
    - after: image (png/jpg)
  - Returns JSON with:
    - message
    - request_id
    - csv_path: path to per-request CSV report
    - geojson_path: path to per-request GeoJSON (may be null if no objects)

Each request is processed in an isolated folder under `Backend/uploads/<uuid>/` to avoid collisions.

## Pipeline module

`Backend/flood_change_pipeline.py` now exposes a parameterized `main(...)` function:

```
from flood_change_pipeline import main
result = main(
    before_img_path="/abs/path/to/before.png",
    after_img_path="/abs/path/to/after.png",
    output_dir="/abs/path/to/output/dir",
    sam_masks_before_dir="/abs/path/to/sam_masks_before",  # optional
    sam_masks_after_dir="/abs/path/to/sam_masks_after",    # optional
    sam_indexed_before=None,  # optional
    sam_indexed_after=None,   # optional
)
print(result)  # {"csv_path": ..., "geojson_path": ...}
```

If mask folders are missing or empty, the pipeline will still run and write an empty CSV with headers.

## Dependencies

Install required packages (example):

```
pip install fastapi uvicorn[standard] opencv-python numpy scikit-image scipy pandas shapely rasterio geopandas
```

Note: `geopandas`/`rasterio` require native dependencies; consider using conda for easier setup.

## Run server (Windows PowerShell)

```
$env:PYTHONPATH = (Resolve-Path .\Backend).Path
python -m uvicorn Backend.server:app --host localhost --port 8000 --reload
```

