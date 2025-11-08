# Flood Analysis Pipeline Architecture

## Quick Start

### Running the Server
```bash
cd Backend
python server.py
```

Server will start at `http://localhost:8000`

### API Endpoint
```
POST /analyze/
```

**Parameters:**
- `before`: Image file (before flood)
- `after`: Image file (after flood)
- `pixel_resolution_m`: Optional ground sampling distance in meters/pixel

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/analyze/" \
  -F "before=@path/to/before.png" \
  -F "after=@path/to/after.png" \
  -F "pixel_resolution_m=0.5"
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT REQUEST                            │
│                    POST /analyze/                                │
│              (before.png, after.png, resolution)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SERVER.PY                                │
│                    (FastAPI Handler)                             │
│                                                                   │
│  • Receives uploads                                              │
│  • Creates work directory                                        │
│  • Detects pixel resolution (DPI/world file/filename)            │
│  • Calls sam.run_full_pipeline()                                 │
│  • Returns JSON response                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                          SAM.PY                                  │
│                  (Pipeline Orchestrator)                         │
│                                                                   │
│  run_full_pipeline() coordinates:                                │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STEP 1: SAM Mask Generation                             │   │
│  │  • Load SAM model                                         │   │
│  │  • Generate masks for before image → sam_masks_before/   │   │
│  │  • Generate masks for after image → sam_masks_after/     │   │
│  │  • Returns mask counts                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STEP 2: flood_change_pipeline.py                        │   │
│  │  • Align images (ECC registration)                        │   │
│  │  • Histogram matching                                     │   │
│  │  • Load SAM masks                                         │   │
│  │  • Vectorize masks to polygons                            │   │
│  │  • Compute RGB proxies (ExG, G-R)                         │   │
│  │  • Semantic classification (VEG/WATER/BARE_SOIL)          │   │
│  │  • IoU matching (Hungarian algorithm)                     │   │
│  │  • Generate change report                                 │   │
│  │  • Output: objects_change_report.csv + .geojson           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STEP 3: geoJSON.py                                       │   │
│  │  • Read GeoJSON polygons                                  │   │
│  │  • Align before/after images                              │   │
│  │  • Create color map for semantic changes                  │   │
│  │  • Draw polygons on images with transparency              │   │
│  │  • Output: before_with_polygons.png, after_...png         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STEP 4: flood_analysis.py                                │   │
│  │  • Pixel-level change detection                           │   │
│  │  • Compute absolute difference                            │   │
│  │  • Threshold change mask                                  │   │
│  │  • Calculate statistics                                   │   │
│  │  • Output: pixel_change_summary.csv, change_mask.png      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  METRICS AGGREGATION                                      │   │
│  │  • Parse CSV results                                       │   │
│  │  • Calculate vegetation loss                              │   │
│  │  • Calculate new water areas                              │   │
│  │  • Convert pixels → m² → hectares                         │   │
│  │  • Enrich GeoJSON with area attributes                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RESULTS RETURNED                              │
│                                                                   │
│  {                                                                │
│    "sam": { before_masks, after_masks },                         │
│    "pipeline": { csv_path, geojson_path },                       │
│    "overlay": { before_overlay, after_overlay },                 │
│    "pixel_analysis": { total_pixels, changed_pixels },           │
│    "metrics": {                                                   │
│      vegetation_loss_pixels, vegetation_loss_area_ha,            │
│      new_water_pixels, new_water_area_ha, ...                    │
│    }                                                              │
│  }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## Output Files Structure

After analysis, the work directory contains:

```
uploads/{request_id}/
├── before.png                          # Uploaded before image
├── after.png                           # Uploaded after image
├── sam_masks_before/                   # SAM masks for before image
│   ├── mask_000.png
│   ├── mask_001.png
│   └── ...
├── sam_masks_after/                    # SAM masks for after image
│   ├── mask_000.png
│   ├── mask_001.png
│   └── ...
├── objects_change_report.csv           # Object-level analysis results
├── objects_change_report.geojson       # Spatial data with changes
├── before_with_polygons.png            # Visualization overlay
├── after_with_polygons.png             # Visualization overlay
└── pixel_analysis/                     # Pixel-level analysis
    ├── pixel_change_summary.csv
    └── change_mask.png
```

## Key Functions

### `sam.py`

#### `generate_before_after_masks()`
Generates SAM instance segmentation masks for both images.

#### `run_full_pipeline()` ⭐ NEW
Main orchestrator function that runs the entire analysis workflow.

### `flood_change_pipeline.py`

#### `main()`
Performs object-level change detection:
- Image alignment and normalization
- Mask vectorization
- Semantic classification
- Change detection via IoU matching

### `geoJSON.py`

#### `overlay_geojson_on_images()`
Creates visualization overlays with color-coded polygons.

### `flood_analysis.py`

#### `run_flood_pixel_analysis()` ⭐ NEW
Performs pixel-level change detection and statistics.

## Configuration

### Environment Variables

```bash
# SAM model configuration
export SAM_MODEL_TYPE=vit_l          # vit_b, vit_l, or vit_h
export SAM_CHECKPOINT_PATH=/path/to/sam_vit_l_0b3195.pth
```

### Threshold Tuning

Edit constants in `flood_change_pipeline.py`:

```python
# Semantic thresholds
EXG_VEG_THRESHOLD = 20.0        # Vegetation detection
WATER_G_R_DIFF = 10.0            # Water detection
DARKNESS_BRIGHTNESS = 90.0       # Darkness threshold

# Spatial thresholds
MIN_MASK_AREA_PIXELS = 500       # Minimum object size
IOU_MATCH_THRESHOLD = 0.2        # Change detection sensitivity
IOU_SAME_OBJECT = 0.5            # Object identity threshold
```

## Dependencies

```bash
pip install fastapi uvicorn opencv-python pillow numpy shapely \
            rasterio geopandas scikit-image scipy pandas torch \
            segment-anything
```

## Development

### Running Pipeline Standalone

```python
from sam import run_full_pipeline

results = run_full_pipeline(
    before_image_path="before.png",
    after_image_path="after.png",
    output_dir="output",
    pixel_resolution_m=0.5  # optional
)
```

### Running Individual Steps

```python
# Step 1: SAM only
from sam import generate_before_after_masks
sam_info = generate_before_after_masks(
    before_image_path="before.png",
    after_image_path="after.png",
    output_before_dir="masks_before",
    output_after_dir="masks_after"
)

# Step 2: Object analysis only
from flood_change_pipeline import main as run_pipeline
results = run_pipeline(
    before_img_path="before.png",
    after_img_path="after.png",
    output_dir="output",
    sam_masks_before_dir="masks_before",
    sam_masks_after_dir="masks_after"
)

# Step 3: Visualization only
from geoJSON import overlay_geojson_on_images
overlay = overlay_geojson_on_images(
    before_img_path="before.png",
    after_img_path="after.png",
    geojson_path="objects_change_report.geojson",
    output_dir="output"
)

# Step 4: Pixel analysis only
from flood_analysis import run_flood_pixel_analysis
pixel_results = run_flood_pixel_analysis(
    before_path="before.png",
    after_path="after.png",
    output_dir="output"
)
```

## Error Handling

The pipeline includes comprehensive error handling:

- **Fatal errors** (SAM failure, pipeline failure) → HTTP 500
- **Warning errors** (overlay, pixel analysis) → Logged, pipeline continues
- **Validation errors** → HTTP 400

Check server logs for detailed error messages and stack traces.

## API Endpoints Reference

### `POST /analyze/`
Main analysis endpoint - runs full pipeline

### `GET /results/{request_id}`
Get summary of available results

### `GET /results/{request_id}/geojson`
Download GeoJSON file

### `GET /results/{request_id}/overlay`
Download overlay visualization

### `GET /results/{request_id}/polygons`
Get polygons as JSON (for web display)

### `POST /analyze-debug/`
Debug endpoint to inspect form data
