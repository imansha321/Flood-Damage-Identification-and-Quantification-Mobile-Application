# Pipeline Refactoring Summary

## Overview
Restructured the flood analysis pipeline to follow a clear execution flow where `server.py` only initiates the process, and `sam.py` orchestrates all the analysis steps.

## Pipeline Flow

```
server.py (API endpoint)
    ↓
sam.py (Pipeline Orchestrator)
    ↓
    ├── Step 1: SAM Mask Generation
    ├── Step 2: flood_change_pipeline (Object matching & semantic analysis)
    ├── Step 3: geoJSON (Overlay visualization)
    └── Step 4: flood_analysis (Pixel-level change detection)
```

## Changes Made

### 1. `sam.py` - New Pipeline Orchestrator
**Added Function:** `run_full_pipeline()`
- Coordinates the entire analysis workflow
- Generates SAM masks for before/after images
- Calls flood_change_pipeline for object-level analysis
- Generates GeoJSON overlays for visualization
- Runs pixel-level flood analysis
- Aggregates metrics and enriches GeoJSON with area measurements
- Returns comprehensive results dictionary

**Function Signature:**
```python
def run_full_pipeline(
    before_image_path: str,
    after_image_path: str,
    output_dir: str,
    model_type: Optional[str] = None,
    checkpoint: Optional[str] = None,
    pixel_resolution_m: Optional[float] = None,
) -> dict
```

**Returns:**
```python
{
    "message": "Full pipeline complete",
    "sam": {...},                    # SAM mask generation info
    "pipeline": {...},               # flood_change_pipeline results
    "overlay": {...},                # GeoJSON overlay visualization info
    "pixel_analysis": {...},         # Pixel-level analysis results
    "metrics": {...}                 # Aggregated metrics
}
```

### 2. `server.py` - Simplified API Handler
**Changes:**
- Removed direct imports of `flood_change_pipeline`, `geoJSON`, `flood_analysis`
- Now only imports and calls `sam.run_full_pipeline()`
- Simplified the `/analyze/` endpoint
- Moved resolution detection logic before pipeline execution
- Updated response structure to use nested results from pipeline

**Before:**
```python
# server.py manually orchestrated all steps:
sam_info = generate_before_after_masks(...)
results = run_pipeline(...)
overlay_info = overlay_geojson_on_images(...)
pixel_analysis = run_flood_pixel_analysis(...)
metrics = compute_metrics(...)
```

**After:**
```python
# server.py delegates to sam.py:
results = run_full_pipeline(
    before_image_path=before_path,
    after_image_path=after_path,
    output_dir=work_dir,
    model_type=model_type,
    checkpoint=checkpoint,
    pixel_resolution_m=effective_res,
)
```

### 3. `flood_analysis.py` - Added Main Function
**Added Function:** `run_flood_pixel_analysis()`
- Performs pixel-level change detection
- Computes change mask and statistics
- Saves results to CSV and PNG
- Returns comprehensive results dictionary

**Function Signature:**
```python
def run_flood_pixel_analysis(
    before_path: str,
    after_path: str,
    output_dir: str
) -> Dict[str, Any]
```

## Benefits

1. **Separation of Concerns:**
   - `server.py`: HTTP API layer only
   - `sam.py`: Pipeline orchestration and business logic
   - Individual modules: Specific analysis tasks

2. **Maintainability:**
   - Changes to pipeline flow only require editing `sam.py`
   - API layer remains stable and simple
   - Clear function boundaries

3. **Reusability:**
   - `run_full_pipeline()` can be called from CLI, tests, or other scripts
   - Not tied to FastAPI/web context

4. **Error Handling:**
   - Centralized error handling in pipeline orchestrator
   - Better error messages with step identification
   - Non-fatal warnings for optional steps (overlay, pixel analysis)

5. **Extensibility:**
   - Easy to add new analysis steps in `sam.py`
   - Pipeline can be configured per-request
   - Modular structure allows parallel development

## Testing the Changes

### Start the server:
```bash
cd Backend
python server.py
```

### Test the endpoint:
```bash
curl -X POST "http://localhost:8000/analyze/" \
  -F "before=@before.png" \
  -F "after=@after.png" \
  -F "pixel_resolution_m=0.5"
```

### Run pipeline directly:
```bash
cd Backend
python sam.py  # Uses default before.png and after.png
```

## File Structure After Refactoring

```
Backend/
├── server.py                 # FastAPI server (calls sam.py only)
├── sam.py                    # Pipeline orchestrator (NEW: run_full_pipeline)
├── flood_change_pipeline.py  # Object-level analysis (unchanged)
├── geoJSON.py               # Visualization (unchanged)
├── flood_analysis.py        # Pixel-level analysis (NEW: run_flood_pixel_analysis)
└── PIPELINE_REFACTORING.md  # This document
```

## Migration Notes

- **No breaking changes** to API endpoints
- Response structure remains the same (with nested objects)
- All existing functionality preserved
- Backward compatible with existing clients
