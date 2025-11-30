# Flood Damage Analysis System

## 🎯 Overview
Flood damage identification and quantification using before/after imagery. The backend performs SAM-based segmentation, alignment, and change detection; the Expo mobile app visualizes results and lets you download overlays, CSV, and GeoJSON.

## ✨ Features

- **Metrics**: Flooded land, vegetation loss, and affected built structures (pixels, m², hectares when resolution is known)
- **Overlays**: Color-coded polygons and pixel-change mask (before/after)
- **Mobile UI**: React Native + Expo, styled with NativeWind
- **Data Export**: CSV and GeoJSON per analysis

## 🏗️ Project Structure

### Backend (Python + FastAPI)
```
Backend/
├── server.py                 # FastAPI app exposing /analyze and results endpoints
├── algorithm.py              # Core image ops, SAM helpers, IoU, proxies
├── requirements.txt          # Python dependencies
├── sam_vit_h_4b8939.pth      # SAM checkpoint (place here)
└── uploads/                  # Per-request outputs (auto-created)
```

### Frontend (React Native + Expo)
```
Flood/
├── app/(tabs)/index.tsx      # Main analysis screen (set API_BASE here)
├── tailwind.config.js        # Tailwind/NativeWind config
├── global.css                # Tailwind directives
└── package.json              # Expo app scripts/deps
```

## 🚀 Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ (recommended for Expo SDK 54)
- Git (for segment-anything dependency)

### Backend (Windows PowerShell)
```powershell
# From repo root
python -m venv .venv; . .\.venv\Scripts\Activate.ps1
pip install -r Backend\requirements.txt

# Place SAM checkpoint in Backend\ (default name used by server.py)
#   Backend\sam_vit_h_4b8939.pth

# Run server
python server.py
```

### Frontend
```powershell
cd Flood
npm install
npx expo start
# or: npm run web | npm run android | npm run ios
```

Environment config:
```ini
# Flood/.env
API_BASE=http://<your-ip>:8000
```
The app reads `API_BASE` from `.env` via `app.config.js` (available at runtime as `Constants.expoConfig.extra.API_BASE`).

## 📖 Usage

1) In the app, pick “Before” and “After” images
2) Tap “Start Analysis” and wait for processing
3) View overlays and metrics; download CSV/GeoJSON as needed

Outputs are written to `uploads/<request_id>/` and served at `/files/<request_id>/...`.

## 🔌 API

### POST `/analyze/`
multipart/form-data with two files:
- `before`: image (png/jpg)
- `after`: image (png/jpg)
- Optional query: `pixel_resolution_m` (meters per pixel)

Example response:
```json
{
  "message": "Analysis complete",
  "request_id": "<uuid>",
  "csv_url": "/files/<uuid>/objects_change_report.csv",
  "geojson_url": "/files/<uuid>/objects_change_report.geojson",
  "overlay_info": {
    "before_overlay_url": "/files/<uuid>/before_with_polygons.png",
    "after_overlay_url": "/files/<uuid>/after_with_polygons.png"
  },
  "pixel_analysis": {
    "change_mask_url": "/files/<uuid>/pixel_change_mask.png",
    "veg_loss_pixels": 1234,
    "water_candidate_pixels": 5678,
    "veg_loss_pct": 1.23,
    "water_candidate_pct": 4.56,
    "veg_loss_m2": 612.0,
    "water_candidate_m2": 2345.0,
    "veg_loss_ha": 0.0612,
    "water_candidate_ha": 0.2345
  },
  "metrics": {
    "total_objects_before": 125,
    "total_objects_after": 140,
    "matched_objects": 92,
    "new_objects": 18,
    "deleted_objects": 5,
    "total_pixels": 2073600,
    "pixel_resolution_m": 0.5
  },
  "sam": { "before_masks": 300, "after_masks": 320 }
}
```

### GET `/results/{request_id}`
Returns quick availability summary and base files URL.

### GET `/results/{request_id}/geojson` | `/overlay` | `/polygons`
Download GeoJSON, legacy overlay image, or polygon coordinates JSON.

## 🧠 How It Works (Pseudocode)

### Server (FastAPI)
- On `POST /analyze/` with two images:
  - Validate both files exist; create a new `request_id` and folder `uploads/<request_id>/`.
  - Save files as `before.png` and `after.png`.
  - Try to determine pixel resolution:
    - Read PNG metadata (DPI) or adjacent world file; otherwise parse filename; allow `?pixel_resolution_m=` override.
  - Load RGB images; align `after` to `before` (ECC) and histogram-match to reduce brightness/color drift.
  - Load SAM checkpoint (env `SAM_CHECKPOINT_PATH` or `Backend/sam_vit_h_4b8939.pth`).
  - Run SAM to create binary masks for both images; warp/flip as needed; save to `sam_masks_before/` and `sam_masks_after/`.
  - Combine masks into label rasters; vectorize to polygons; compute per-polygon proxies (ExG, G-R, brightness).
  - Assign semantic class per polygon via thresholds (vegetation, water, possible water, bare soil).
  - Compute IoU between before/after polygons; perform Hungarian matching; tag results as `matched_same`, `possible_change`, `new_after`, or `deleted_after`.
  - Compute pixel-level change masks (vegetation loss and water candidates); save `pixel_change_mask.png` and `pixel_change_summary.csv`.
  - Render `before_with_polygons.png` and `after_with_polygons.png` overlays.
  - Write `objects_change_report.csv` and, if any geometries, `objects_change_report.geojson`.
  - Respond with JSON containing: relative URLs for CSV/GeoJSON/overlays, pixel-analysis metrics, object counts, and mask counts.
- Serve static files under `/files/<request_id>/...` and helper GET endpoints to fetch GeoJSON/summary/polygons.

### Client (Expo app)
- User picks two images (Before/After) via the image picker.
- Build `FormData` with both files; `fetch(POST, API_BASE + '/analyze/')` without setting Content-Type manually.
- Show loading spinner; wait for JSON response.
- On success:
  - Read `overlay_info` and build absolute URLs (`API_BASE + relative_path`); display before/after overlays.
  - Read `metrics` and `pixel_analysis`; if `pixel_resolution_m` is present, compute areas in m²/ha from pixel counts.
  - Show cards for objects before/after and three categories: flooded land, vegetation loss, built structures affected.
  - Provide buttons to open CSV/GeoJSON in a browser.
  - Enable overlay downloads:
    - Web: fetch image, crop white margins with canvas, trigger download.
    - Native: save image to gallery with `expo-media-library`.
- `API_BASE` is set in `Flood/app/(tabs)/index.tsx`; for real devices, use your PC’s LAN IP.

## 🎨 Legends & Classes
- Blue: Flooded water / new water candidates
- Green: Vegetation (used to quantify loss)
- Orange: Built structures affected
- Gray: No significant change / bare soil

Thresholds and heuristics are implemented in `Backend/algorithm.py` and `Backend/server.py`.

## ⚙️ Configuration
- `Flood/app/(tabs)/index.tsx`: `API_BASE` for your backend URL
- `Backend/server.py` CORS `allow_origins`: adjust for your LAN IP/ports
- Pixel resolution is auto-detected (PNG metadata/world-file/filename); you can override with `?pixel_resolution_m=...`

## 📁 Outputs per request
- `objects_change_report.csv`: object matches, IoU, semantic change, areas (px)
- `objects_change_report.geojson`: polygons with attributes
- `before_with_polygons.png`, `after_with_polygons.png`: overlays
- `pixel_change_mask.png`, `pixel_change_summary.csv`: pixel-based analysis


## 🙏 Acknowledgments
- Segment Anything (Meta AI)
- FastAPI
- Expo & NativeWind

