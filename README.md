# Flood Damage Analysis System

## 🎯 Overview

This enhanced flood damage identification and quantification system provides comprehensive analysis of flood impact through satellite/aerial imagery. The system uses AI-powered segmentation (SAM) combined with semantic classification to detect and quantify flood damage across three main categories.

## ✨ Key Features

### 📊 Comprehensive Metrics
- **Flooded Land Area**: Total area covered by new water presence
- **Vegetation Loss**: Area where greenery was destroyed
- **Built Structures Affected**: Infrastructure impacted by flooding

All metrics provided in:
- Pixel counts
- Square meters (m²)
- Hectares (ha)

### 🎨 Color-Coded Visualization
- 🔵 **Blue**: Flooded areas
- 🟢 **Green**: Vegetation loss
- 🟠 **Orange**: Built structures affected
- ⚪ **Gray**: No significant change

### 🖼️ Segmentation Overlays
- Side-by-side before/after comparison
- Color-coded polygons showing change types
- Transparent overlays to see original imagery

### 📱 Modern Mobile UI
- Built with React Native + Expo
- Styled with NativeWind (Tailwind CSS)
- Responsive cards with gradients
- Professional presentation

## 🏗️ Architecture

### Backend (Python + FastAPI)
```
Backend/
├── server.py                 # FastAPI server with /analyze/ endpoint
├── sam.py                    # SAM mask generation + full pipeline
├── flood_change_pipeline.py  # Object matching & semantic analysis
├── geoJSON.py               # Overlay generation with color coding
└── flood_analysis.py        # Pixel-level analysis
```

### Frontend (React Native + Expo)
```
Flood/
├── app/
│   └── (tabs)/
│       └── index.tsx        # Main analysis screen
├── babel.config.js          # NativeWind configuration
├── metro.config.js          # Metro bundler with CSS support
├── tailwind.config.js       # Tailwind configuration
├── global.css              # Tailwind directives
└── nativewind-env.d.ts     # TypeScript definitions
```

## 🚀 Getting Started

### Prerequisites

# Backend
python 3.8+
pip install fastapi uvicorn opencv-python pillow numpy shapely rasterio geopandas scikit-image scipy pandas segment-anything torch torchvision

# Frontend

Node.js 16+
npm 
Expo CLI


### Installation

#### Backend Setup

cd Backend

# Install dependencies
pip install -r requirements.txt

# Download SAM checkpoint

# Place sam_vit_l_0b3195.pth in Backend/ directory

# Start server
python server.py


#### Frontend Setup

cd Flood

# Install dependencies
npm install

# Start Expo development server
npm run web

npx expo start


## 📖 Usage

### 1. Upload Images
- Tap "📷 Before" to select pre-flood image
- Tap "📷 After" to select post-flood image

### 2. Run Analysis
- Tap "🔍 Start Analysis" button
- Wait for processing (typically 30-60 seconds)

### 3. View Results
The app displays:
- **Object counts**: Before/after comparison
- **Flooded land area**: Blue-coded with area measurements
- **Vegetation loss**: Green-coded with area measurements
- **Built structures affected**: Orange-coded with area measurements
- **Color legend**: Visual guide to overlay colors
- **Segmentation overlays**: Side-by-side visualizations

### 4. Export Data
- Tap "📍 GeoJSON" to download spatial data
- Tap "📄 CSV Report" to download tabular data

## 🎨 Color Coding System

### Semantic Classifications
The system classifies each detected object as:
- **VEGETATION**: Green areas (ExG > 20)
- **WATER**: Water bodies (G-R > 10, brightness < 90)
- **POSSIBLE_WATER**: Water candidates
- **BARE_SOIL**: Bare ground, roads, buildings

### Change Detection Colors

| Change Type | Color | RGB | Description |
|------------|-------|-----|-------------|
| New Water | 🔵 Blue | (0, 100, 255) | Areas now flooded |
| Vegetation Loss | 🟢 Green | (50, 200, 0) | Lost greenery |
| Structures Affected | 🟠 Orange | (255, 140, 0) | Buildings/infrastructure flooded |
| No Change | ⚪ Gray | (180, 180, 180) | Stable areas |

## 📊 API Response Format

```json
{
  "message": "Analysis complete",
  "request_id": "uuid-string",
  "csv_url": "/files/uuid/objects_change_report.csv",
  "geojson_url": "/files/uuid/objects_change_report.geojson",
  "overlay_info": {
    "before_overlay_url": "/files/uuid/before_with_polygons.png",
    "after_overlay_url": "/files/uuid/after_with_polygons.png"
  },
  "metrics": {
    "total_objects_before": 125,
    "total_objects_after": 140,
    "vegetation_loss_pixels": 83000,
    "new_water_pixels": 125000,
    "built_structures_affected_pixels": 42000,
    "total_flooded_pixels": 125000,
    "pixel_resolution_m": 0.5,
    "vegetation_loss_area_m2": 20750.0,
    "vegetation_loss_area_ha": 2.075,
    "new_water_area_m2": 31250.0,
    "new_water_area_ha": 3.125,
    "built_structures_affected_area_m2": 10500.0,
    "built_structures_affected_area_ha": 1.05,
    "total_flooded_area_m2": 31250.0,
    "total_flooded_area_ha": 3.125
  }
}
```

## 🔧 Configuration

### Backend Thresholds
Edit in `flood_change_pipeline.py`:
```python
EXG_VEG_THRESHOLD = 20.0      # Excess Green threshold
WATER_G_R_DIFF = 10.0         # Green-Red difference
DARKNESS_BRIGHTNESS = 90.0    # Brightness threshold
MIN_MASK_AREA_PIXELS = 500    # Minimum object size
IOU_MATCH_THRESHOLD = 0.2     # Object matching IoU
```

### Frontend API Base URL
Edit in `Flood/app/(tabs)/index.tsx`:
```typescript
const API_BASE = 'http://localhost:8000'; // Change for production
```

### Tailwind Colors
Edit in `Flood/tailwind.config.js`:
```javascript
theme: {
  extend: {
    colors: {
      primary: '#0ea5e9',    // Blue
      secondary: '#16a34a',  // Green
      accent: '#8b5cf6',     // Purple
      danger: '#ef4444',     // Red
    },
  },
}
```

## 📁 Output Files

### GeoJSON
Contains polygon geometries with attributes:
- `id`: Object identifier
- `semantic_change`: Change classification
- `area_before_px`, `area_after_px`: Pixel areas
- `area_before_m2`, `area_after_m2`: Metric areas
- `area_before_ha`, `area_after_ha`: Hectare areas
- `iou`: Intersection over Union score
- `status`: Match status

### CSV Report
Tabular format with columns:
- before_id, after_id
- iou, status
- area_before_px, area_after_px
- pct_change
- semantic_change
- RGB proxy values (ExG, G-R)

### Overlay Images
- `before_with_polygons.png`: Overlays on before image
- `after_with_polygons.png`: Color-coded overlays on after image

## 🧪 Testing

### Test with Sample Data
```bash
# Place test images in Backend/
cp path/to/before.png Backend/before.png
cp path/to/after.png Backend/after.png

# Run standalone pipeline
cd Backend
python sam.py
```

### Expected Output
```
[STEP 1/4] Running SAM mask generation...
[STEP 2/4] Running flood change pipeline...
[STEP 3/4] Generating GeoJSON overlay visualization...
[STEP 4/4] Running pixel-level flood analysis...
[SUCCESS] Full pipeline completed successfully!
```

## 🐛 Troubleshooting

### Backend Issues

**SAM checkpoint not found**
```bash
# Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints
# Place in Backend/ directory
export SAM_CHECKPOINT_PATH=/path/to/sam_vit_l_0b3195.pth
```

**CUDA/GPU errors**
```python
# System will automatically fallback to CPU
# For GPU support, install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Frontend Issues

**NativeWind not working**
```bash
# Clear cache and restart
npx expo start --clear

# Verify babel.config.js has nativewind preset
```

**Images not uploading**
```bash
# Check CORS settings in server.py
# Verify API_BASE URL is correct
# Check network connectivity
```

## 🙏 Acknowledgments

- **SAM (Segment Anything Model)**: Meta AI Research
- **FastAPI**: Modern Python web framework
- **Expo**: React Native development platform
- **NativeWind**: Tailwind CSS for React Native

