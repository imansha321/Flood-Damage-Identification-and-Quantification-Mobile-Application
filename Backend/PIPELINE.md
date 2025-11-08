# Pipeline Summary

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
