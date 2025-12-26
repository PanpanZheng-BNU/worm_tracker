# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Worm-Tracker is a Python-based toolbox for tracking the position of multiple *C. elegans* worms in behavioral experiments. The pipeline processes video recordings to detect worms, track their movements using IoU-based tracking, and generate trajectory data.

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate worm-tracker
```

Dependencies: Python 3.9, NumPy, OpenCV, Plotly, Pandas, progressbar2, Matplotlib

## Core Architecture

The tracking pipeline consists of three main stages:

### 1. Video Processing (`detector/`)
- `concat_videos.py`: Discovers and organizes video files by subject ID from directory structure
  - Uses naming convention: `{strain}_{group}_{plate_id}.avi` (e.g., `N2_group_3.avi`)
  - Returns dictionary mapping subject IDs to sorted lists of video paths

- `detector.py`: Performs worm detection using background subtraction
  - Uses `cv2.createBackgroundSubtractorKNN()` for motion detection
  - Requires ROI center marked with red circle in `.jpg` file (e.g., `N2_group_3.jpg`)
  - Detects ROI center using `utils/detect_circle.py` (HSV color detection + Hough circles)
  - ROI is rectangular: `[roi_x ± radius, roi_y ± radius]` (not circular)
  - Contour area filter: 5-250 pixels to detect individual worms
  - Outputs per-frame CSVs with columns: `frame, x, y, w, h, cX, cY`
  - Saves `centroids.txt` containing `[roi_x, roi_y]`

### 2. Tracking (`tracker/`)
- `iou.py`: Simple IoU-based tracker implementation
  - `simple_iou_tracker(detections, t_min, sigma_iou)`: Main tracking algorithm
    - `t_min`: Minimum track length in frames (default: 10)
    - `sigma_iou`: IoU threshold for matching (default: 0.3)
  - Tracks objects by maximizing IoU between consecutive frames
  - Returns trackers with: `start_frame, end_frame, bboxes, centroids`

- `simple_tracking.py`: Orchestrates tracking process
  - `mv_centroids()`: Copies centroids.txt files to tracker output directory
  - `generate_trackers_and_long_df()`:
    - Reads per-frame CSVs and sorts by frame number
    - Runs `simple_iou_tracker()` on detections
    - Outputs `trackers.json` with all tracker data
    - Outputs `long_dfs.csv` concatenating all frame detections
    - Cleans up intermediate CSV files after processing

- `concat_trackers.py`: Advanced tracker concatenation functions
  - `find_initial()`: Identifies stable initial trackers based on frame coverage
  - `generate_summarize()`: Creates summary dataframe from trackers

### 3. Utilities (`utils/`)
- `detect_circle.py`: Detects red circles in ROI images using HSV masking
- `detect_rectangle.py`: Rectangle detection utilities
- `correcting.py`: Video correction utilities for calibration videos
- `extract_img.py`: Frame extraction from videos
- `odor_position.py`: Odor position detection utilities

## Main Pipeline

Run the complete detection and tracking pipeline:

```bash
python main.py --p2vs "/path/to/videos/" --radius 900 --date "11.04"
```

### Required Inputs
1. **Videos**: `.avi` files in the specified directory
2. **ROI markers**: Corresponding `.jpg` files with red circle marking plate center
   - Must match video naming: `N2_group_3.avi` → `N2_group_3.jpg`
   - Mark plate center with red circle in HSV range [0,100,100]-[10,255,255] or [160,100,100]-[179,255,255]

### Key Arguments
- `--p2vs`: Path to directory containing videos (REQUIRED)
- `--radius`: ROI radius in pixels (default: 900)
- `--date`: Date string to prevent name conflicts (default: current date MM.DD)
- `--pool`: Number of parallel processes (default: 4)
- `--p2det`: Detection results output path (default: ./detect_results)
- `--p2trackers`: Tracker results output path (default: ./simple_trackers_result)
- `--vis`: Show visualization during processing (default: false)
- `--img`: Save individual frame images (default: false)
- `--video`: Save annotated video output (default: false)

### Pipeline Stages
1. **Detection** (`detect_main()`):
   - Discovers all videos using naming convention
   - Processes videos in parallel (multiprocessing pool)
   - Each video generates per-frame CSVs in `{p2det}/{subj}_{date}/csv/`
   - Saves ROI center in `centroids.txt`

2. **Tracking** (`mv_centroids()` + `generate_trackers_and_long_df()`):
   - Copies centroids.txt files to tracker directories
   - Loads all per-frame CSVs and applies IoU tracking
   - Outputs to `{p2trackers}/{subj}/`:
     - `trackers.json`: All tracker trajectories
     - `long_dfs.csv`: Concatenated detection data
     - `centroids.txt`: ROI center coordinates

## Utility Scripts

### Correcting Videos
```bash
python utils/correcting.py --p2v "/path/to/video.avi"
```

### Extract Frames
```bash
python utils/extract_img.py  # (check file for specific usage)
```

## Important Conventions

### Naming Requirements
- Video files: `{strain}_{group}_{plate_id}.avi` format
- ROI marker images: Same basename as video with `.jpg` extension
- Correction videos: `{date}_correcting.avi`

### ROI Definition
- ROI center must be manually marked with red circle in `.jpg` file
- ROI is rectangular, not circular: center ± radius in both x and y directions
- Default radius: 900 pixels (adjust via `--radius` based on plate size)
- Only trajectories within ROI are tracked

### Output Structure
```
detect_results/
  {subject}_{date}/
    csv/
      {subject}_frame{N}.csv
    centroids.txt
    sample_img_{N}.jpg  # Random sample frames

simple_trackers_result/
  {subject}/
    trackers.json
    long_dfs.csv
    centroids.txt
```

### Detection Parameters
- Background subtractor: KNN with history=3500, dist2Threshold=80
- Morphological operations: 3x3 ellipse kernel for opening and closing
- Contour area threshold: 5-250 pixels (filters worm-sized objects)
- Bounding box representation: (x, y, w, h) format

### Tracking Parameters
- Minimum track length (`t_min`): 10 frames
- IoU threshold (`sigma_iou`): 0.3
- Matching strategy: Greedy matching by maximum IoU

## Code Structure Notes

- All modules use `sys.path.append()` for relative imports
- Multiprocessing used for parallel video processing
- Progress bars via `progressbar2` library
- ROI detection uses HSV color space for red circle detection
- Frame numbering starts at 1 (not 0)
- Trackers use 0-indexed arrays but store 1-indexed frame numbers
