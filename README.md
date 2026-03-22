# 6-DoF Camera Motion Estimation (Advanced Robotics)

This repo contains our experiments and final submission for estimating **6-DoF camera motion** between frames:
- **Two-image mode (final submission):** outputs **Rotation matrix R (3×3)** and **Translation vector t (3×1)**.
- **Video mode (experiments):** runs frame-to-frame estimation over time and plots the **3D motion path**.

See `report.pdf` for the full write-up and results.

---

## Requirements

- Python 3.9+ recommended

Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: ArUco/ChArUco requires `opencv-contrib-python` (not the basic `opencv-python`).

---

## Quick start (final submission script)

### With calibration (recommended)
```bash
python pose_estimator_2frames.py --img1 "frame_000000.jpg" --img2 "frame_000010.jpg" --calib "calibration_filtered.npz"
```

### Without calibration (fallback K from image size)
```bash
python pose_estimator_2frames.py --img1 "frame_000000.jpg" --img2 "frame_000010.jpg"
```

Output is printed to terminal:
- `Rotation Matrix R:`
- `Translation Vector t:` (direction only; scale is unknown)

---

## Reproducing the report experiments (figures)

> If your video is stored elsewhere, update the `--video` path accordingly (examples assume `Videos/Loai_aruco.mp4`).

### Fig 4 — Two-frame ORB baseline
```bash
python pose_estimator.py --img1 "frame_000000.jpg" --img2 "frame_000010.jpg"
```

### Fig 5 — First video trajectory (baseline)
```bash
python trajectory_from_video.py --video "Videos/Loai_aruco.mp4" --prefer_sift --step 5
```

### Fig 6.1 / 6.2 — Matching improvement (crossCheck → KNN + ratio)
Older / simpler:
```bash
python video_motion_path.py --video "Videos/Loai_aruco.mp4"
```

Improved:
```bash
python run_vo_on_video.py --video "Videos/Loai_aruco.mp4"
```

### Fig 7.1 / 7.2 — ORB vs SIFT (same video)
ORB:
```bash
python trajectory_from_video.py --video "Videos/Loai_aruco.mp4" --step 5
```

SIFT:
```bash
python trajectory_from_video.py --video "Videos/Loai_aruco.mp4" --prefer_sift --step 5
```

### Fig 8.1 / 8.2 — Stricter filtering (example: inlier threshold)
```bash
python run_vo_on_video.py --video "Videos/Loai_aruco.mp4" --min_inliers 15
python run_vo_on_video.py --video "Videos/Loai_aruco.mp4" --min_inliers 50
```

### Fig 9.1 / 9.2 — Smoothing
```bash
python run_vo_on_video.py --video "Videos/Loai_aruco.mp4" --smooth 1
python run_vo_on_video.py --video "Videos/Loai_aruco.mp4" --smooth 9
```

### Fig 10.1 / 10.2 — SIFT vs ArUco (scale reference)
`--use_aruco` is not needed in this script; it attempts ArUco detection automatically.
```bash
python sift_aruco_path_updated.py --video "Videos/Loai_aruco.mp4" --calib "calibration_filtered.npz" --marker_size_cm 2.5 --aruco_dict DICT_4X4_50
```

### Fig 11.1 / 11.2 — ChArUco calibration attempt (debug)
```bash
python calibrate_charuco_from_video.py --video "Videos/Loai_aruco.mp4" --dict DICT_4X4_50 --squares_x 5 --squares_y 7 --square_length 0.03 --marker_length 0.022 --every_n_frames 1 --min_charuco_corners 6 --show
```

### Fig 12 — Calibration proof + effect
Print calibration contents:
```bash
python -c "import numpy as np; d=np.load('calibration_filtered.npz'); print('K=\n', d['K']); print('\ndist=\n', d['dist'])"
```

Run final estimator with calibration:
```bash
python pose_estimator_2frames.py --img1 "frame_000000.jpg" --img2 "frame_000010.jpg" --calib "calibration_filtered.npz"
```

Run final estimator without calibration:
```bash
python pose_estimator_2frames.py --img1 "frame_000000.jpg" --img2 "frame_000010.jpg"
```

### Fig 13 — Final submission output
```bash
python pose_estimator_2frames.py --img1 "frame_000000.jpg" --img2 "frame_000010.jpg" --calib "calibration_filtered.npz"
```

---

## Repo contents (key files)

- `pose_estimator_2frames.py` — **final** 2-frame submission script (prints only R and t)
- `pose_estimator.py` — initial baseline
- `trajectory_from_video.py`, `run_vo_on_video.py`, `video_motion_path.py` — video experiments + improvements
- `sift_aruco_path_updated.py` — SIFT trajectory with ArUco reference when detected
- `calibrate_charuco_from_video.py` — calibration attempt / debugging
- `calibration_filtered.npz` — camera intrinsics (K) + distortion (dist)
- `results/` — saved plots used in the report
- `run.txt` — the exact commands we ran
