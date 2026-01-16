import os
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciRot


# ---------------------------------------------------------
# If you don't know the real intrinsics, we estimate K
# from frame size (good enough for testing VO).
# ---------------------------------------------------------
def build_K_from_frame_size(w, h, fx=None):
    if fx is None:
        fx = 0.9 * w  # simple guess
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)


class VisualOdometry:
    def __init__(self, K):
        self.K = K

        # ✅ Better ORB settings (more features, more stable)
        self.orb = cv2.ORB_create(
            nfeatures=8000,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=7
        )

        # ✅ KNN matcher (better than crossCheck)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # ✅ Make RANSAC deterministic
        cv2.setRNGSeed(0)

    # 1) Extract features
    def extract_features(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    # 2) Match using KNN + Lowe Ratio Test
    def match_features(self, des1, des2, ratio=0.75):
        knn_matches = self.bf.knnMatch(des1, des2, k=2)

        good = []
        for m, n in knn_matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        good = sorted(good, key=lambda m: m.distance)
        return good

    # 3) Convert matches to point arrays
    def get_matched_points(self, kp1, kp2, matches, max_matches=500):
        matches = matches[:max_matches]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    # 4) Estimate pose using Essential Matrix
    def estimate_pose(self, pts1, pts2):
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.7
        )

        if E is None or mask is None:
            raise ValueError("Essential matrix could not be computed.")

        # ✅ Keep only inliers
        mask = mask.ravel().astype(bool)
        pts1_in = pts1[mask]
        pts2_in = pts2[mask]

        if len(pts1_in) < 8:
            raise ValueError("Not enough inliers after RANSAC.")

        _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, self.K)

        return R, t, len(pts1_in)

    # 5) Rotation matrix -> Euler angles
    def rotation_to_euler(self, R):
        rotation = SciRot.from_matrix(R)
        return rotation.as_euler("xyz", degrees=True)


# ---------------------------------------------------------
# Extract frames from a video into a folder
# ---------------------------------------------------------
def extract_frames(video_path: str, output_dir: str, every_n_frames: int = 1):
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎞️ Extracting frames from: {video_path}")
    print(f"📦 Saving into folder: {output_dir}")
    print(f"📌 Total frames reported: {total}")
    print(f"📌 Saving every {every_n_frames} frame(s)")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            fname = output_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Saved {saved_idx} frames.\n")
    return saved_idx


# ---------------------------------------------------------
# Run VO between every consecutive frames in a folder
# ---------------------------------------------------------
def run_vo_on_frames_folder(frames_dir: str):
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        raise RuntimeError(f"Frames folder not found: {frames_dir}")

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_files) < 2:
        raise RuntimeError("Not enough frames to run VO (need at least 2).")

    # Build K from first frame
    first = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape[:2]
    K = build_K_from_frame_size(w, h)
    vo = VisualOdometry(K)

    print("📷 Estimated Camera Intrinsics K (for testing):")
    print(K)

    print("\n============= RUNNING VISUAL ODOMETRY =============")

    for i in range(len(frame_files) - 1):
        f1 = str(frame_files[i])
        f2 = str(frame_files[i + 1])

        img1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)

        kp1, des1 = vo.extract_features(img1)
        kp2, des2 = vo.extract_features(img2)

        if des1 is None or des2 is None:
            print(f"[{i:04d}] Skipped (no descriptors found).")
            continue

        matches = vo.match_features(des1, des2)

        if len(matches) < 20:
            print(f"[{i:04d}] Skipped (too few good matches: {len(matches)}).")
            continue

        pts1, pts2 = vo.get_matched_points(kp1, kp2, matches, max_matches=500)

        try:
            R, t, inliers = vo.estimate_pose(pts1, pts2)
            rx, ry, rz = vo.rotation_to_euler(R)
            tx, ty, tz = t.flatten()

            print(f"\n[{i:04d}] {frames_dir.name}: frame_{i:06d} -> frame_{i+1:06d}")
            print(f"  Good matches: {len(matches)} | Inliers: {inliers}")
            print(f"  Translation (unscaled): tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
            print(f"  Rotation (deg):         rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f}")

        except Exception as e:
            print(f"[{i:04d}] Failed pose estimation: {e}")

    print("\n===================================================\n")


# ---------------------------------------------------------
# Run VO on exactly 2 frames
# ---------------------------------------------------------
def run_vo_on_two_frames(img1_path: str, img2_path: str):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise RuntimeError("Could not load one of the images.")

    h, w = img1.shape[:2]
    K = build_K_from_frame_size(w, h)
    vo = VisualOdometry(K)

    kp1, des1 = vo.extract_features(img1)
    kp2, des2 = vo.extract_features(img2)

    if des1 is None or des2 is None:
        raise RuntimeError("No descriptors found in one of the frames.")

    matches = vo.match_features(des1, des2)

    if len(matches) < 20:
        raise RuntimeError(f"Too few good matches after ratio test: {len(matches)}")

    pts1, pts2 = vo.get_matched_points(kp1, kp2, matches, max_matches=500)

    R, t, inliers = vo.estimate_pose(pts1, pts2)

    rx, ry, rz = vo.rotation_to_euler(R)
    tx, ty, tz = t.flatten()

    print("\n============= CAMERA MOTION (6 DoF) =============")
    print(f"Frame1: {img1_path}")
    print(f"Frame2: {img2_path}")
    print(f"\nGood matches: {len(matches)} | RANSAC inliers: {inliers}")

    print("\nTranslation (unscaled direction):")
    print(f"  tx = {tx:.4f},  ty = {ty:.4f},  tz = {tz:.4f}")

    print("\nRotation (degrees):")
    print(f"  rx = {rx:.2f}°,  ry = {ry:.2f}°,  rz = {rz:.2f}°")
    print("=================================================\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main(video_filename: str):
    video_path = Path(video_filename)
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path.resolve()}")

    out_folder = video_path.stem

    # ✅ Best: skip frames for better baseline
    extract_frames(video_path, out_folder, every_n_frames=5)

    run_vo_on_frames_folder(out_folder)


if __name__ == "__main__":
    # ✅ Example for a video:
    # main("Paris.mp4")
    # main("Cinematic.mp4")

    # ✅ Example for two frames:
    img1_path = r"archive (1)/Images/Images/132050764_0000_00_0000_P00_01.jpg"
    img2_path = r"archive (1)/Images/Images/132050765_0000_00_0000_P00_01.jpg"

    # img1_path = r"Paris/frame_000200.jpg"
    # img2_path = r"Paris/frame_000380.jpg"

    # img1_path = r"Cinematic/frame_000200.jpg"
    # img2_path = r"Cinematic/frame_000300.jpg"

    run_vo_on_two_frames(img1_path, img2_path)
