import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciRot


# ---------------------------------------------------------
# Fallback K if you don't provide calibration
# ---------------------------------------------------------
def build_K_from_frame_size(w, h, fx=None):
    if fx is None:
        fx = 0.9 * w
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)


def load_calibration_npz(npz_path: str, target_w: int, target_h: int):
    """
    Loads K and dist from npz file and scales K if needed.
    We try to infer original resolution from cx, cy (approx).
    """
    data = np.load(npz_path)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64)

    # Infer original size from principal point (approx).
    # Works well if cx ~ w/2 and cy ~ h/2.
    old_w = int(round(2.0 * K[0, 2]))
    old_h = int(round(2.0 * K[1, 2]))

    if old_w <= 0 or old_h <= 0:
        return K, dist

    if (old_w, old_h) != (target_w, target_h):
        sx = target_w / old_w
        sy = target_h / old_h

        K_scaled = K.copy()
        K_scaled[0, 0] *= sx
        K_scaled[1, 1] *= sy
        K_scaled[0, 2] *= sx
        K_scaled[1, 2] *= sy

        return K_scaled, dist

    return K, dist


class VisualOdometry:
    def __init__(self, K, dist=None):
        self.K = K
        self.dist = dist

        # Better ORB settings
        self.orb = cv2.ORB_create(
            nfeatures=8000,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=7
        )

        # KNN matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # deterministic
        cv2.setRNGSeed(0)

    def extract_features(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    def match_features(self, des1, des2, ratio=0.75):
        knn_matches = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
        good = sorted(good, key=lambda m: m.distance)
        return good

    def get_matched_points(self, kp1, kp2, matches, max_matches=500):
        matches = matches[:max_matches]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    def undistort_points_if_needed(self, pts):
        """
        pts shape: (N,2) pixels.
        Returns undistorted pixel points (still in pixel coords).
        """
        if self.dist is None:
            return pts

        pts_u = cv2.undistortPoints(
            pts.reshape(-1, 1, 2),
            self.K,
            self.dist,
            P=self.K  # keep pixel coordinates
        )
        return pts_u.reshape(-1, 2)

    def estimate_pose(self, pts1, pts2):
        """
        pts1, pts2 are pixel coords.
        We undistort them first (if dist available).
        """
        pts1_u = self.undistort_points_if_needed(pts1)
        pts2_u = self.undistort_points_if_needed(pts2)

        E, mask = cv2.findEssentialMat(
            pts1_u, pts2_u, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.7
        )

        if E is None or mask is None:
            raise ValueError("Essential matrix could not be computed.")

        mask = mask.ravel().astype(bool)
        pts1_in = pts1_u[mask]
        pts2_in = pts2_u[mask]

        if len(pts1_in) < 8:
            raise ValueError("Not enough inliers after RANSAC.")

        _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, self.K)

        return R, t, len(pts1_in)

    def rotation_to_euler(self, R):
        rotation = SciRot.from_matrix(R)
        return rotation.as_euler("xyz", degrees=True)


def extract_frames(video_path: str, output_dir: str, every_n_frames: int = 1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n🎞️ Extracting frames from: {video_path}")
    print(f"📦 Saving into folder: {output_dir}")
    print(f"📌 Total frames: {total}")
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


def run_vo_on_two_frames(img1_path: str, img2_path: str, calib_npz: str = None):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise RuntimeError("Could not load one of the images.")

    h, w = img1.shape[:2]

    if calib_npz is not None:
        K, dist = load_calibration_npz(calib_npz, w, h)
        print("✅ Using calibration from:", calib_npz)
    else:
        K = build_K_from_frame_size(w, h)
        dist = None
        print("⚠️ Using estimated K (no calibration file provided)")

    vo = VisualOdometry(K, dist)

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

    print("\nRotation matrix R:")
    print(R)

    print("\nTranslation vector t (unscaled direction):")
    print(t)

    print("\n(Extra readable)")
    print(f"Translation: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
    print(f"Rotation:    rx={rx:.2f}°, ry={ry:.2f}°, rz={rz:.2f}°")
    print("=================================================\n")


def run_vo_on_frames_folder(frames_dir: str, calib_npz: str = None):
    frames_dir = Path(frames_dir)
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if len(frame_files) < 2:
        raise RuntimeError("Not enough frames to run VO.")

    first = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape[:2]

    if calib_npz is not None:
        K, dist = load_calibration_npz(calib_npz, w, h)
        print("✅ Using calibration from:", calib_npz)
    else:
        K = build_K_from_frame_size(w, h)
        dist = None
        print("⚠️ Using estimated K (no calibration file provided)")

    vo = VisualOdometry(K, dist)

    print("\n============= RUNNING VISUAL ODOMETRY =============")
    for i in range(len(frame_files) - 1):
        img1 = cv2.imread(str(frame_files[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(frame_files[i + 1]), cv2.IMREAD_GRAYSCALE)

        kp1, des1 = vo.extract_features(img1)
        kp2, des2 = vo.extract_features(img2)

        if des1 is None or des2 is None:
            print(f"[{i:04d}] Skipped (no descriptors found).")
            continue

        matches = vo.match_features(des1, des2)

        if len(matches) < 20:
            print(f"[{i:04d}] Skipped (too few matches: {len(matches)}).")
            continue

        pts1, pts2 = vo.get_matched_points(kp1, kp2, matches, max_matches=500)

        try:
            R, t, inliers = vo.estimate_pose(pts1, pts2)
            rx, ry, rz = vo.rotation_to_euler(R)
            tx, ty, tz = t.flatten()

            print(f"\n[{i:04d}] {frames_dir.name}: frame_{i:06d} -> frame_{i+1:06d}")
            print(f"  Good matches: {len(matches)} | Inliers: {inliers}")
            print(f"  Translation: tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
            print(f"  Rotation:    rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f}")

        except Exception as e:
            print(f"[{i:04d}] Failed pose estimation: {e}")

    print("\n===================================================\n")


def main():
    parser = argparse.ArgumentParser(description="VO with optional calibration (.npz containing K and dist)")

    parser.add_argument("--img1", type=str, help="Path to first image")
    parser.add_argument("--img2", type=str, help="Path to second image")
    parser.add_argument("--calib", type=str, default=None, help="Path to calibration_filtered.npz (K, dist)")

    parser.add_argument("--video", type=str, default=None, help="Optional video file to extract frames from")
    parser.add_argument("--step", type=int, default=5, help="Frame step when extracting from video")
    parser.add_argument("--run_folder", type=str, default=None, help="Run VO on an existing frames folder")

    args = parser.parse_args()

    if args.img1 and args.img2:
        run_vo_on_two_frames(args.img1, args.img2, args.calib)
        return

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise RuntimeError(f"Video not found: {video_path.resolve()}")

        out_folder = video_path.stem
        extract_frames(video_path, out_folder, every_n_frames=args.step)
        run_vo_on_frames_folder(out_folder, args.calib)
        return

    if args.run_folder:
        run_vo_on_frames_folder(args.run_folder, args.calib)
        return

    print("❌ No action provided.\nUse either:\n"
          "  --img1 path --img2 path [--calib calibration_filtered.npz]\n"
          "  --video video.mp4 [--calib calibration_filtered.npz]\n"
          "  --run_folder FramesFolder [--calib calibration_filtered.npz]")


if __name__ == "__main__":
    main()
