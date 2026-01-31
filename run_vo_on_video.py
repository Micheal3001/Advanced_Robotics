import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Calibration (optional)
# ----------------------------
def load_calibration(npz_path: str):
    d = np.load(npz_path)
    K = d["K"].astype(np.float64)
    dist = d["dist"].astype(np.float64).reshape(1, -1)  # (1,5) or (1,8)
    return K, dist


def build_K_from_frame_size(w, h, fx=None):
    if fx is None:
        fx = 0.9 * w
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


# ----------------------------
# ORB matching (two modes):
# 1) crossCheck baseline
# 2) KNN + ratio test improved
# ----------------------------
class ORBMatcher:
    def __init__(self, use_crosscheck=False, nfeatures=8000, ratio=0.75):
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            fastThreshold=7
        )
        self.use_crosscheck = use_crosscheck
        self.ratio = ratio

        if use_crosscheck:
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        cv2.setRNGSeed(0)

    def extract(self, gray):
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def match(self, des1, des2):
        if des1 is None or des2 is None:
            return []

        if self.use_crosscheck:
            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)
            return matches

        # KNN + ratio test
        knn = self.bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        good.sort(key=lambda m: m.distance)
        return good

    @staticmethod
    def matched_points(kp1, kp2, matches, max_matches=800):
        matches = matches[:max_matches]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2


# ----------------------------
# Pose from two frames
# ----------------------------
def estimate_relative_pose(gray1, gray2, K, dist, matcher: ORBMatcher,
                           ransac_thresh=1.0, min_matches=50, min_inliers=25):
    kp1, des1 = matcher.extract(gray1)
    kp2, des2 = matcher.extract(gray2)

    matches = matcher.match(des1, des2)
    if len(matches) < min_matches:
        return None

    pts1, pts2 = matcher.matched_points(kp1, kp2, matches, max_matches=800)
    if len(pts1) < 8:
        return None

    # Undistort -> improves geometry if you have a real calibration
    pts1u = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    pts2u = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        pts1u, pts2u, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_thresh
    )
    if E is None or mask is None:
        return None

    mask = mask.ravel().astype(bool)
    inliers = int(mask.sum())
    if inliers < min_inliers:
        return None

    pts1_in = pts1u[mask]
    pts2_in = pts2u[mask]

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
    return R, t, inliers, len(matches)


# ----------------------------
# Smoothing
# ----------------------------
def smooth_trajectory(traj, window=9):
    if window <= 1 or len(traj) < window:
        return traj
    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    out = traj.copy()

    for axis in range(3):
        x = traj[:, axis]
        x_pad = np.pad(x, (window // 2, window // 2), mode="edge")
        out[:, axis] = np.convolve(x_pad, kernel, mode="valid")
    return out


# ----------------------------
# Plot
# ----------------------------
def plot_traj(traj_m, title, save_path=None):
    traj_cm = traj_m * 100.0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(traj_cm[:, 0], traj_cm[:, 2], -traj_cm[:, 1], linewidth=2)
    ax.scatter(traj_cm[0, 0], traj_cm[0, 2], -traj_cm[0, 1], s=60, marker="o", label="START")
    ax.scatter(traj_cm[-1, 0], traj_cm[-1, 2], -traj_cm[-1, 1], s=60, marker="o", label="END")

    ax.set_title(title)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Z (cm)")
    ax.set_zlabel("Y (Up/Down)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"🖼️ Saved plot: {save_path}")

    plt.show()


# ----------------------------
# Main (video)
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="Run VO on a video and plot trajectory (for report figures).")
    p.add_argument("--video", required=True, help="Path to input video (.mp4)")
    p.add_argument("--calib", default=None, help="Optional calibration .npz (K, dist)")

    p.add_argument("--step", type=int, default=5, help="Compare frame i -> i+step")
    p.add_argument("--resize", type=float, default=1.0, help="Resize factor (0.5 faster)")
    p.add_argument("--max_frames", type=int, default=0, help="0 = no limit")

    # Matching + filtering knobs
    p.add_argument("--crosscheck", action="store_true",
                   help="Baseline matching: BFMatcher(crossCheck=True). (Default is improved KNN+ratio.)")
    p.add_argument("--min_matches", type=int, default=50)
    p.add_argument("--min_inliers", type=int, default=25)
    p.add_argument("--ransac_thresh", type=float, default=1.0)

    # Trajectory smoothing
    p.add_argument("--smooth", type=int, default=9, help="Moving-average window. 1 = no smoothing")

    # Output
    p.add_argument("--save_plot", default="vo_traj.png", help="Output plot filename")
    args = p.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path.resolve()}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    if args.resize != 1.0:
        frame0 = cv2.resize(frame0, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

    gray_prev = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    h, w = gray_prev.shape[:2]

    if args.calib:
        K, dist = load_calibration(args.calib)
        print("✅ Using calibration:", args.calib)
    else:
        K = build_K_from_frame_size(w, h)
        dist = np.zeros((1, 5), dtype=np.float64)
        print("⚠️ No calibration -> using estimated K")

    matcher = ORBMatcher(use_crosscheck=args.crosscheck, nfeatures=8000, ratio=0.75)

    # Accumulate motion (scale is unknown, so this is “relative units”)
    R_wc = np.eye(3, dtype=np.float64)
    p_w = np.zeros((3, 1), dtype=np.float64)

    traj = [p_w.flatten().copy()]

    frames_seen = 1
    updates = 0
    skipped = 0

    while True:
        # jump step frames
        frame = None
        for _ in range(args.step):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            break

        frames_seen += args.step
        if args.max_frames > 0 and frames_seen >= args.max_frames:
            break

        if args.resize != 1.0:
            frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        out = estimate_relative_pose(
            gray_prev, gray, K, dist, matcher,
            ransac_thresh=args.ransac_thresh,
            min_matches=args.min_matches,
            min_inliers=args.min_inliers
        )

        if out is None:
            skipped += 1
            traj.append(p_w.flatten().copy())
            gray_prev = gray
            continue

        R_rel, t_rel, inliers, matches_count = out

        # integrate (unscaled)
        p_w = p_w + (R_wc @ t_rel)
        R_wc = R_wc @ R_rel.T

        traj.append(p_w.flatten().copy())
        gray_prev = gray
        updates += 1

        # small periodic print (useful for screenshots)
        if updates % 10 == 0:
            print(f"[update {updates:04d}] matches={matches_count:4d}  inliers={inliers:4d}  skipped={skipped}")

    cap.release()

    traj = np.array(traj, dtype=np.float64)
    traj_sm = smooth_trajectory(traj, window=args.smooth)

    mode = "crossCheck baseline" if args.crosscheck else "KNN+ratio improved"
    print("\n✅ Done!")
    print(f"Video: {video_path.name}")
    print(f"Mode: {mode}")
    print(f"step={args.step}, min_matches={args.min_matches}, min_inliers={args.min_inliers}, smooth={args.smooth}")
    print(f"Updates: {updates}, Skipped: {skipped}, Total points: {len(traj_sm)}")

    plot_traj(
        traj_sm,
        title=f"VO Trajectory ({mode})",
        save_path=args.save_plot
    )


if __name__ == "__main__":
    main()
