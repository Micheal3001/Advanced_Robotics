import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Camera intrinsics (K guess)
# ----------------------------
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
# SIFT + FLANN matching
# ----------------------------
class SIFTMatcher:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=4000)

        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        cv2.setRNGSeed(0)

    def extract(self, gray):
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des

    def match(self, des1, des2, ratio=0.75):
        if des1 is None or des2 is None:
            return []

        knn = self.flann.knnMatch(des1, des2, k=2)

        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        good.sort(key=lambda x: x.distance)
        return good

    def matched_points(self, kp1, kp2, matches, max_matches=500):
        matches = matches[:max_matches]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2


# ----------------------------
# ArUco camera pose (scaled)
# ----------------------------
def aruco_camera_position(gray, K, dist, marker_length_m, dict_name="DICT_4X4_50"):
    if not hasattr(cv2, "aruco"):
        return None

    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        return None

    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    corners, ids, _ = aruco.detectMarkers(gray, dictionary)

    if ids is None or len(ids) == 0:
        return None

    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length_m, K, dist)

    rvec = rvecs[0].reshape(3)
    tvec = tvecs[0].reshape(3, 1)

    R_cm, _ = cv2.Rodrigues(rvec)

    # Camera position in marker frame (invert marker->camera)
    t_cam = -R_cm.T @ tvec
    return t_cam.flatten()  # meters


# ----------------------------
# Essential matrix pose
# ----------------------------
def estimate_pose_from_points(K, pts1, pts2, ransac_thresh=0.7):
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_thresh
    )
    if E is None or mask is None:
        raise ValueError("E failed")

    mask = mask.ravel().astype(bool)
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    if len(pts1_in) < 8:
        raise ValueError("Too few inliers")

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
    return R, t, int(mask.sum())


# ----------------------------
# Simple smoothing (moving average)
# ----------------------------
def smooth_trajectory(traj, window=9):
    if len(traj) < window or window < 3:
        return traj

    window = int(window)
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
# Plot helpers (cm)
# ----------------------------
def _axis_map_cm(traj_m):
    """Convert meters->cm and map axes to (X, Z, Y) with Y-up convention."""
    traj_cm = traj_m * 100.0
    x = traj_cm[:, 0]
    y = -traj_cm[:, 1]  # make up positive
    z = traj_cm[:, 2]
    return x, z, y


def plot_both_cm(traj_sift_m, traj_aruco_m, title, save_path=None):
    sx, sz, sy = _axis_map_cm(traj_sift_m)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(sx, sz, sy, label="SIFT (Smoothed, cm)", linewidth=2)

    if traj_aruco_m is not None and len(traj_aruco_m) > 2:
        axx, azz, ayy = _axis_map_cm(traj_aruco_m)
        ax.plot(axx, azz, ayy, "--", label="ArUco (Ground Truth, cm)", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Z (cm)")
    ax.set_zlabel("Y (Up/Down)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)

    plt.show()


def plot_separate_cm(traj_sift_m, traj_aruco_m, title="Separate Views (cm)"):
    """
    Second figure:
    Left: SIFT only
    Right: ArUco only
    (Both in cm)
    """
    fig = plt.figure(figsize=(12, 5))

    # --- SIFT ---
    ax1 = fig.add_subplot(121, projection="3d")
    sx, sz, sy = _axis_map_cm(traj_sift_m)
    ax1.plot(sx, sz, sy, label="SIFT (Smoothed, cm)", linewidth=2)
    ax1.set_title("SIFT Only (cm)")
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Z (cm)")
    ax1.set_zlabel("Y (Up/Down)")
    ax1.legend()

    # --- ArUco ---
    ax2 = fig.add_subplot(122, projection="3d")
    if traj_aruco_m is not None and len(traj_aruco_m) > 2:
        axx, azz, ayy = _axis_map_cm(traj_aruco_m)
        ax2.plot(axx, azz, ayy, "--", label="ArUco (Ground Truth, cm)", linewidth=2)
    ax2.set_title("ArUco Only (cm)")
    ax2.set_xlabel("X (cm)")
    ax2.set_ylabel("Z (cm)")
    ax2.set_zlabel("Y (Up/Down)")
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input mp4 video")
    parser.add_argument("--step", type=int, default=3, help="Compare frame i -> i+step")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor (0.5 faster)")
    parser.add_argument("--marker_size_cm", type=float, default=2.5, help="ArUco marker size in cm")
    parser.add_argument("--aruco_dict", default="DICT_4X4_50", help="ArUco dictionary")
    parser.add_argument("--smooth", type=int, default=9, help="SIFT smoothing window (odd recommended)")
    parser.add_argument("--save_plot", default="trajectory.png", help="Save plot for combined figure")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path.resolve()}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    # Read first frame
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    if args.resize != 1.0:
        frame0 = cv2.resize(frame0, None, fx=args.resize, fy=args.resize)

    gray_prev = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    h, w = gray_prev.shape[:2]

    K = build_K_from_frame_size(w, h)
    dist = np.zeros((5, 1), dtype=np.float64)

    sift = SIFTMatcher()
    kp_prev, des_prev = sift.extract(gray_prev)

    # Pose accumulation
    R_cw = np.eye(3, dtype=np.float64)
    p_w = np.zeros((3, 1), dtype=np.float64)

    traj_sift = [p_w.flatten().copy()]
    traj_aruco = []

    marker_m = args.marker_size_cm / 100.0

    prev_aruco = aruco_camera_position(gray_prev, K, dist, marker_m, dict_name=args.aruco_dict)
    if prev_aruco is not None:
        traj_aruco.append(prev_aruco)

    while True:
        # skip frames for baseline
        for _ in range(args.step):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break

        if not ret:
            break

        if args.resize != 1.0:
            frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- ArUco ground truth ---
        cur_aruco = aruco_camera_position(gray, K, dist, marker_m, dict_name=args.aruco_dict)
        if cur_aruco is not None:
            traj_aruco.append(cur_aruco)

        # --- SIFT motion ---
        kp, des = sift.extract(gray)
        matches = sift.match(des_prev, des, ratio=0.75)

        if len(matches) < 30:
            traj_sift.append(p_w.flatten().copy())
            kp_prev, des_prev = kp, des
            prev_aruco = cur_aruco
            continue

        pts1, pts2 = sift.matched_points(kp_prev, kp, matches, max_matches=500)

        try:
            R_rel, t_rel, inliers = estimate_pose_from_points(K, pts1, pts2, ransac_thresh=0.7)

            if inliers < 30:
                traj_sift.append(p_w.flatten().copy())
                kp_prev, des_prev = kp, des
                prev_aruco = cur_aruco
                continue

            # ✅ scale SIFT translation using ArUco delta
            scale = 1.0
            if prev_aruco is not None and cur_aruco is not None:
                delta_aruco = np.linalg.norm(cur_aruco - prev_aruco)  # meters
                delta_vo = np.linalg.norm(t_rel)
                if delta_vo > 1e-6:
                    scale = delta_aruco / delta_vo

            p_w = p_w + scale * (R_cw @ t_rel)
            R_cw = R_cw @ R_rel.T

            traj_sift.append(p_w.flatten().copy())

        except Exception:
            traj_sift.append(p_w.flatten().copy())

        kp_prev, des_prev = kp, des
        prev_aruco = cur_aruco

    traj_sift = np.array(traj_sift)
    traj_aruco = np.array(traj_aruco)

    traj_sift_smoothed = smooth_trajectory(traj_sift, window=args.smooth)

    print("\n✅ Done!")
    print("Estimated K:\n", K)
    print("SIFT points:", len(traj_sift_smoothed))
    print("ArUco points:", len(traj_aruco))
    print(f"Step used: {args.step} (frame i -> i+{args.step})")

    # Plot 1: Combined (SIFT vs ArUco in cm)
    plot_both_cm(
        traj_sift_smoothed,
        traj_aruco,
        title="Drone Motion in Centimeters (SIFT vs ArUco)",
        save_path=args.save_plot if args.save_plot else None
    )

    # Plot 2: Separate views (SIFT only + ArUco only, both in cm)
    plot_separate_cm(
        traj_sift_smoothed,
        traj_aruco,
        title="Drone Motion Paths (Both in cm)"
    )


if __name__ == "__main__":
    main()
