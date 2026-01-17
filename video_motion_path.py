import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================
# Calibration loader (K, dist)
# ============================
def load_calibration(npz_path: str):
    data = np.load(npz_path)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64).reshape(1, -1)
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


# ============================
# Rotation matrix -> Euler XYZ (deg)
# ============================
def rotmat_to_euler_xyz_deg(R):
    """
    Returns Euler angles (rx, ry, rz) in degrees using XYZ convention.
    rx = roll, ry = pitch, rz = yaw
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arctan2(-R[2, 0], sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arctan2(-R[2, 0], sy)
        rz = 0.0

    return np.degrees([rx, ry, rz])


# ============================
# SIFT Matching
# ============================
class SIFTMatcher:
    def __init__(self, nfeatures=6000):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)

        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=80)
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

    def matched_points(self, kp1, kp2, matches, max_matches=800):
        matches = matches[:max_matches]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2


# ============================
# ArUco pose (optional GT)
# ============================
def detect_aruco_camera_center(gray, K, dist, marker_length_m, dict_name="DICT_4X4_50"):
    if not hasattr(cv2, "aruco"):
        return None

    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        return None

    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))

    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = getattr(aruco, "CORNER_REFINE_SUBPIX", 1)
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None or len(ids) == 0:
        return None

    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length_m, K, dist)
    rvec = rvecs[0].reshape(3)
    tvec = tvecs[0].reshape(3, 1)

    R_cm, _ = cv2.Rodrigues(rvec)
    cam_center_marker = (-R_cm.T @ tvec).flatten()
    return cam_center_marker  # meters


# ============================
# Pose estimation between 2 frames (reusable)
# ============================
def estimate_relative_pose(img1_gray, img2_gray, K, dist, sift: SIFTMatcher,
                           min_matches=50, min_inliers=25, ransac_thresh=1.0):
    kp1, des1 = sift.extract(img1_gray)
    kp2, des2 = sift.extract(img2_gray)

    matches = sift.match(des1, des2, ratio=0.75)
    if len(matches) < min_matches:
        return None

    pts1, pts2 = sift.matched_points(kp1, kp2, matches, max_matches=800)

    pts1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    pts2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
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

    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)
    return R, t, inliers, len(matches)


# ============================
# Simple smoothing (moving average)
# ============================
def smooth_trajectory(traj, window=9):
    if len(traj) < window or window < 3:
        return traj

    if window % 2 == 0:
        window += 1

    kernel = np.ones(window) / window
    out = traj.copy()

    for axis in range(traj.shape[1]):
        x = traj[:, axis]
        x_pad = np.pad(x, (window // 2, window // 2), mode="edge")
        out[:, axis] = np.convolve(x_pad, kernel, mode="valid")

    return out


# ============================
# Plot: Motion + Rotation side-by-side (cm + degrees)
# ============================
def plot_motion_and_rotation(traj_vo_m, traj_aruco_m, euler_deg, title, save_path=None):
    vo = traj_vo_m * 100.0  # meters -> cm

    fig = plt.figure(figsize=(14, 6))

    # ---- Left: 3D motion ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")

    ax1.plot(vo[:, 0], vo[:, 2], -vo[:, 1], label="SIFT (Smoothed, cm)", linewidth=2)

    if traj_aruco_m is not None and len(traj_aruco_m) > 2:
        ar = traj_aruco_m * 100.0
        ax1.plot(ar[:, 0], ar[:, 2], -ar[:, 1], "--", label="ArUco (Ground Truth, cm)", linewidth=2)

    ax1.set_title("Drone Motion (cm)")
    ax1.set_xlabel("X (cm)")
    ax1.set_ylabel("Z (cm)")
    ax1.set_zlabel("Y (Up/Down)")
    ax1.legend()

    # ---- Right: Rotation over time ----
    ax2 = fig.add_subplot(1, 2, 2)

    steps = np.arange(len(euler_deg))
    ax2.plot(steps, euler_deg[:, 0], label="Roll (rx)", linewidth=2)
    ax2.plot(steps, euler_deg[:, 1], label="Pitch (ry)", linewidth=2)
    ax2.plot(steps, euler_deg[:, 2], label="Yaw (rz)", linewidth=2)

    ax2.set_title("Rotation (degrees)")
    ax2.set_xlabel("Step index")
    ax2.set_ylabel("Angle (deg)")
    ax2.grid(True)
    ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"🖼️ Saved plot: {save_path}")

    plt.show()


# ============================
# Main pipeline
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video (.mp4)")
    parser.add_argument("--calib", default=None, help="Optional calibration npz (K, dist)")

    parser.add_argument("--step", type=int, default=1, help="Compare frame i -> i+step")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor (0.5 faster)")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = no limit")

    parser.add_argument("--use_aruco", action="store_true", help="Enable ArUco ground truth + scale")
    parser.add_argument("--aruco_dict", default="DICT_4X4_50", help="ArUco dictionary")
    parser.add_argument("--marker_size_cm", type=float, default=2.5, help="Marker size in cm")

    parser.add_argument("--smooth", type=int, default=9, help="Smoothing window (odd better)")
    parser.add_argument("--min_matches", type=int, default=50)
    parser.add_argument("--min_inliers", type=int, default=25)
    parser.add_argument("--default_step_cm", type=float, default=1.0,
                        help="If ArUco scale missing, move this much per step (cm)")

    parser.add_argument("--save_plot", default="trajectory.png", help="Output plot image")
    args = parser.parse_args()

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

    # K/dist
    if args.calib:
        K, dist = load_calibration(args.calib)
        print("✅ Loaded calibration from:", args.calib)
    else:
        K = build_K_from_frame_size(w, h)
        dist = np.zeros((1, 5), dtype=np.float64)
        print("⚠️ No calibration provided -> using estimated K")

    print("K:\n", K)
    print("dist:\n", dist)

    sift = SIFTMatcher(nfeatures=6000)

    # trajectory state (world = first camera frame)
    R_wc = np.eye(3, dtype=np.float64)
    p_w = np.zeros((3, 1), dtype=np.float64)

    traj_vo = [p_w.flatten().copy()]
    traj_aruco = []

    # ✅ NEW: store Euler rotation angles per step
    euler_list = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]

    marker_m = args.marker_size_cm / 100.0
    prev_aruco = None
    last_scale_m = args.default_step_cm / 100.0  # fallback scale in meters

    if args.use_aruco:
        prev_aruco = detect_aruco_camera_center(gray_prev, K, dist, marker_m, args.aruco_dict)
        if prev_aruco is not None:
            traj_aruco.append(prev_aruco.copy())

    frame_count = 1
    good_updates = 0
    aruco_hits = 0

    while True:
        for _ in range(args.step):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            break

        frame_count += 1
        if args.max_frames > 0 and frame_count > args.max_frames:
            break

        if args.resize != 1.0:
            frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- ArUco position (optional) ---
        cur_aruco = None
        if args.use_aruco:
            cur_aruco = detect_aruco_camera_center(gray, K, dist, marker_m, args.aruco_dict)
            if cur_aruco is not None:
                traj_aruco.append(cur_aruco.copy())
                aruco_hits += 1

        # --- Pose estimate from two frames ---
        out = estimate_relative_pose(
            gray_prev, gray, K, dist, sift,
            min_matches=args.min_matches,
            min_inliers=args.min_inliers,
            ransac_thresh=1.0
        )

        if out is None:
            traj_vo.append(p_w.flatten().copy())
            euler_list.append(euler_list[-1].copy())  # ✅ keep last rotation
            gray_prev = gray
            prev_aruco = cur_aruco if cur_aruco is not None else prev_aruco
            continue

        R_rel, t_rel, inliers, matches_count = out

        # --- Scale using ArUco delta when possible ---
        if args.use_aruco and prev_aruco is not None and cur_aruco is not None:
            delta_aruco = float(np.linalg.norm(cur_aruco - prev_aruco))  # meters
            delta_vo = float(np.linalg.norm(t_rel))
            if delta_vo > 1e-8 and delta_aruco > 1e-8:
                measured_scale = delta_aruco / delta_vo
                if 1e-4 <= measured_scale <= 0.50:  # clamp jumps
                    last_scale_m = 0.7 * last_scale_m + 0.3 * measured_scale

        # --- Integrate motion ---
        delta_world = last_scale_m * (R_wc @ t_rel)
        p_w = p_w + delta_world

        R_wc = R_wc @ R_rel.T

        traj_vo.append(p_w.flatten().copy())

        # ✅ NEW: track Euler angles
        euler_list.append(rotmat_to_euler_xyz_deg(R_wc))

        good_updates += 1

        gray_prev = gray
        if cur_aruco is not None:
            prev_aruco = cur_aruco

    cap.release()

    traj_vo = np.array(traj_vo, dtype=np.float64)
    traj_vo_smoothed = smooth_trajectory(traj_vo, window=args.smooth)

    euler_arr = np.array(euler_list, dtype=np.float64)
    euler_smoothed = smooth_trajectory(euler_arr, window=max(5, args.smooth))  # ✅ smooth rotation too

    traj_aruco = np.array(traj_aruco, dtype=np.float64) if len(traj_aruco) > 0 else None

    print("\n✅ Finished!")
    print("Frames processed:", frame_count)
    print("Good VO updates:", good_updates)
    print("ArUco detections:", aruco_hits)
    print("Final fallback scale (cm/step):", last_scale_m * 100.0)

    plot_motion_and_rotation(
        traj_vo_smoothed,
        traj_aruco,
        euler_smoothed,
        title="Drone Motion + Rotation (SIFT + optional ArUco)",
        save_path=args.save_plot
    )


if __name__ == "__main__":
    main()
