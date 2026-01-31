import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciRot
import matplotlib.pyplot as plt


# ----------------------------
# Camera intrinsics (K)
# ----------------------------
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


# ----------------------------
# Feature extractor + matcher
# ----------------------------
class FeatureVO:
    def __init__(self, K, prefer_sift=True):
        self.K = K
        self.use_sift = False

        if prefer_sift and hasattr(cv2, "SIFT_create"):
            self.sift = cv2.SIFT_create(nfeatures=4000)
            self.use_sift = True

            # FLANN for SIFT (float descriptors)
            index_params = dict(algorithm=1, trees=5)  # KDTree
            search_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            # ORB fallback
            self.orb = cv2.ORB_create(
                nfeatures=8000,
                scaleFactor=1.2,
                nlevels=8,
                fastThreshold=7
            )
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # Make RANSAC deterministic (repeatable results)
        cv2.setRNGSeed(0)

    def extract(self, img_gray):
        if self.use_sift:
            kp, des = self.sift.detectAndCompute(img_gray, None)
        else:
            kp, des = self.orb.detectAndCompute(img_gray, None)
        return kp, des

    def match(self, des1, des2, ratio=0.75):
        if des1 is None or des2 is None:
            return []

        if self.use_sift:
            # KNN match for SIFT
            knn = self.flann.knnMatch(des1, des2, k=2)
        else:
            # KNN match for ORB
            knn = self.bf.knnMatch(des1, des2, k=2)

        good = []
        for m_n in knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < ratio * n.distance:
                good.append(m)

        good = sorted(good, key=lambda x: x.distance)
        return good

    def matched_points(self, kp1, kp2, matches, max_matches=500):
        matches = matches[:max_matches]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    def estimate_pose(self, pts1, pts2, ransac_thresh=0.7):
        # Essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=ransac_thresh
        )

        if E is None or mask is None:
            raise ValueError("Essential matrix could not be computed")

        mask = mask.ravel().astype(bool)
        pts1_in = pts1[mask]
        pts2_in = pts2[mask]

        if len(pts1_in) < 8:
            raise ValueError("Too few inliers after RANSAC")

        _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, self.K)
        return R, t, int(mask.sum())


# ----------------------------
# Optional: ArUco pose (scaled)
# ----------------------------
def aruco_camera_position(img_gray, K, dist, marker_length_m, dict_name="DICT_4X4_50"):
    if not hasattr(cv2, "aruco"):
        return None

    aruco = cv2.aruco

    if not hasattr(aruco, dict_name):
        return None

    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dict_name))
    corners, ids, _ = aruco.detectMarkers(img_gray, dictionary)

    if ids is None or len(ids) == 0:
        return None

    # pose of marker in camera coords
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length_m, K, dist)

    # Use first detected marker
    rvec = rvecs[0].reshape(3)
    tvec = tvecs[0].reshape(3, 1)

    R_cm, _ = cv2.Rodrigues(rvec)

    # Invert: camera pose in marker frame
    R_mc = R_cm.T
    t_mc = -R_cm.T @ tvec  # camera position in marker coordinates

    return t_mc.flatten()  # (x, y, z) in meters


# ----------------------------
# Trajectory from video
# ----------------------------
def trajectory_from_video(
    video_path,
    step=5,
    resize=1.0,
    prefer_sift=True,
    ratio=0.75,
    max_matches=500,
    ransac_thresh=0.7,
    max_frames=0,
    use_aruco=False,
    marker_size_cm=2.5,
    aruco_dict="DICT_4X4_50"
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")

    if resize != 1.0:
        frame0 = cv2.resize(frame0, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    h, w = gray0.shape[:2]

    K = build_K_from_frame_size(w, h)
    dist = np.zeros((5, 1), dtype=np.float64)  # assume no distortion (better if you have real calib)
    vo = FeatureVO(K, prefer_sift=prefer_sift)

    kp_prev, des_prev = vo.extract(gray0)

    # Pose accumulation
    R_cw = np.eye(3, dtype=np.float64)   # camera->world rotation
    p_w = np.zeros((3, 1), dtype=np.float64)  # camera position in world

    traj_vo = [p_w.flatten().copy()]
    traj_aruco = []  # scaled in meters (if available)

    marker_length_m = marker_size_cm / 100.0

    frame_index = 0
    used_steps = 0

    while True:
        # Skip frames for a better baseline
        for _ in range(step):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return K, np.array(traj_vo), np.array(traj_aruco)

        frame_index += step
        used_steps += 1

        if max_frames > 0 and used_steps >= max_frames:
            break

        if resize != 1.0:
            frame = cv2.resize(frame, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optional ArUco “ground truth”
        if use_aruco:
            pos_m = aruco_camera_position(gray, K, dist, marker_length_m, dict_name=aruco_dict)
            if pos_m is not None:
                traj_aruco.append(pos_m)

        kp, des = vo.extract(gray)
        matches = vo.match(des_prev, des, ratio=ratio)

        if len(matches) < 30:
            # Not enough matches → keep same pose
            traj_vo.append(p_w.flatten().copy())
            kp_prev, des_prev = kp, des
            continue

        pts1, pts2 = vo.matched_points(kp_prev, kp, matches, max_matches=max_matches)

        try:
            R_rel, t_rel, inliers = vo.estimate_pose(pts1, pts2, ransac_thresh=ransac_thresh)

            # If inliers too low, skip update
            if inliers < 30:
                traj_vo.append(p_w.flatten().copy())
                kp_prev, des_prev = kp, des
                continue

            # ---- Accumulate pose ----
            # Translation is a direction only (unit scale)
            # Convert direction from camera to world and accumulate
            p_w = p_w + (R_cw @ t_rel)

            # Update orientation
            R_cw = R_cw @ R_rel.T

            traj_vo.append(p_w.flatten().copy())

        except Exception:
            traj_vo.append(p_w.flatten().copy())

        kp_prev, des_prev = kp, des

    cap.release()
    return K, np.array(traj_vo), np.array(traj_aruco)


# ----------------------------
# Plotting
# ----------------------------
def plot_trajectory(traj_vo, traj_aruco=None, save_path=None, title="Camera Trajectory"):
    # OpenCV: X right, Y down, Z forward
    # For nicer plot: make Y "up"
    x = traj_vo[:, 0]
    y = -traj_vo[:, 1]
    z = traj_vo[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, z, y, label="VO (SIFT/ORB)")  # axis order for aesthetics

    if traj_aruco is not None and len(traj_aruco) > 0:
        xa = traj_aruco[:, 0]
        ya = traj_aruco[:, 1]
        za = traj_aruco[:, 2]
        ax.plot(xa, za, ya, "--", label="ArUco (scaled)")

    ax.set_title(title)
    ax.set_xlabel("X (Right/Left)")
    ax.set_ylabel("Z (Forward/Back)")
    ax.set_zlabel("Y (Up/Down)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Estimate drone/camera trajectory from an .mp4 video")
    parser.add_argument("--video", required=True, help="Path to input .mp4 video")
    parser.add_argument("--step", type=int, default=5, help="Process every N frames (bigger = more motion)")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor (0.5 = faster)")
    parser.add_argument("--prefer_sift", action="store_true", help="Use SIFT if available (recommended)")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test (0.7-0.8 good)")
    parser.add_argument("--max_matches", type=int, default=500, help="Max matches used per step")
    parser.add_argument("--ransac_thresh", type=float, default=0.7, help="RANSAC threshold (0.5-1.0)")
    parser.add_argument("--max_steps", type=int, default=0, help="Limit number of processed steps (0 = all)")
    parser.add_argument("--csv", default="", help="Save VO trajectory to CSV")
    parser.add_argument("--plot", default="trajectory.png", help="Save plot PNG (empty to disable saving)")
    parser.add_argument("--title", default="Drone Motion Path", help="Plot title")

    # ArUco options
    parser.add_argument("--use_aruco", action="store_true", help="Also estimate ArUco path (scaled)")
    parser.add_argument("--marker_size_cm", type=float, default=2.5, help="ArUco marker size in cm")
    parser.add_argument("--aruco_dict", default="DICT_4X4_50", help="ArUco dictionary name")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path.resolve()}")

    K, traj_vo, traj_aruco = trajectory_from_video(
        video_path=video_path,
        step=args.step,
        resize=args.resize,
        prefer_sift=args.prefer_sift,
        ratio=args.ratio,
        max_matches=args.max_matches,
        ransac_thresh=args.ransac_thresh,
        max_frames=args.max_steps,
        use_aruco=args.use_aruco,
        marker_size_cm=args.marker_size_cm,
        aruco_dict=args.aruco_dict
    )

    print("\n✅ Done!")
    print("Estimated K:\n", K)
    print("VO trajectory points:", len(traj_vo))
    if args.use_aruco:
        print("ArUco points:", len(traj_aruco))

    # Save CSV
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out, traj_vo, delimiter=",", header="x,y,z", comments="")
        print(f"📄 Saved VO trajectory CSV: {out}")

    # Plot
    save_plot = args.plot if args.plot.strip() else None
    plot_trajectory(traj_vo, traj_aruco if args.use_aruco else None, save_path=save_plot, title=args.title)
    if save_plot:
        print(f"🖼️ Saved plot: {save_plot}")


if __name__ == "__main__":
    main()
