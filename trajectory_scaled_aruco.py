import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_K_from_frame_size(w, h, fx=None):
    if fx is None:
        fx = 0.9 * w
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float64)


class FeatureVO:
    def __init__(self, K, prefer_sift=True):
        self.K = K
        self.use_sift = False

        if prefer_sift and hasattr(cv2, "SIFT_create"):
            self.sift = cv2.SIFT_create(nfeatures=4000)
            self.use_sift = True
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.orb = cv2.ORB_create(nfeatures=8000, fastThreshold=7)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        cv2.setRNGSeed(0)

    def extract(self, img_gray):
        if self.use_sift:
            return self.sift.detectAndCompute(img_gray, None)
        else:
            return self.orb.detectAndCompute(img_gray, None)

    def match(self, des1, des2, ratio=0.75):
        if des1 is None or des2 is None:
            return []

        knn = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
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
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
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

        _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, self.K)
        return R, t, int(mask.sum())


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

    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length_m, K, dist)

    rvec = rvecs[0].reshape(3)
    tvec = tvecs[0].reshape(3, 1)

    R_cm, _ = cv2.Rodrigues(rvec)

    # camera position in marker frame
    t_cam = -R_cm.T @ tvec
    return t_cam.flatten()


def plot_trajectory_cm(traj_vo_m, traj_aruco_m, title="Drone Motion in Centimeters"):
    # convert to cm
    vo = traj_vo_m * 100.0
    ar = traj_aruco_m * 100.0

    # nicer axis: X right/left, Z forward/back, Y up/down
    x = vo[:, 0]
    y = -vo[:, 1]
    z = vo[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, z, y, label="VO (scaled, cm)")

    if len(ar) > 0:
        xa = ar[:, 0]
        ya = -ar[:, 1]
        za = ar[:, 2]
        ax.plot(xa, za, ya, "--", label="ArUco (cm)")

    ax.set_title(title)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Z (cm)")
    ax.set_zlabel("Y (cm)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--prefer_sift", action="store_true")
    parser.add_argument("--marker_size_cm", type=float, default=2.5)
    parser.add_argument("--aruco_dict", default="DICT_4X4_50")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    h, w = gray0.shape[:2]

    K = build_K_from_frame_size(w, h)
    dist = np.zeros((5, 1), dtype=np.float64)

    vo = FeatureVO(K, prefer_sift=args.prefer_sift)

    kp_prev, des_prev = vo.extract(gray0)

    # pose accumulation (world coords)
    R_cw = np.eye(3, dtype=np.float64)
    p_w = np.zeros((3, 1), dtype=np.float64)

    traj_vo_scaled = [p_w.flatten().copy()]
    traj_aruco = []

    marker_m = args.marker_size_cm / 100.0

    # first aruco position
    prev_aruco = aruco_camera_position(gray0, K, dist, marker_m, dict_name=args.aruco_dict)

    if prev_aruco is not None:
        traj_aruco.append(prev_aruco)

    while True:
        # skip frames
        for _ in range(args.step):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                traj_vo_scaled = np.array(traj_vo_scaled)
                traj_aruco = np.array(traj_aruco)
                print("✅ Done!")
                print("K=\n", K)
                print("VO points:", len(traj_vo_scaled))
                print("ArUco points:", len(traj_aruco))
                plot_trajectory_cm(traj_vo_scaled, traj_aruco)
                return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ArUco
        cur_aruco = aruco_camera_position(gray, K, dist, marker_m, dict_name=args.aruco_dict)
        if cur_aruco is not None:
            traj_aruco.append(cur_aruco)

        # VO
        kp, des = vo.extract(gray)
        matches = vo.match(des_prev, des)

        if len(matches) < 30:
            traj_vo_scaled.append(p_w.flatten().copy())
            kp_prev, des_prev = kp, des
            prev_aruco = cur_aruco
            continue

        pts1, pts2 = vo.matched_points(kp_prev, kp, matches, max_matches=500)

        try:
            R_rel, t_rel, inliers = vo.estimate_pose(pts1, pts2)
            if inliers < 30:
                traj_vo_scaled.append(p_w.flatten().copy())
                kp_prev, des_prev = kp, des
                prev_aruco = cur_aruco
                continue

            # ✅ SCALE using ArUco if we have two aruco positions
            scale = 1.0
            if prev_aruco is not None and cur_aruco is not None:
                delta_aruco = np.linalg.norm(cur_aruco - prev_aruco)  # meters
                delta_vo = np.linalg.norm(t_rel)
                if delta_vo > 1e-6:
                    scale = delta_aruco / delta_vo

            # accumulate
            p_w = p_w + scale * (R_cw @ t_rel)
            R_cw = R_cw @ R_rel.T

            traj_vo_scaled.append(p_w.flatten().copy())

        except Exception:
            traj_vo_scaled.append(p_w.flatten().copy())

        kp_prev, des_prev = kp, des
        prev_aruco = cur_aruco


if __name__ == "__main__":
    main()
