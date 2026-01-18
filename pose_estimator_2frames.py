import argparse
from pathlib import Path
import cv2
import numpy as np


# ============================
# Calibration loader (K, dist)
# ============================
def load_calibration(npz_path: str):
    data = np.load(npz_path)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64).reshape(1, -1)
    return K, dist


def build_K_from_frame_size(w, h, fx=None):
    """
    Fallback estimation if no calibration is provided.
    Not perfect, but works for testing.
    """
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

        # FLANN matcher for SIFT
        index_params = dict(algorithm=1, trees=5)   # KDTree
        search_params = dict(checks=80)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # deterministic behavior
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
# Pose estimation between 2 frames
# ============================
def estimate_relative_pose(img1_gray, img2_gray, K, dist, sift: SIFTMatcher,
                           min_matches=50, min_inliers=25, ransac_thresh=1.0):
    """
    Returns:
      R (3x3), t (3x1), inliers_count, matches_count
    """

    kp1, des1 = sift.extract(img1_gray)
    kp2, des2 = sift.extract(img2_gray)

    matches = sift.match(des1, des2, ratio=0.75)
    if len(matches) < min_matches:
        raise RuntimeError(f"Too few matches: {len(matches)} (need >= {min_matches})")

    pts1, pts2 = sift.matched_points(kp1, kp2, matches, max_matches=800)

    # Undistort points for better accuracy
    pts1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    pts2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_thresh
    )

    if E is None or mask is None:
        raise RuntimeError("Essential matrix estimation failed.")

    mask = mask.ravel().astype(bool)
    inliers = int(mask.sum())

    if inliers < min_inliers:
        raise RuntimeError(f"Too few inliers: {inliers} (need >= {min_inliers})")

    pts1_in = pts1[mask]
    pts2_in = pts2[mask]

    _, R, t, _ = cv2.recoverPose(E, pts1_in, pts2_in, K)

    return R, t, inliers, len(matches)


# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser(
        description="Estimate 6DoF relative pose between two images using SIFT + Essential Matrix."
    )

    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", required=True, help="Path to second image")
    parser.add_argument("--calib", default=None, help="Optional calibration file .npz (must contain K, dist)")

    # Optional tuning (good defaults)
    parser.add_argument("--min_matches", type=int, default=50)
    parser.add_argument("--min_inliers", type=int, default=25)
    parser.add_argument("--ransac_thresh", type=float, default=1.0)

    args = parser.parse_args()

    img1_path = Path(args.img1)
    img2_path = Path(args.img2)

    if not img1_path.exists():
        raise RuntimeError(f"Image 1 not found: {img1_path.resolve()}")
    if not img2_path.exists():
        raise RuntimeError(f"Image 2 not found: {img2_path.resolve()}")

    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(img2_path), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise RuntimeError("Could not load one of the images.")

    h, w = img1.shape[:2]

    # Load calibration or estimate K
    if args.calib:
        K, dist = load_calibration(args.calib)
        print("✅ Using calibration from:", args.calib)
    else:
        K = build_K_from_frame_size(w, h)
        dist = np.zeros((1, 5), dtype=np.float64)
        print("⚠️ No calibration provided -> using estimated K")

    print("\nK =\n", K)
    print("\ndist =\n", dist)

    sift = SIFTMatcher(nfeatures=6000)

    # Estimate pose
    R, t, inliers, matches_count = estimate_relative_pose(
        img1, img2,
        K, dist,
        sift,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        ransac_thresh=args.ransac_thresh
    )

    rx, ry, rz = rotmat_to_euler_xyz_deg(R)
    tx, ty, tz = t.flatten()

    # Output
    print("\n================= RESULT =================")
    print("Matches:", matches_count)
    print("Inliers:", inliers)

    print("\nRotation Matrix R:")
    print(R)

    print("\nTranslation Vector t (direction only, scale unknown):")
    print(t)

    print("\nEuler angles (deg):")
    print(f"  rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f}")

    print("\nTranslation (unit direction):")
    print(f"  tx={tx:.4f}, ty={ty:.4f}, tz={tz:.4f}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
