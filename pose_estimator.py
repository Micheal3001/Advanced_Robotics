import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciRot
import sys
import os


# ---------------------------------------------------------
# NOTE: Replace these with your real drone camera intrinsics
# ---------------------------------------------------------
K = np.array([
    [700,   0, 320],
    [  0, 700, 240],
    [  0,   0,   1]
], dtype=np.float64)


class VisualOdometry:
    def __init__(self, K):
        self.K = K
        self.orb = cv2.ORB_create(5000)               # feature detector
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)  # brute-force matcher

    # ----------------------------
    # 1. Extract ORB keypoints
    # ----------------------------
    def extract_features(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des

    # ----------------------------
    # 2. Match ORB features
    # ----------------------------
    def match_features(self, des1, des2):
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)
        return matches

    # ---------------------------------------------------------------
    # 3. Convert matched keypoints to pixel coordinate arrays
    # ---------------------------------------------------------------
    def get_matched_points(self, kp1, kp2, matches, max_matches=200):
        matches = matches[:max_matches]  # keep best matches only
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        return pts1, pts2

    # ---------------------------------------------------------------
    # 4. Compute Essential Matrix + Recover Camera Pose
    # ---------------------------------------------------------------
    def estimate_pose(self, pts1, pts2):
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            threshold=1.0,
            prob=0.999
        )

        if E is None:
            raise ValueError("Essential matrix could not be computed.")

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t

    # ----------------------------
    # 5. Convert R → Euler angles
    # ----------------------------
    def rotation_to_euler(self, R):
        rotation = SciRot.from_matrix(R)
        euler = rotation.as_euler("xyz", degrees=True)
        return euler  # rx, ry, rz


# ----------------------------------------------------------------------
# MAIN PROGRAM
# ----------------------------------------------------------------------
def main(img1_path, img2_path):

    # Load grayscale images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        print(f"Checked path 1: {os.path.abspath(img1_path)}")
        print(f"Checked path 2: {os.path.abspath(img2_path)}")
        return

    vo = VisualOdometry(K)

    # 1. Extract features
    kp1, des1 = vo.extract_features(img1)
    kp2, des2 = vo.extract_features(img2)

    # 2. Match features
    matches = vo.match_features(des1, des2)

    # 3. Keep best matches only
    pts1, pts2 = vo.get_matched_points(kp1, kp2, matches)

    # 4. Estimate pose (Rotation + Translation)
    R, t = vo.estimate_pose(pts1, pts2)

    # 5. Convert rotation matrix → Euler angles
    rx, ry, rz = vo.rotation_to_euler(R)
    tx, ty, tz = t.flatten()

    # ---------------------------------------------
    # PRINT FINAL 6 DOF RESULT
    # ---------------------------------------------
    print("\n============= CAMERA MOTION (6 DoF) =============")
    print(f"Translation (meters, unscaled):")
    print(f"  tx = {tx:.4f},  ty = {ty:.4f},  tz = {tz:.4f}")

    print("\nRotation (degrees):")
    print(f"  rx = {rx:.2f}°,  ry = {ry:.2f}°,  rz = {rz:.2f}°")
    print("==================================================\n")


if __name__ == "__main__":
    # Hardcoded paths (raw strings)
    # img1_path = r"archive (1)/Images/Images/132050002_0000_00_0000_P00_03.jpg"
    # img2_path = r"archive (1)/Images/Images/132050003_0000_00_0000_P00_03.jpg"

    img1_path = r"Paris/frame_000350.jpg"
    img2_path = r"Paris/frame_000380.jpg"


    main(img1_path, img2_path)
