import argparse
from pathlib import Path
import cv2
import numpy as np


def get_aruco_dictionary(dict_name: str):
    """
    dict_name example: DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, ...
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module not found. Install: opencv-contrib-python")

    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        raise ValueError(f"Unknown dictionary '{dict_name}'. Example: DICT_4X4_50")

    return aruco.getPredefinedDictionary(getattr(aruco, dict_name))


def create_charuco_board(squares_x, squares_y, square_length, marker_length, dictionary):
    """
    OpenCV has 2 APIs depending on version:
    - CharucoBoard_create(...)
    - CharucoBoard((sx, sy), ...)
    """
    aruco = cv2.aruco
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(
            squaresX=squares_x,
            squaresY=squares_y,
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=dictionary,
        )
    # Newer OpenCV API
    return aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics K from a ChArUco board video"
    )

    parser.add_argument("--video", required=True, help="Path to .mp4 video")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary name (must match your printed board)")

    # ChArUco board layout (MUST match your printed board!)
    parser.add_argument("--squares_x", type=int, default=5, help="Number of chessboard squares in X direction")
    parser.add_argument("--squares_y", type=int, default=7, help="Number of chessboard squares in Y direction")

    # Real dimensions (choose ONE unit and keep it consistent!)
    # Recommended: meters
    parser.add_argument("--square_length", type=float, default=0.03, help="Chessboard square length (meters)")
    parser.add_argument("--marker_length", type=float, default=0.022, help="ArUco marker side length (meters)")

    # Video processing
    parser.add_argument("--every_n_frames", type=int, default=5, help="Use 1 frame out of every N frames")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor (0.5 = faster)")
    parser.add_argument("--min_charuco_corners", type=int, default=20, help="Minimum detected ChArUco corners to accept a frame")

    # Output
    parser.add_argument("--out", default="camera_calib_charuco.npz", help="Output file (.npz) to save K and dist")

    # Optional preview
    parser.add_argument("--show", action="store_true", help="Show detection preview while running")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise RuntimeError(f"Video not found: {video_path.resolve()}")

    dictionary = get_aruco_dictionary(args.dict)
    board = create_charuco_board(
        args.squares_x,
        args.squares_y,
        args.square_length,
        args.marker_length,
        dictionary,
    )

    aruco = cv2.aruco

    # Detector params (good defaults)
    detector_params = aruco.DetectorParameters()
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, detector_params)
    else:
        detector = None  # old API fallback

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    all_charuco_corners = []
    all_charuco_ids = []

    frame_idx = 0
    used_frames = 0
    accepted_frames = 0

    image_size = None

    print("\n🎥 Starting ChArUco calibration from video...")
    print(f"Video: {video_path}")
    print(f"Board: squares_x={args.squares_x}, squares_y={args.squares_y}")
    print(f"Sizes: square_length={args.square_length} m, marker_length={args.marker_length} m")
    print(f"Using every {args.every_n_frames} frames")
    print("--------------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.every_n_frames != 0:
            continue

        used_frames += 1

        if args.resize != 1.0:
            frame = cv2.resize(frame, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (w, h)

        # ---- Detect markers ----
        if detector is not None:
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=detector_params)

        if ids is None or len(ids) == 0:
            if args.show:
                cv2.imshow("ChArUco detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue

        # ---- Interpolate ChArUco corners ----
        # This gives you subpixel chessboard corner positions
        ok, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board
        )

        if charuco_ids is None or len(charuco_ids) < args.min_charuco_corners:
            if args.show:
                vis = frame.copy()
                aruco.drawDetectedMarkers(vis, corners, ids)
                cv2.imshow("ChArUco detection", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue

        accepted_frames += 1
        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)

        if args.show:
            vis = frame.copy()
            aruco.drawDetectedMarkers(vis, corners, ids)
            aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
            cv2.putText(vis, f"Accepted frames: {accepted_frames}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("ChArUco detection", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    print("\n✅ Finished scanning video.")
    print(f"Frames checked: {used_frames}")
    print(f"Frames accepted (enough ChArUco corners): {accepted_frames}")

    if accepted_frames < 10:
        raise RuntimeError(
            "Not enough valid frames for calibration. "
            "Try recording slower, closer, with more angles, and increase lighting."
        )

    # ---- Calibrate camera ----
    print("\n🧮 Calibrating camera...")

    camera_matrix_init = None
    dist_init = None

    # Some OpenCV versions have calibrateCameraCharucoExtended
    if hasattr(aruco, "calibrateCameraCharucoExtended"):
        retval, K, dist, rvecs, tvecs, stdIn, stdEx, perViewErr = aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=camera_matrix_init,
            distCoeffs=dist_init
        )
    else:
        retval, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=camera_matrix_init,
            distCoeffs=dist_init
        )

    print("\n================ CALIBRATION RESULT ================")
    print(f"Reprojection error: {retval:.6f}")
    print("\nCamera Matrix K:")
    print(K)
    print("\nDistortion Coefficients (k1,k2,p1,p2,k3...):")
    print(dist.ravel())
    print("====================================================\n")

    # Save results
    out_path = Path(args.out)
    np.savez(out_path, K=K, dist=dist, image_size=np.array(image_size))
    print(f"💾 Saved calibration to: {out_path.resolve()}")
    print("You can load it later using: data=np.load(file); K=data['K']; dist=data['dist']")


if __name__ == "__main__":
    main()
