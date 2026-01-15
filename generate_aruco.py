import cv2
import numpy as np


def generate_aruco_marker(marker_id=0, side_pixels=400,
                          dictionary_name="DICT_4X4_50",
                          output_path="aruco_marker.png"):
    # Make sure ArUco module exists
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco not found. Install: pip install opencv-contrib-python"
        )

    # Map string -> OpenCV dictionary constant
    aruco = cv2.aruco
    if not hasattr(aruco, dictionary_name):
        raise ValueError(f"Unknown dictionary '{dictionary_name}'")

    dictionary_id = getattr(aruco, dictionary_name)
    dictionary = aruco.getPredefinedDictionary(dictionary_id)

    # Create marker image
    marker_img = np.zeros((side_pixels, side_pixels), dtype=np.uint8)

    # Different OpenCV versions support different generator names
    if hasattr(aruco, "generateImageMarker"):
        # Newer OpenCV
        aruco.generateImageMarker(dictionary, marker_id, side_pixels, marker_img, 1)
    else:
        # Older OpenCV uses drawMarker
        aruco.drawMarker(dictionary, marker_id, side_pixels, marker_img, 1)

    # Save
    ok = cv2.imwrite(output_path, marker_img)
    if not ok:
        raise RuntimeError(f"Failed to write image to: {output_path}")

    print(f"✅ Saved ArUco marker ID={marker_id} to: {output_path}")

    # Optional: show it
    cv2.imshow("ArUco Marker", marker_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    generate_aruco_marker(
        marker_id=23,
        side_pixels=600,
        dictionary_name="DICT_6X6_250",
        output_path="aruco_23.png"
    )
