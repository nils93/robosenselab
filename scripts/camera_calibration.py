import cv2
import numpy as np
import glob
import json
import os

def calibrate_camera(
    checkerboard=(9, 6),
    image_dir="data/camera_calibration/calibration_images/*.jpg",
    save_path="outputs/camera_calibration/camera_calibration.npz"
):
    """
    Führt die Kamerakalibrierung durch und speichert die Ergebnisse.
    Zusätzlich wird ein Megapose-kompatibles camera_data.json erzeugt.
    """
    print(f"Lade Bilder von: {image_dir}")
    # 3D-Punkte vorbereiten
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    # Arrays zum Speichern
    objpoints = []
    imgpoints = []

    # Bilder laden
    images = glob.glob(image_dir)

    resolution = None  # wird aus dem ersten gültigen Bild gelesen

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if resolution is None:
            h, w = gray.shape
            resolution = [h, w]

        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Ecken visualisieren
            cv2.drawChessboardCorners(img, checkerboard, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    if resolution is None:
        raise RuntimeError("Keine gültigen Bilder zur Kalibrierung gefunden.")

    # Kalibrierung
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, resolution[::-1], None, None
    )

    print("Kameramatrix:\n", camera_matrix)
    print("Verzerrungskoeffizienten:\n", dist_coeffs)
    print("Auflösung (h, w):", resolution)

    # Megapose-kompatibles JSON
    megapose_json_path = os.path.join(os.path.dirname(save_path), "camera_data.json")
    megapose_data = {
        "K": camera_matrix.tolist(),
        "resolution": resolution
    }
    with open(megapose_json_path, "w") as f:
        json.dump(megapose_data, f, indent=4)
    print(f"Done. camera_data.json gespeichert unter: {megapose_json_path}")


# Direkt ausführbar
if __name__ == "__main__":
    calibrate_camera()
