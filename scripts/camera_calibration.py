import cv2
import numpy as np
import glob

def calibrate_camera(
    checkerboard=(9, 6),
    image_dir="data/camera_calibration/calibration_images/*.jpg",
    save_path="outputs/camera_calibration/camera_calibration.npz"
):
    """
    Führt die Kamerakalibrierung durch und speichert die Ergebnisse.
    """
    # 🟩 3D-Punkte vorbereiten
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    # 🟩 Arrays zum Speichern
    objpoints = []  # 3D Punkte im Weltkoordinatensystem
    imgpoints = []  # 2D Punkte im Bild

    # 🟩 Bilder laden
    images = glob.glob(image_dir)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Schachbrett finden
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Optional: Ecken visualisieren
            cv2.drawChessboardCorners(img, checkerboard, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    # 🟩 Kalibrierung
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Kameramatrix:\n", camera_matrix)
    print("Verzerrungskoeffizienten:\n", dist_coeffs)

    # 🟩 Speichern
    np.savez(save_path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Kalibrierungsergebnisse gespeichert unter: {save_path}")

# Direkt ausführbar
if __name__ == "__main__":
    calibrate_camera()
