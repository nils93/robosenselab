import cv2
import numpy as np
import glob
import json
import os

def calibrate_camera(
    checkerboard=(9, 6),
    image_dir="data/camera_calibration/calibration_images/*.jpg",
    save_path="outputs/camera_calibration/camera_data.json",
    show_preview=True
):
    """
    Führt eine Kamerakalibrierung anhand von Checkerboard-Bildern durch
    und speichert das Ergebnis im MegaPose-kompatiblen Format (.json).

    Parameter:
    ----------
    checkerboard : tuple
        Anzahl der inneren Ecken (Spalten, Zeilen) im Checkerboard-Muster.
    image_dir : str
        Glob-Pfad zu den Kalibrierbildern.
    save_path : str
        Pfad zur Ausgabedatei (camera_data.json).
    show_preview : bool
        Zeigt gefundene Checkerboard-Ecken im Bildfenster an (optional).
    
    Raises:
    -------
    RuntimeError:
        Wenn keine gültigen Bilder für die Kalibrierung gefunden werden.
    """
    print(f"📷 Lade Kalibrierbilder von: {image_dir}")

    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    resolution = None

    images = glob.glob(image_dir)

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

            if show_preview:
                cv2.drawChessboardCorners(img, checkerboard, corners, ret)
                cv2.imshow('Checkerboard', img)
                cv2.waitKey(100)

    cv2.destroyAllWindows()

    if resolution is None or not objpoints:
        raise RuntimeError("❌ Keine gültigen Checkerboard-Erkennungen gefunden.")

    ret, K, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, resolution[::-1], None, None
    )

    print("✅ Kalibrierung erfolgreich.")
    print("Kameramatrix (K):\n", K)
    print("Auflösung (h, w):", resolution)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    camera_data = {
        "K": K.tolist(),
        "resolution": resolution
    }

    with open(save_path, "w") as f:
        json.dump(camera_data, f, indent=4)

    print(f"💾 camera_data.json gespeichert unter: {save_path}")


if __name__ == "__main__":
    calibrate_camera()
