#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def get_image_points_from_clicks(image_path, num_points):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Bild konnte nicht geladen werden: {image_path}")
        return np.array([])

    image_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Geklickt: ({x}, {y})")
            image_points.append([x, y])

    cv2.imshow("Bild", image)
    cv2.setMouseCallback("Bild", click_event)

    print(f"Bitte klicke {num_points} Punkte an. ESC zum Abbrechen.")
    while len(image_points) < num_points:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC zum Abbrechen
            print("Abgebrochen.")
            break

    cv2.destroyAllWindows()
    return np.array(image_points, dtype=np.float32)

def generate_correspondences(gt_image_paths, object_points_3d, save_file="image_points_all.npz", overwrite=False):
    if os.path.exists(save_file) and not overwrite:
        print(f"2D-Korrespondenzen existieren bereits → Lade aus '{save_file}'")
        data = np.load(save_file, allow_pickle=True)
        all_image_points = data['image_points']
    else:
        all_image_points = []
        for img_path in gt_image_paths:
            print(f"\nBild: {img_path}")
            img_points = get_image_points_from_clicks(img_path, len(object_points_3d))
            all_image_points.append(img_points)

        # Speichere als .npz (inkl. object_points_3d für Klarheit)
        np.savez(save_file, object_points=object_points_3d, image_points=all_image_points)
        print(f"2D-Korrespondenzen gespeichert in '{save_file}'")

    return all_image_points

def plot_image_points(image_points, bild_index=0):
    points = image_points[bild_index]
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='r', label='2D-Punkte')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(f"2D-Punkte Bildindex {bild_index}")
    plt.xlabel("x (Pixel)")
    plt.ylabel("y (Pixel)")
    plt.show()

if __name__ == "__main__":
    # Beispielmodell
    model = "morobot-s_Achse-1A_gray"
    model_dir = f"data/ground_truth/{model}"
    image_files = sorted(glob.glob(f"{model_dir}/*.png"))

    # Beispielhafte 3D-Objektpunkte (müssen angepasst werden!)
    object_points_3d = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0]
    ], dtype=np.float32)

    save_file = f"{model}_image_points.npz"
    all_image_points = generate_correspondences(image_files, object_points_3d, save_file=save_file, overwrite=False)

    # Punkte-Plot als Vorschau
    plot_image_points(all_image_points, bild_index=0)
