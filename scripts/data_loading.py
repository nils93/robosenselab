import cv2
import numpy as np
import os

def load_rgbd_image(image_number):
    """
    Lädt ein RGB-D-Bild basierend auf der Bildnummer.

    :param image_number: Bildnummer (z.B. 0 für '0.png', 1 für '1.png', etc.)
    :return: Tuple (RGB-Bild, Tiefenbild)
    """
    # Bildnummer als String formatieren
    image_filename = f"{image_number}.png"
    
    # Bestimme die Pfade basierend auf der Ordnerstruktur
    rgb_path = os.path.join("data", "rgbd_images", "RGB", image_filename)
    depth_path = os.path.join("data", "rgbd_images", "Depth", image_filename)
    
    # Lade das RGB-Bild (3 Kanäle, Farbformat)
    rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_image is None:
        raise FileNotFoundError(f"RGB-Bild nicht gefunden: {rgb_path}")

    # Lade das Tiefenbild (angenommen als 16-bit PNG oder ähnliches)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(f"Tiefenbild nicht gefunden: {depth_path}")

    return rgb_image, depth_image

# Beispielaufruf
if __name__ == "__main__":
    image_number = 0  # Hier kannst du die Bildnummer anpassen (z. B. 0 bis 9)
    rgb_image, depth_image = load_rgbd_image(image_number)
    print(f"RGB-Bild Shape: {rgb_image.shape}")
    print(f"Tiefenbild Shape: {depth_image.shape}")
