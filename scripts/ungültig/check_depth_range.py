import numpy as np
import cv2

def check_depth_range(depth_image):
    """
    Überprüft den Bereich der Tiefenwerte im Bild.
    
    :param depth_image: Das Tiefenbild als 16-bit Array.
    :return: Minimale und maximale Tiefenwerte im Bild.
    """
    min_depth = np.min(depth_image)
    max_depth = np.max(depth_image)
    
    print(f"Minimale Tiefenwert: {min_depth}")
    print(f"Maximale Tiefenwert: {max_depth}")
    
    return min_depth, max_depth

# Lade das Tiefenbild (16-bit Tiefenbild, Werte in Millimetern)
depth_image = cv2.imread("data/rgbd_images/Depth/0.png", cv2.IMREAD_UNCHANGED)

if depth_image is None:
    raise FileNotFoundError("Tiefenbild nicht gefunden")

# Überprüfe den Bereich der Tiefenwerte
min_depth, max_depth = check_depth_range(depth_image)

# Optional: Hinweise zur Umrechnung der Tiefenwerte
if max_depth > 1000:  # Typischer Bereich für Millimeter
    print("Die Tiefenwerte scheinen in Millimetern vorzuliegen.")
else:
    print("Die Tiefenwerte scheinen in Metern vorzuliegen.")
