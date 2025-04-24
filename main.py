import sys
import os

sys.path.append('scripts')  # Füge den "scripts" Ordner zum Suchpfad hinzu

import matplotlib
matplotlib.use('Agg')  # Setzt das Backend auf 'Agg' für die Bildsicherung ohne GUI

import cv2
import numpy as np
import matplotlib.pyplot as plt
from data_loading import load_rgbd_image  # Importiere die Funktion aus dem Skript 'data_loading.py'

def main():
    # Bildnummer (z.B. 0 für '0.png', 1 für '1.png', etc.)
    image_number = 0

    # Lade das RGB-D-Bild
    rgb_image, depth_image = load_rgbd_image(image_number)

    # Ausgabe der Shapes der Bilder
    print(f"RGB-Bild Shape: {rgb_image.shape}")
    print(f"Tiefenbild Shape: {depth_image.shape}")
    
    # Erstelle den Output-Ordner, falls er nicht existiert
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Speichere das RGB-Bild und das Tiefenbild als PNG-Dateien im 'output'-Ordner
    plt.figure(figsize=(10, 8))
    
    # RGB Bild speichern
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    plt.title("RGB Image")
    plt.axis('off')

    # Tiefenbild speichern
    plt.subplot(1, 2, 2)
    plt.imshow(depth_image, cmap='gray')
    plt.title("Depth Image")
    plt.axis('off')

    # Speichere das Bild als PNG-Datei im 'output'-Ordner
    output_path = os.path.join(output_dir, 'rgbd_images_output.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Die Bilder wurden als '{output_path}' gespeichert.")

if __name__ == "__main__":
    main()
