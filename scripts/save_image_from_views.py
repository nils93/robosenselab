import trimesh
import os
import pyvista as pv
import numpy as np
import csv

from scripts.calculate_camera_positions import calculate_camera_positions

# Funktion zum Speichern des Modells aus verschiedenen Perspektiven mit angepasstem Dateinamen
def save_image_from_views(mesh, output_dir, model_name):
    # Liste der Kamerapositionen für die 6 Ansichten
    camera_positions = calculate_camera_positions(mesh)   
    #print(f"Camera positions: {camera_positions}")

    subset = "val"
    
    # Erstelle das Verzeichnis für das Modell, falls nicht vorhanden
    model_dir = os.path.join(output_dir, subset, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Erstelle die PyVista-Scene aus dem Trimesh-Mesh
    scene = pv.Plotter(off_screen=True)
    scene.add_mesh(mesh, scalars=None, rgb=True)

    # Speichern der Bilder aus jeder Perspektive
    for i, camera_pos in enumerate(camera_positions):
        # Setze die Kameraansicht
        if i == 4:  # Oben
            scene.camera_position = [camera_pos, (0, 0, 0), (0, 0, 1)]
        elif i == 5:  # Unten
            scene.camera_position = [camera_pos, (0, 0, 0), (0, 0, -1)]
        else:
            scene.camera_position = [camera_pos, (0, 0, 0), (0, 1, 0)]  # Für alle anderen Perspektiven

        # Benennung des Bildes: Du kannst das Schema hier anpassen
        view_name = ""
        if i == 0:
            view_name = "front"
        elif i == 1:
            view_name = "back"
        elif i == 2:
            view_name = "left"
        elif i == 3:
            view_name = "right"
        elif i == 4:
            view_name = "top"
        elif i == 5:
            view_name = "bottom"

        # Benennung des Bildes
        output_path = os.path.join(model_dir, f"{model_name}_{view_name}_view.png")
        
        # Rendering und Speichern des Bildes
        scene.render()
        scene.screenshot(output_path)  # Speichern des Bildes

        relative_path = os.path.join(subset, model_name, f"{model_name}_{view_name}_view.png")
        csv_path = os.path.join(output_dir, "labels.csv")
        append_label_entry(csv_path, relative_path, model_name)

def append_label_entry(csv_path, relative_path, label):
    header = ["filepath", "label"]
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([relative_path, label])


