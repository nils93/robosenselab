import trimesh
import os
import pyvista as pv
import numpy as np

# Funktion zum Verschieben des Modells in den Ursprung
def move_model_to_origin(mesh):
    center = mesh.centroid  # Berechne den Mittelpunkt des Modells
    mesh.apply_translation(-center)  # Verschiebe das Modell zum Ursprung
    return mesh

# Funktion zum Speichern des Modells aus verschiedenen Perspektiven mit angepasstem Dateinamen
def save_image_from_views(mesh, output_dir, model_name):
    # Liste der Kamerapositionen für die 6 Ansichten
    camera_positions = [
        (0.0, 0.0, -300),  # Vorne
        (0.0, 0.0, 300),   # Hinten
        (-300, 0.0, 0.0),  # Links
        (300, 0.0, 0.0),   # Rechts
        (0.0, 300, 0.0),   # Oben
        (0.0, -300, 0.0),  # Unten
    ]
    
    # Erstelle das Verzeichnis für das Modell, falls nicht vorhanden
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Erstelle die PyVista-Scene aus dem Trimesh-Mesh
    scene = pv.Plotter(off_screen=True)
    scene.add_mesh(mesh)

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

        print(f"Bild gespeichert: {output_path}")

# Funktion zum Laden des Modells mit Trimesh
def load_obj_with_trimesh(filename):
    mesh = trimesh.load_mesh(filename)
    return mesh

# Hauptfunktion zum Laden und Speichern des Modells
def main(obj_file, output_dir):
    # Lade das Modell mit Trimesh
    model_name = os.path.basename(obj_file).split('.')[0]  # Extrahiere den Modellnamen ohne Erweiterung
    mesh = load_obj_with_trimesh(obj_file)

    # Verschiebe das Modell in den Ursprung
    mesh = move_model_to_origin(mesh)

    # Speichern der Bilder aus 6 verschiedenen Perspektiven
    save_image_from_views(mesh, output_dir, model_name)

if __name__ == "__main__":
    obj_file = 'data/cad_models/morobot-s_Achse-1A_gray.obj'  # Ersetze dies mit dem Pfad zu deiner .obj-Datei
    output_dir = './output_images'  # Zielverzeichnis für die Bilder
    os.makedirs(output_dir, exist_ok=True)

    main(obj_file, output_dir)
    print("Bilder aus verschiedenen Perspektiven gespeichert.")