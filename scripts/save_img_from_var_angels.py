import trimesh
import os
import pyvista as pv
import numpy as np

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