import trimesh
import os
import pyvista as pv
import numpy as np
from scripts.save_image_from_views import save_image_from_views
from scripts.render_obj import render_obj
from scripts.move_model_to_origin import move_model_to_origin

# Funktion zum Laden und Vorbereiten des 3D-Modells mit Trimesh
def load_obj_with_trimesh(filename, output_dir, model_name):
    # Lade das Modell mit Trimesh
    mesh = trimesh.load_mesh(filename)

    # Verschiebe das Modell in den Ursprung
    mesh = move_model_to_origin(mesh)

    # Überprüfe, ob Farben im Modell definiert sind
    if mesh.visual.vertex_colors is not None:
        print(f"Farben wurden für das Modell geladen: {mesh.visual.vertex_colors}")
    else:
        print("Keine Farben im Modell gefunden!")

    # Um die Farben korrekt in PyVista darzustellen, sicherstellen, dass sie im [0, 1] Bereich liegen
    mesh.visual.vertex_colors = mesh.visual.vertex_colors / 255.0  # Farben in den Bereich [0, 1] konvertieren

    # PyVista Mesh erstellen
    pv_mesh = pv.wrap(mesh)
    
    # Setze die Farben als "point_data" in PyVista
    pv_mesh.point_data["colors"] = mesh.visual.vertex_colors  # Speichern der Farben für PyVista

    # Speichern der Bilder aus verschiedenen Perspektiven
    save_image_from_views(pv_mesh, output_dir, model_name)

    return pv_mesh

# Main Loop
def main(obj_file, output_dir, model_name):
    # Lade das Modell mit Trimesh und setze die Farbe in PyVista, speichere die Bilder
    pv_mesh = load_obj_with_trimesh(obj_file, output_dir, model_name)

    # Rendern des Modells (optional)
    #render_obj(pv_mesh)

if __name__ == "__main__":
    obj_file = 'data/cad_models/morobot-s_Achse-1A_yellow.obj'  # Ersetze dies mit dem Pfad zu deiner .obj-Datei
    output_dir = './output_images'  # Verzeichnis zum Speichern der Bilder
    model_name = os.path.basename(obj_file).split('.')[0]  # Modellname ohne Erweiterung
    main(obj_file, output_dir, model_name)
