import os
import trimesh
import pyvista as pv
from scripts.save_image_from_views import save_image_from_views
from scripts.move_model_to_origin import move_model_to_origin

# Funktion, um ein einzelnes Modell zu verarbeiten
def process_pv_mesh(obj_file, output_dir):
    model_name = os.path.basename(obj_file).split('.')[0]  # Modellname ohne Erweiterung
    #print(f"\nVerarbeite Modell: {model_name}")

    # Lade das Modell mit Trimesh
    mesh = trimesh.load_mesh(obj_file)

    # Verschiebe das Modell in den Ursprung
    mesh = move_model_to_origin(mesh)

    # Um die Farben korrekt in PyVista darzustellen, sicherstellen, dass sie im [0, 1] Bereich liegen
    mesh.visual.vertex_colors = mesh.visual.vertex_colors / 255.0  # Farben in den Bereich [0, 1] konvertieren

    # PyVista Mesh erstellen
    pv_mesh = pv.wrap(mesh)

    # Setze die Farben als "point_data" in PyVista
    pv_mesh.point_data["colors"] = mesh.visual.vertex_colors  # Speichern der Farben für PyVista

    return pv_mesh, model_name  # Rückgabe von mesh und model_name
