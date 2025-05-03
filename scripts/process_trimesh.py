import os
import trimesh
from scripts.move_model_to_origin import move_model_to_origin

# Funktion, um ein einzelnes Modell zu verarbeiten
def process_trimesh(obj_file, output_dir):
    model_name = os.path.basename(obj_file).split('.')[0]  # Modellname ohne Erweiterung

    # Lade das Modell mit Trimesh
    mesh = trimesh.load_mesh(obj_file)

    # Verschiebe das Modell in den Ursprung
    mesh = move_model_to_origin(mesh)

    # Sicherstellen, dass Farben im [0, 1]-Bereich liegen (optional, aber hilfreich)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        mesh.visual.vertex_colors = mesh.visual.vertex_colors / 255.0

    return mesh, model_name  # 🔁 Jetzt: Rückgabe des Trimesh-Objekts
