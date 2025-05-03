import trimesh
import os
import pyvista as pv
import numpy as np

from scripts.load_all_obj_models import load_all_obj_models
from scripts.save_image_from_views import save_image_from_views
from scripts.move_model_to_origin import move_model_to_origin

# Funktion, um alle Modelle zu verarbeiten
def process_all_models(model_directory, output_dir):
    # Lade alle .obj-Modelle im angegebenen Verzeichnis
    model_files = load_all_obj_models(model_directory)

    # Zeige die aufgelisteten Modelle und frage nach Bestätigung
    print("Die folgenden CAD-Modelle wurden gefunden:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")

    # Benutzereingabe zur Bestätigung
    confirm = input("\nMöchten Sie fortfahren und alle Modelle verarbeiten? (yes/no): ").strip().lower()

    if confirm == "yes":
        for obj_file in model_files:
            model_name = os.path.basename(obj_file).split('.')[0]  # Modellname ohne Erweiterung
            print(f"\nVerarbeite Modell: {model_name}")

            # Lade das Modell mit Trimesh
            mesh = trimesh.load_mesh(obj_file)

            # Verschiebe das Modell in den Ursprung
            mesh = move_model_to_origin(mesh)

            # Speichern der Bilder für jedes Modell
            save_image_from_views(mesh, output_dir, model_name)
    else:
        print("Verarbeitung abgebrochen.")