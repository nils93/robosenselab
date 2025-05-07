import os
from tqdm import tqdm  # Importiere tqdm für Fortschrittsanzeige
from scripts.load_all_obj_models import load_all_obj_models
from scripts.ask_for_confirmation import ask_for_confirmation
from scripts.check_if_images_exist import check_if_images_exist
from scripts.process_pv_mesh import process_pv_mesh
from scripts.process_trimesh import process_trimesh
from scripts.save_image_from_views import save_image_from_views
from scripts.save_augmented_views import save_augmented_views

# Funktion, um alle Modelle zu verarbeiten
def process_all_models(model_directory, output_dir, n_views=50):
    # Lade alle .obj-Modelle im angegebenen Verzeichnis
    model_files = load_all_obj_models(model_directory)

    # Zeige die aufgelisteten Modelle und frage nach Bestätigung
    print("Die folgenden CAD-Modelle wurden gefunden:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")

    # Benutzereingabe zur Bestätigung
    if ask_for_confirmation():
        # Fortschrittsanzeige mit tqdm, ohne dass zusätzliche Ausgaben für jedes Modell erfolgen
        print(f"Verarbeite {len(model_files)} Modelle...")

        # Zähler für die verarbeiteten Modelle
        processed_models = 0

        # Filtere Modelle heraus, bei denen Bilder bereits existieren
        models_to_process = [model_file for model_file in model_files if not check_if_images_exist(output_dir, os.path.basename(model_file).split('.')[0])]

        n_classic_views = 6  # Anzahl der klassischen Ansichten (vorne, hinten, links, rechts, oben, unten)
        total_images = len(models_to_process) * (n_classic_views + n_views) # Gesamtbildanzahl
        print(f"{total_images} Bilder werden gespeichert...")

        # Fortschrittsanzeige für nur die verarbeiteten Modelle
        with tqdm(total=total_images, desc="Gesamtfortschritt", unit="Bild") as pbar:
            for obj_file in models_to_process:
                model_name = os.path.basename(obj_file).split('.')[0]  # Modellname ohne Erweiterung

                # Verarbeite jedes Modell und erhalte das mesh und den model_name
                pv_mesh, model_name = process_pv_mesh(obj_file, output_dir)
                
                # Speichern der Bilder für das Modell
                save_image_from_views(pv_mesh, output_dir, model_name)

                pbar.update(n_classic_views)  # klassische Perspektiven: 6 Bilder

                trimesh_mesh, model_name = process_trimesh(obj_file, output_dir)

                # Speichern der augmentierten Ansichten
                save_augmented_views(trimesh_mesh, output_dir, model_name, n_views=n_views, progress_bar=pbar, split_ratio=0.8)
                
                # Fortschritt aktualisieren
                processed_models += 1

        print("Alle Modelle wurden erfolgreich verarbeitet und die Bilder gespeichert.")
    else:
        print("Verarbeitung abgebrochen.")
        return
