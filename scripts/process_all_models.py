import os
from tqdm import tqdm  # Importiere tqdm für Fortschrittsanzeige
from scripts.load_all_obj_models import load_all_obj_models
from scripts.save_image_from_views import save_image_from_views
from scripts.move_model_to_origin import move_model_to_origin
from scripts.ask_for_confirmation import ask_for_confirmation
from scripts.process_single_model import process_single_model

# Funktion, um alle Modelle zu verarbeiten
def process_all_models(model_directory, output_dir):
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

        # Iteriere durch die Modelle und aktualisiere die Fortschrittsanzeige
        for obj_file in tqdm(model_files, desc="Verarbeitung", unit="Modell"):
            # Verarbeite jedes Modell ohne zusätzliche Konsolenausgaben
            #process_single_model(obj_file, output_dir)
            pv_mesh, model_name = process_single_model(obj_file, output_dir)
            save_image_from_views(pv_mesh, output_dir, model_name)
    else:
        print("Verarbeitung abgebrochen.")
        return
    print("Alle Modelle wurden erfolgreich verarbeitet und die Bilder gespeichert.")
    return