import os
import shutil
from tqdm import tqdm  # Importiere tqdm für Fortschrittsanzeige
from scripts.load_all_obj_models import load_all_obj_models
from scripts.ask_for_confirmation import ask_for_confirmation
from scripts.check_if_images_exist import check_if_images_exist
from scripts.count_augmented_images import count_augmented_images
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

    # Benutzerabfrage: Alles neu oder nur augmentierte Bilder ergänzen?
    print("\nOptionen:")
    print("[1] Alles neu erstellen (löscht train/, val/ und labels.csv)")
    print("[2] Nur augmentierte Bilder hinzufügen (bestehende Daten bleiben erhalten)")
    choice = input("Bitte Auswahl eingeben [1/2]: ").strip()

    if choice == "1":
        print("→ Vorhandene Daten werden gelöscht...")
        for subfolder in ["train", "val"]:
            subfolder_path = os.path.join(output_dir, subfolder)
            if os.path.exists(subfolder_path):
                shutil.rmtree(subfolder_path)
        csv_path = os.path.join(output_dir, "labels.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        generate_classic = True
        print("→ Vorhandene Daten wurden erfolgreich gelöscht!")
    elif choice == "2":
        print("→ Es werden nur augmentierte Ansichten ergänzt.")
        generate_classic = False
    else:
        print("Ungültige Eingabe. Verarbeitung abgebrochen.")
        return

    total_images = len(model_files) * (n_views + (6 if generate_classic else 0))
    print(f"\n{total_images} Bilder werden verarbeitet...\n")

    with tqdm(total=total_images, desc="Gesamtfortschritt", unit="Bild") as pbar:
        for obj_file in model_files:
            model_name = os.path.splitext(os.path.basename(obj_file))[0]

            if generate_classic:
                pv_mesh, _ = process_pv_mesh(obj_file, output_dir)
                num_saved = save_image_from_views(pv_mesh, output_dir, model_name)
                pbar.update(num_saved)

            trimesh_mesh, _ = process_trimesh(obj_file, output_dir)
            start_index = count_augmented_images(output_dir, model_name)

            save_augmented_views(
                trimesh_mesh,
                output_dir,
                model_name,
                n_views=n_views,
                progress_bar=pbar,
                split_ratio=0.8,
                start_index=start_index
            )

    print("\n✅ Alle Modelle wurden erfolgreich verarbeitet.")