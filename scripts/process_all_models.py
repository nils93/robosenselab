import os
import shutil
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scripts.load_all_obj_models import load_all_obj_models
from scripts.ask_for_confirmation import ask_for_confirmation
from scripts.check_if_images_exist import check_if_images_exist
from scripts.count_augmented_images import count_augmented_images
from scripts.process_pv_mesh import process_pv_mesh
from scripts.process_trimesh import process_trimesh
from scripts.save_image_from_views import save_image_from_views
from scripts.save_augmented_views import save_augmented_views
from scripts.process_single_model_wrapper import process_single_model_wrapper
from scripts.print_duration import print_duration

def process_all_models(model_directory, output_dir, n_views=50):
    model_files = load_all_obj_models(model_directory)

    print("🔍 Die folgenden CAD-Modelle wurden gefunden:")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i:2d}. {model_file}")

    print("\n⚙️  Optionen:")
    print("[1] Alles neu erstellen (löscht train/, val/ und labels.csv)")
    print("[2] Nur augmentierte Bilder hinzufügen (bestehende Daten bleiben erhalten)")
    choice = input("Bitte Auswahl eingeben [1/2]: ").strip()

    if choice == "1":
        print("🗑️  Vorhandene Daten werden gelöscht...")
        for subfolder in ["train", "val"]:
            subfolder_path = os.path.join(output_dir, subfolder)
            if os.path.exists(subfolder_path):
                shutil.rmtree(subfolder_path)
        csv_path = os.path.join(output_dir, "labels.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        generate_classic = True
        print("✅ Ordner und CSV wurden erfolgreich entfernt.")
    elif choice == "2":
        print("🔄 Es werden nur augmentierte Ansichten ergänzt.")
        generate_classic = False
    else:
        print("❌ Ungültige Eingabe. Verarbeitung abgebrochen.")
        return

    available_cores = cpu_count()
    print(f"\n🧠 Dein System hat {available_cores} logische CPU-Kerne.")

    while True:
        try:
            user_cores = int(input(f"💡 Wie viele Kerne sollen verwendet werden? [1-{available_cores}]: ").strip())
            if 1 <= user_cores <= available_cores:
                break
            else:
                print("❌ Ungültige Eingabe. Bitte gib eine Zahl im gültigen Bereich ein.")
        except ValueError:
            print("❌ Bitte eine ganze Zahl eingeben.")

    total_images = len(model_files) * (n_views + (6 if generate_classic else 0))
    print(f"\n🚀 Starte Verarbeitung mit {user_cores} Kern(en)...")
    print(f"📦 Anzahl der Modelle: {len(model_files)}")
    print(f"🖼️  Ansichten pro Modell: {n_views} augmentiert + {'6 klassisch' if generate_classic else '0 klassisch'}")
    print(f"📊 Erwartete Gesamtbilder: {total_images}\n")

    tasks = [(obj_file, output_dir, n_views, generate_classic) for obj_file in model_files]
    start_time = time.time()
    with Pool(processes=user_cores) as pool:
        with tqdm(total=total_images, desc="Gesamtfortschritt", unit="Bild", dynamic_ncols=True, smoothing=0.1) as pbar:
            for completed_images in pool.imap_unordered(process_single_model_wrapper, tasks, chunksize=1):
                pbar.update(completed_images)

    duration = time.time() - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    print("\n✅ Alle Modelle wurden erfolgreich verarbeitet.")
    print_duration(start_time, len(model_files), total_images)
