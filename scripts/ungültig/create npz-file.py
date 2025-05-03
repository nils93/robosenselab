import os
import cv2
import numpy as np
from robosenselab.scripts.ungültig.model_utils import load_all_cad_models_and_extract_features
from robosenselab.scripts.ungültig.match import match_features
from robosenselab.scripts.ungültig.load import load_rgbd_image  # Importiere die load_rgbd_image-Funktion

def main():
    cad_models_directory = 'data/cad_models'  # Verzeichnis zu den CAD-Modellen
    output_directory = 'output'  # Der Ordner, in dem die .npz-Datei gespeichert wird
    output_file = os.path.join(output_directory, 'cad_model_features.npz')  # Dateiname für die .npz-Datei im output-Ordner

    # Überprüfe, ob der Output-Ordner existiert, wenn nicht, erstelle ihn
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Der Ordner '{output_directory}' wurde erstellt.")

    # Überprüfe, ob die .npz-Datei existiert
    if os.path.exists(output_file):
        print(f"Die Datei {output_file} existiert bereits. Lade die Features...")
        # Lade die gespeicherten Keypoints und Deskriptoren aus der .npz-Datei
        try:
            with np.load(output_file, allow_pickle=True) as data:
                cad_models = data['cad_models']
            print("Features wurden erfolgreich geladen.")
        except Exception as e:
            print(f"Fehler beim Laden der .npz-Datei: {e}")
            print(f"Die Datei {output_file} ist ungültig. Lösche sie und extrahiere neue Features...")
            os.remove(output_file)
            cad_models = load_all_cad_models_and_extract_features(cad_models_directory, output_file)
    else:
        print(f"Die Datei {output_file} existiert nicht. Extrahiere und speichere die Features...")
        # Extrahiere und speichere die Features, falls die Datei nicht existiert
        cad_models = load_all_cad_models_and_extract_features(cad_models_directory, output_file)
        print(f"Die Features wurden in {output_file} gespeichert.")


if __name__ == "__main__":
    main()