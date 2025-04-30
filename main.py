import os
import pickle
from scripts.model_utils import load_all_cad_models_and_extract_features

def main():
    cad_models_directory = 'data/cad_models'  # Verzeichnis zu den CAD-Modellen
    output_directory = 'output'  # Der Ordner, in dem die Pickle-Datei gespeichert wird
    output_file = os.path.join(output_directory, 'cad_model_features.pkl')  # Dateiname für die Pickle-Datei im output-Ordner

    # Überprüfe, ob der Output-Ordner existiert, wenn nicht, erstelle ihn
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Der Ordner '{output_directory}' wurde erstellt.")

    # Überprüfe, ob die Pickle-Datei existiert
    if os.path.exists(output_file):
        print(f"Die Datei {output_file} existiert bereits. Lade die Features...")
        # Lade die gespeicherten Keypoints und Deskriptoren
        with open(output_file, 'rb') as f:
            cad_models = pickle.load(f)
        print("Features wurden erfolgreich geladen.")
    else:
        print(f"Die Datei {output_file} existiert nicht. Extrahiere und speichere die Features...")
        # Extrahiere und speichere die Features, falls die Datei nicht existiert
        cad_models = load_all_cad_models_and_extract_features(cad_models_directory, output_file)
        print(f"Die Features wurden in {output_file} gespeichert.")

    # Hier kannst du nun fortfahren, die geladenen cad_models für das Matching oder andere Aufgaben zu verwenden
    for cad_model in cad_models:
        print(f"Verarbeite Modell: {cad_model['filename']}")

if __name__ == "__main__":
    main()
