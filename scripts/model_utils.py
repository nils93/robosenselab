import os
import trimesh
import pickle
from scripts.feature_extraction import extract_orb_features

def load_all_cad_models_and_extract_features(cad_models_directory, output_file='cad_model_features.pkl'):
    """
    Lädt alle CAD-Modelle aus dem angegebenen Verzeichnis, extrahiert deren Keypoints und Deskriptoren
    und speichert diese in einer Datei.
    
    :param cad_models_directory: Pfad zum Verzeichnis, das die CAD-Modelle enthält
    :param output_file: Dateiname, in dem die Keypoints und Deskriptoren gespeichert werden
    :return: None
    """
    cad_models = []

    # Basisverzeichnis: das Verzeichnis von model_utils.py (Script-Verzeichnis)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gehe eine Ebene nach oben und dann in den Projektordner
    full_path_to_models = os.path.join(base_dir, 'data', 'cad_models')  # Erstelle den vollständigen Pfad zu 'data/cad_models'

    print(f"Vollständiger Pfad zu den CAD-Modellen: {full_path_to_models}")  # Gib den berechneten Pfad aus
    
    # Überprüfe, ob der Pfad existiert
    if not os.path.exists(full_path_to_models):
        print(f"Fehler: Der Pfad {full_path_to_models} existiert nicht.")
        return cad_models

    # Durchlaufe alle Dateien im Verzeichnis
    for cad_model_filename in os.listdir(full_path_to_models):
        cad_model_path = os.path.join(full_path_to_models, cad_model_filename)

        # Überprüfe den vollen Pfad der CAD-Datei
        print(f"Versuche, das CAD-Modell zu laden: {cad_model_path}")

        if cad_model_filename.endswith('.obj'):  # Nur .obj-Dateien berücksichtigen
            try:
                # Lade das CAD-Modell mit trimesh
                cad_model_mesh = trimesh.load_mesh(cad_model_path, force='mesh')

                # Wenn das Modell erfolgreich geladen wurde, extrahiere Keypoints und Deskriptoren
                if cad_model_mesh.is_empty:
                    print(f"Warnung: Das CAD-Modell {cad_model_filename} ist leer oder ungültig.")
                    continue

                # Erzeuge eine 2D-Projektion des Modells (wir verwenden keine 3D-Visualisierung)
                cad_model_image = cad_model_mesh.scene().save_image(resolution=(640, 480))  # Bild der 3D-Projektion des Modells

                if cad_model_image is None:
                    print(f"Warnung: Das CAD-Modell {cad_model_filename} konnte nicht als Bild projiziert werden.")
                    continue

                # Extrahiere Keypoints und Deskriptoren
                cad_keypoints, cad_descriptors = extract_orb_features(cad_model_image)

                cad_models.append({
                    'filename': cad_model_filename,
                    'keypoints': cad_keypoints,
                    'descriptors': cad_descriptors
                })
            except Exception as e:
                print(f"Fehler beim Laden des CAD-Modells {cad_model_filename}: {e}")
                continue
    
    # Speichern der Keypoints und Deskriptoren in einer Datei
    with open(output_file, 'wb') as f:
        pickle.dump(cad_models, f)
        print(f"Keypoints und Deskriptoren wurden in {output_file} gespeichert.")

    return cad_models
