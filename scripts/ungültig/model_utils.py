import trimesh
import numpy as np
import cv2
import os


def render_views(cad_model_path, resolution=(640, 480)):
    """
    Rendert 6 Ansichten eines CAD-Modells (oben, unten, links, rechts, vorne, hinten).
    
    :param cad_model_path: Pfad zum CAD-Modell (z. B. .obj)
    :param resolution: Auflösung des gerenderten Bildes
    :return: Liste von 2D-Bildern der 6 Ansichten
    """
    cad_model_mesh = trimesh.load_mesh(cad_model_path)

    # Erstelle die 6 verschiedenen Ansichten
    views = []
    
    # Definiere die Kamerapositionen und Ausrichtungen für oben, unten, links, rechts, vorne, hinten
    # Hier definieren wir die Position und die Zielrichtung der Kamera für jede Ansicht
    camera_positions = [
        [0, 0, 2],   # Vorne
        [0, 0, -2],  # Hinten
        [0, -2, 0],  # Links
        [0, 2, 0],   # Rechts
        [2, 0, 0],   # Oben
        [-2, 0, 0]   # Unten
    ]
    
    # Zielpunkt, auf den die Kamera schaut
    target_position = [0, 0, 0]  # Mitte des Modells
    
    # Rendere jede Ansicht
    for position in camera_positions:
        # Erstelle eine Kamera mit Position und Ziel
        scene = cad_model_mesh.scene()
        scene.camera.look_at(position, target_position)

        # Speichere das Bild der Szene
        image = scene.save_image(resolution=resolution)
        views.append(image)
        
    return views


def extract_orb_features(image):
    """
    Extrahiert ORB-Keypoints und Deskriptoren aus einem Bild.
    
    :param image: Das 2D-Bild
    :return: Keypoints und Deskriptoren
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    return keypoints, descriptors


def load_all_cad_models_and_extract_features(cad_models_directory, output_file='output/cad_model_features.npz'):
    cad_models = []

    # Basisverzeichnis: das Verzeichnis von model_utils.py (Script-Verzeichnis)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Gehe eine Ebene nach oben und dann in den Projektordner
    full_path_to_models = os.path.join(base_dir, cad_models_directory)  # Erstelle den vollständigen Pfad zu 'data/cad_models'

    print(f"Vollständiger Pfad zu den CAD-Modellen: {full_path_to_models}")  # Gib den berechneten Pfad aus
    
    # Überprüfe, ob der Pfad existiert
    if not os.path.exists(full_path_to_models):
        print(f"Fehler: Der Pfad {full_path_to_models} existiert nicht.")
        return cad_models

    # Durchlaufe alle Dateien im Verzeichnis
    for cad_model_filename in os.listdir(full_path_to_models):
        cad_model_path = os.path.join(full_path_to_models, cad_model_filename)

        if not cad_model_filename.endswith('.obj'):  # Nur .obj-Dateien berücksichtigen
            continue

        print(f"Versuche, das CAD-Modell zu laden: {cad_model_path}")

        try:
            # Rendern der 6 Ansichten des CAD-Modells
            rendered_views = render_views(cad_model_path)
            
            # Extrahiere Keypoints und Deskriptoren für jede Ansicht
            cad_keypoints_all_views = []
            cad_descriptors_all_views = []
            
            for view in rendered_views:
                cad_keypoints, cad_descriptors = extract_orb_features(view)
                cad_keypoints_all_views.append(cad_keypoints)
                cad_descriptors_all_views.append(cad_descriptors)

            cad_models.append({
                'filename': cad_model_filename,
                'keypoints': cad_keypoints_all_views,
                'descriptors': cad_descriptors_all_views
            })
        except Exception as e:
            print(f"Fehler beim Laden des CAD-Modells {cad_model_filename}: {e}")
            continue

    # Speichern der Keypoints und Deskriptoren in einer .npz-Datei
    try:
        np.savez(output_file, cad_models=cad_models)
        print(f"Keypoints und Deskriptoren wurden in {output_file} gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der .npz-Datei: {e}")
    
    return cad_models
