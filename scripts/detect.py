# scripts/detect.py

from .match import match_keypoints
from .feature_extraction import extract_orb_features
import cv2

def detect_objects(rgb_image, cad_models):
    """
    Erkenne Objekte im RGB-Bild, indem die Deskriptoren des Bildes mit den Deskriptoren der CAD-Modelle verglichen werden.
    
    :param rgb_image: Das Eingabebild (RGB)
    :param cad_models: Liste der geladenen CAD-Modelle mit ihren Deskriptoren
    :return: None (Zeigt die Übereinstimmungen in einem Fenster an)
    """
    # Extrahiere ORB-Deskriptoren aus dem RGB-Bild
    keypoints_image, descriptors_image = extract_orb_features(rgb_image)

    # Vergleiche Bild-Deskriptoren mit den Deskriptoren jedes CAD-Modells
    for cad_model in cad_models:
        matches = match_keypoints(descriptors_image, cad_model['descriptors'])

        # Visualisiere die besten Matches
        image_with_matches = cv2.drawMatches(
            rgb_image, keypoints_image, cad_model['filename'], cad_model['keypoints'],
            matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Zeige das Bild mit den Matches
        cv2.imshow(f"Matches - {cad_model['filename']}", image_with_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
