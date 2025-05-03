import cv2
import numpy as np

def match_features(rgb_image, cad_models):
    """
    Führt das Matching der Features zwischen einem RGB-Bild und den CAD-Modellen durch.
    
    :param rgb_image: Das RGB-Bild, das mit den CAD-Modellen verglichen werden soll
    :param cad_models: Liste der CAD-Modelle mit deren Keypoints und Deskriptoren
    :return: Liste der Übereinstimmungen für jedes CAD-Modell
    """
    # ORB-Detector und Matcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Extrahiere die Keypoints und Deskriptoren aus dem RGB-Bild
    keypoints_rgb, descriptors_rgb = orb.detectAndCompute(rgb_image, None)
    
    matches_list = []
    
    # Vergleiche das RGB-Bild mit jedem CAD-Modell
    for cad_model in cad_models:
        cad_keypoints = cad_model['keypoints']
        cad_descriptors = cad_model['descriptors']

        # Vergleiche Deskriptoren
        matches = bf.match(descriptors_rgb, cad_descriptors)
        
        # Sortiere die Übereinstimmungen nach Distanz (nähere Übereinstimmungen zuerst)
        matches = sorted(matches, key=lambda x: x.distance)

        # Speichere die Übereinstimmungen
        matches_list.append({
            'filename': cad_model['filename'],
            'matches': matches
        })
    
    return matches_list
