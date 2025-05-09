import cv2
import numpy as np

def extract_orb_features(image):
    """
    Extrahiert ORB-Deskriptoren aus einem Bild.
    
    :param image: Das Eingabebild (in Graustufen)
    :return: Keypoints und Deskriptoren
    """
    # Konvertiere das Bild zu Graustufen (ORB funktioniert am besten mit Graustufenbildern)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialisiere den ORB-Detektor
    orb = cv2.ORB_create()

    # Finde die Keypoints und Deskriptoren
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors

# Beispiel: Verwendung der Funktion
image = cv2.imread("path_to_image.jpg")  # Lade das Bild
keypoints, descriptors = extract_orb_features(image)  # Extrahiere die HOG-Features

# Zeige die Keypoints im Bild an (optional)
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
cv2.imshow("ORB Keypoints", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
