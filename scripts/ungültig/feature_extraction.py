import cv2

def extract_orb_features(image):
    """
    Extrahiert ORB-Deskriptoren aus einem Bild.
    
    :param image: Das Eingabebild (in Graustufen)
    :return: Keypoints und Deskriptoren
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors
