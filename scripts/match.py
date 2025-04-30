import cv2

def match_keypoints(descriptors_image, descriptors_model):
    """
    Vergleicht die Deskriptoren des Bildes mit denen des CAD-Modells.
    
    :param descriptors_image: Deskriptoren des Bildes
    :param descriptors_model: Deskriptoren des CAD-Modells
    :return: Übereinstimmungen
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_image, descriptors_model)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
