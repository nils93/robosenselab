import cv2

def load_rgbd_image(image_number):
    """
    Lädt das RGB- und Tiefenbild basierend auf der Bildnummer.
    
    :param image_number: Die Bildnummer (0-9)
    :return: Das RGB-Bild und das Tiefenbild
    """
    rgb_image_path = f"data/rgbd_images/RGB/{image_number}.png"
    rgb_image = cv2.imread(rgb_image_path, cv2.IMREAD_COLOR)
    if rgb_image is None:
        raise FileNotFoundError(f"RGB-Bild {rgb_image_path} nicht gefunden.")

    depth_image_path = f"data/rgbd_images/Depth/{image_number}.png"
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(f"Tiefenbild {depth_image_path} nicht gefunden.")

    return rgb_image, depth_image
