import os
import cv2
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from scripts.camera_to_quaternion import camera_to_quaternion
from scripts.model import MultiTaskCNN

def eval_pose(image_name="0.png"):

    # ======= Parameter =======
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "rgbd_images"))
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cnn_pose_model.pt"))

    # ======= Dateinamen abfragen =======
    bild_name = input("🖼️  Bildname eingeben (z. B. 0.png): ").strip()

    rgb_path = os.path.join(DATA_DIR, "RGB", bild_name)
    depth_path = os.path.join(DATA_DIR, "Depth", bild_name)

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        raise FileNotFoundError(f"❌ Bild nicht gefunden:\n- {rgb_path}\n- {depth_path}")


    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)


    K = np.array([[616.7415, 0.0, 324.8176],
                [0.0, 616.9197, 238.0456],
                [0.0, 0.0, 1.0]])

    CLASS_NAMES = sorted([
        "morobot-s_Achse-1A_gray", "morobot-s_Achse-1A_yellow", "morobot-s_Achse-1B_gray", "morobot-s_Achse-1B_yellow",
        "morobot-s_Achse-2A_gray", "morobot-s_Achse-2A_yellow", "morobot-s_Achse-2B_gray", "morobot-s_Achse-2B_yellow",
        "morobot-s_Achse-3A_gray", "morobot-s_Achse-3A_yellow", "morobot-s_Achse-3A-rrr_gray", "morobot-s_Achse-3A-rrr_yellow",
        "morobot-s_Achse-3B_gray", "morobot-s_Achse-3B_yellow", "morobot-s_Achse-3B-rrr_gray", "morobot-s_Achse-3B-rrr_yellow",
        "morobot-s_Linearachse_gray", "morobot-s_Linearachse_yellow", "morobot-s_Zahnrad_gray", "morobot-s_Zahnrad_yellow"
    ])

    # ======= Modell laden =======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskCNN(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # ======= Transforms =======
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # Dummy-Vorschlag: Teile das Bild in Zellen (Grid) und klassifiziere jede
    H, W, _ = rgb.shape
    step = 128
    detections = []

    for y in range(0, H, step):
        for x in range(0, W, step):
            crop = rgb[y:y+step, x:x+step]
            if crop.shape[0] < 128 or crop.shape[1] < 128:
                continue  # Rand überspringen

            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                class_out, pose_out = model(input_tensor)
                _, pred = torch.max(class_out, 1)


            label = CLASS_NAMES[pred.item()]

            # Werte direkt aus dem Netzwerk
            center = pose_out[0, :3].cpu().numpy().tolist()
            quat   = pose_out[0, 3:].cpu().numpy().tolist()


            detections.append({
                "label": label,
                "bbox": [x, y, x + step, y + step],
                "position": center,
                "quaternion": quat
            })


    # ======= Ausgabe =======
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rgb, det["label"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imshow("Erkennung", rgb)
    cv2.waitKey(1)
    input("🔚 Drücke Enter, um das Programm zu beenden...")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    eval_pose()

