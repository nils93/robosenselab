from ultralytics import YOLO
import os
import json

def predict_yolo(
    model="outputs/yolo_runs/train_long_run/weights/best.pt",
    source="data/predict",
    conf=0.25,
    device=0,
    save=True,
    project="outputs/yolo_runs",
    name="infer_run_long",
    line_width=1
):
    print("Starte YOLO Inference mit:")
    print(f" Modell: {model}")
    print(f" Quelle: {source}")
    print(f" Konfidenz: {conf}")
    print(f" Gerät: {device}")
    print(f" Speichern: {save}")
    print(f" Projekt: {project}, Name: {name}")
    print(f" Linien-Stärke: {line_width}")

    # Lade Modell
    yolo_model = YOLO(model)

    # Inferenz
    results = yolo_model.predict(
        source=source,
        conf=conf,
        device=device,
        save=save,
        project=project,
        name=name,
        line_width=line_width,
        verbose=False  # <- unterdrückt diese Ausgaben
    )

    # Labels als JSON speichern
    output_dir = os.path.join(project, name)
    os.makedirs(output_dir, exist_ok=True)

    CLASS_ID_TO_LABEL = {
        0: "morobot-s_Achse-1A_gray",
        1: "morobot-s_Achse-1A_yellow",
        2: "morobot-s_Achse-1B_yellow",
        3: "morobot-s_Achse-3B_gray"
    }

    for i, r in enumerate(results):
        label_data = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            if cls_id in CLASS_ID_TO_LABEL:
                label = CLASS_ID_TO_LABEL[cls_id]
            else:
                label = f"unknown_class_{cls_id}"

            label_data.append({
                "label": label,
                "bbox_modal": xyxy,
            })


        image_name = os.path.basename(r.path)
        json_name = os.path.splitext(image_name)[0] + ".json"
        json_path = os.path.join(output_dir, json_name)

        with open(json_path, 'w') as f:
            json.dump(label_data, f, indent=4)

    print("Inference abgeschlossen!")

# Direkt ausführen, wenn du das Skript standalone startest
if __name__ == "__main__":
    predict_yolo()
