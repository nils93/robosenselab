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
    """
    Führt eine Inferenz mit einem trainierten YOLO-Modell durch und speichert
    die Ergebnisse als JSON-Dateien mit Bounding Boxes im MegaPose-kompatiblen Format.

    Für jedes Bild wird eine JSON-Datei erstellt mit folgendem Format:
    [
        {
            "label": "<KLASSENNAME>",
            "bbox_modal": [x1, y1, x2, y2]
        },
        ...
    ]

    Parameter:
    ----------
    model : str
        Pfad zur YOLOv8-Modellgewichtedatei (.pt).
    source : str
        Pfad zum Eingabeverzeichnis (oder Video/Bild).
    conf : float
        Konfidenzschwelle für die Detektion (0–1).
    device : int
        CUDA-Gerätenummer (z. B. 0) oder -1 für CPU.
    save : bool
        Ob Ergebnisbilder mit Bounding Boxes gespeichert werden sollen.
    project : str
        Zielverzeichnis für YOLO-Inferenzläufe.
    name : str
        Unterordnername innerhalb von 'project' zur Ablage.
    line_width : int
        Dicke der gezeichneten Bounding Boxes (nur bei Bildspeicherung relevant).

    Hinweise:
    ---------
    - Klassennamen werden per `CLASS_ID_TO_LABEL`-Mapping zugeordnet.
    - Nur Bounding Boxes, keine Segmente oder Keypoints.
    - Die resultierenden JSON-Dateien sind vorbereitet für MegaPose-Workflows.
    """
    print("Starte YOLO Inference mit:")
    print(f" Modell: {model}")
    print(f" Quelle: {source}")
    print(f" Konfidenz: {conf}")
    print(f" Gerät: {device}")
    print(f" Speichern: {save}")
    print(f" Projekt: {project}, Name: {name}")
    print(f" Linien-Stärke: {line_width}")

    # Lade YOLOv8-Modell
    yolo_model = YOLO(model)

    # Inferenz durchführen
    results = yolo_model.predict(
        source=source,
        conf=conf,
        device=device,
        save=save,
        project=project,
        name=name,
        line_width=line_width,
        verbose=False
    )

    # Zielverzeichnis sicherstellen
    output_dir = os.path.join(project, name)
    os.makedirs(output_dir, exist_ok=True)

    # Mapping: Klassen-ID zu Labelnamen
    CLASS_ID_TO_LABEL = {
        0: "morobot-s_Achse-1A_gray",
        1: "morobot-s_Achse-1A_yellow",
        2: "morobot-s_Achse-1B_yellow",
        3: "morobot-s_Achse-3B_gray"
    }

    # Ergebnisse pro Bild verarbeiten
    for r in results:
        label_data = []
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            label = CLASS_ID_TO_LABEL.get(cls_id, f"unknown_class_{cls_id}")

            label_data.append({
                "label": label,
                "bbox_modal": xyxy,
            })

        image_name = os.path.basename(r.path)
        json_name = os.path.splitext(image_name)[0] + ".json"
        json_path = os.path.join(output_dir, json_name)

        with open(json_path, 'w') as f:
            json.dump(label_data, f, indent=4)

    print("✅ Inference abgeschlossen und JSON-Dateien gespeichert.")

if __name__ == "__main__":
    predict_yolo()
