from ultralytics import YOLO

def predict_yolo(
    model="outputs/yolo_runs/train_long_run/weights/best.pt",
    source="data/predict",
    conf=0.25,
    device=0,
    save=True,
    save_txt=True,
    project="outputs/yolo_runs",
    name="infer_run_long",
    line_width=1
):
    print("Starte YOLO Inference mit:")
    print(f" Modell: {model}")
    print(f" Quelle: {source}")
    print(f" Konfidenz: {conf}")
    print(f" Gerät: {device}")
    print(f" Speichern: {save}, Save TXT: {save_txt}")
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
        save_txt=save_txt,
        project=project,
        name=name,
        line_width=line_width
        #font_size=font_size
    )

    print("Inference abgeschlossen!")

# Direkt ausführen, wenn du das Skript standalone startest
if __name__ == "__main__":
    predict_yolo()
