import subprocess

def train_yolo(
    model="yolo11s.pt",
    data_yaml="yolo/data.yaml",
    epochs=500,
    imgsz=640,
    batch=8,
    device=0,
    project="outputs/yolo_runs",
    name="train_long_run"
):
    """
    Startet das Training eines YOLO-Modells über das CLI-Interface von Ultralytics YOLO (v8).

    Diese Funktion ruft intern den `yolo`-Kommandozeilenbefehl auf, um ein Training
    im `detect`-Modus zu starten. Sie eignet sich für automatisierte Pipelines.

    Parameter:
    ----------
    model : str
        Pfad zu einem Pretrained-Modell oder Modellnamen (z. B. 'yolov8s.pt' oder 'yolo11s.pt').
    data_yaml : str
        Pfad zur YOLO-kompatiblen .yaml-Datei mit Trainings- und Validierungsdaten.
    epochs : int
        Anzahl der Trainings-Epochen.
    imgsz : int
        Bildgröße (quadratisch) in Pixeln für Training und Validierung.
    batch : int
        Batch-Größe.
    device : int
        Index der verwendeten GPU (z. B. 0) oder -1 für CPU.
    project : str
        Pfad zum Projektordner für Ausgaben.
    name : str
        Name des Unterordners innerhalb von 'project', in dem das Training gespeichert wird.

    Hinweise:
    ---------
    - Voraussetzung: `yolo`-CLI muss im Systempfad verfügbar sein (z. B. über `pip install ultralytics`).
    - Fehler beim Training werden abgefangen und ausgegeben.
    """
    command = [
        "yolo",
        "detect",
        "train",
        f"model={model}",
        f"data={data_yaml}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"device={device}",
        f"project={project}",
        f"name={name}",
    ]

    print("Starte YOLO Training:")
    print(" ".join(command))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Fehler beim Training:", e)

if __name__ == "__main__":
    train_yolo()
