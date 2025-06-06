from pathlib import Path
from PIL import Image
import shutil

def prepare_megapose(example_name: str):
    """
    Bereitet die Datenstruktur für ein MegaPose-Beispiel vor.

    Folgende Schritte werden durchgeführt:
    1. Kopiert alle YOLO-JSON-Dateien aus 'outputs/yolo_runs/infer_run_long' in 'inputs/'.
    2. Konvertiert alle JPG-Bilder aus 'data/predict/' nach PNG und speichert sie in 'inputs/'.
    3. Kopiert die 'camera_data.json' aus der Kalibrierung nach '<example>/camera_data.json'.

    Parameter:
    ----------
    example_name : str
        Name des Beispielordners unter 'megapose6d/local_data/examples/<example_name>'

    Zielstruktur nach Ausführung:
    -----------------------------
    megapose6d/local_data/examples/<example_name>/
    ├── inputs/
    │   ├── <bildname>.png
    │   ├── <bildname>.json
    ├── camera_data.json

    Voraussetzungen:
    ----------------
    - YOLO-Labels als JSON-Dateien unter: outputs/yolo_runs/infer_run_long/
    - Eingabebilder (JPG) unter:          data/predict/
    - Kamera-Parameter unter:             outputs/camera_calibration/camera_data.json
    """
    base_path = Path("megapose6d/local_data/examples") / example_name

    # Zielpfade
    inputs_dir = base_path / "inputs"
    camera_file_dst = base_path / "camera_data.json"

    # Quellpfade
    src_json_dir = Path("outputs/yolo_runs/infer_run_long")
    src_image_dir = Path("data/predict")
    src_camera_file = Path("outputs/camera_calibration/camera_data.json")

    # Eingabeordner anlegen
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON-Dateien kopieren
    for json_file in src_json_dir.glob("*.json"):
        shutil.copy(json_file, inputs_dir)
        print(f"📄 Kopiert: {json_file.name}")

    # 2. JPG → PNG konvertieren
    for jpg_path in src_image_dir.glob("*.jpg"):
        png_path = inputs_dir / (jpg_path.stem + ".png")
        img = Image.open(jpg_path)
        img.save(png_path)
        print(f"🖼️ Konvertiert: {jpg_path.name} → {png_path.name}")

    # 3. Kamera-Parameter kopieren
    if src_camera_file.exists():
        shutil.copy(src_camera_file, camera_file_dst)
        print(f"📷 Kamera-Parameter kopiert nach {camera_file_dst}")
    else:
        print("❌ Kamera-JSON nicht gefunden!")

if __name__ == "__main__":
    prepare_megapose("morobot")
