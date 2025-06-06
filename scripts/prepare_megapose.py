from pathlib import Path
from PIL import Image
import shutil


def prepare_megapose(example_name: str):
    base_path = Path("megapose6d/local_data/examples") / example_name

    # Zielpfade
    inputs_dir = base_path / "inputs"
    meshes_dir = base_path / "meshes"
    camera_file_dst = base_path / "camera_data.json"

    # Quellpfade
    src_json_dir = Path("outputs/yolo_runs/infer_run_long")
    src_image_dir = Path("data/predict")
    src_camera_file = Path("outputs/camera_calibration/camera_data.json")

    # Verzeichnisse anlegen
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON-Dateien kopieren
    for json_file in src_json_dir.glob("*.json"):
        shutil.copy(json_file, inputs_dir)
        print(f"📄 Kopiert: {json_file.name}")

    # 2. JPG → PNG konvertieren (aus data/predict)
    for jpg_path in src_image_dir.glob("*.jpg"):
        png_path = inputs_dir / (jpg_path.stem + ".png")
        img = Image.open(jpg_path)
        img.save(png_path)
        print(f"🖼️ Konvertiert: {jpg_path.name} → {png_path.name}")

    # 3. Kamera-JSON kopieren
    if src_camera_file.exists():
        shutil.copy(src_camera_file, camera_file_dst)
        print(f"📷 Kamera-Parameter kopiert nach {camera_file_dst}")
    else:
        print("❌ Kamera-JSON nicht gefunden!")

if __name__ == "__main__":
    prepare_megapose("morobot")
