import shutil
from pathlib import Path

def merge_results(example_name: str):
    """
    Führt Ergebnisdaten aus verschiedenen Verzeichnissen (YOLO, MegaPose) für ein Beispiel zusammen.

    Strukturierte Ausgabe erfolgt im Ordner: outputs/results/<bildname>/

    Für jedes Bild wird ein Ordner angelegt, der folgende Dateien enthalten kann:
    - original.jpg/png           → Originalbild aus inputs/
    - detection_overlay.jpg/png → YOLO-Ergebnisbild mit Bounding Boxes
    - pose.json                  → Posen der erkannten Objekte
    - <Visualisierungen>        → Falls vorhanden, gesamte Visualisierung aus MegaPose

    Parameter:
    ----------
    example_name : str
        Name des Beispielordners unter `megapose6d/local_data/examples/<example_name>`

    Voraussetzungen:
    - MegaPose-Ergebnisse unter:      megapose6d/local_data/examples/<example_name>/outputs/
    - Visualisierungen unter:        megapose6d/local_data/examples/<example_name>/visualizations/
    - Eingabebilder unter:           megapose6d/local_data/examples/<example_name>/inputs/
    - YOLO-Inferenzbilder unter:     outputs/yolo_runs/infer_run_long/
    - Zielordner:                    outputs/results/

    Hinweise:
    - Bestehende Zielordner werden nicht überschrieben.
    - Unterstützt .jpg und .png als Bildformate.
    """
    base_path = Path("megapose6d/local_data/examples") / example_name
    inputs_dir = base_path / "inputs"
    visualizations_dir = base_path / "visualizations"
    pose_dir = base_path / "outputs"
    yolo_dir = Path("outputs/yolo_runs/infer_run_long")
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Visualisierungen kopieren
    if visualizations_dir.exists():
        for subdir in visualizations_dir.iterdir():
            if subdir.is_dir():
                target_subdir = results_dir / subdir.name
                if not target_subdir.exists():
                    shutil.copytree(subdir, target_subdir)
                else:
                    print(f"⚠️ Ordner {subdir.name} existiert bereits – überspringe Kopie.")
    else:
        print("⚠️ Kein Visualisierungsordner vorhanden – überspringe.")

    # 2. Bilder aus inputs/
    for img_path in inputs_dir.glob("*.[jp][pn]g"):
        target_dir = results_dir / img_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"original{img_path.suffix}"
        shutil.copy(img_path, target_path)

    # 3. Bilder aus YOLO
    for img_path in yolo_dir.glob("*.[jp][pn]g"):
        target_dir = results_dir / img_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"detection_overlay{img_path.suffix}"
        shutil.copy(img_path, target_path)

    # 4. Posen kopieren
    for pose_path in pose_dir.glob("*_pose.json"):
        stem = pose_path.stem.replace("_pose", "")
        target_dir = results_dir / stem
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / "pose.json"
        shutil.copy(pose_path, target_path)

    print("✅ Ergebnisse wurden erfolgreich zusammengeführt.")
