import os
import subprocess
from pathlib import Path

from scripts.merge_results import merge_results
from scripts.camera_calibration import calibrate_camera
from scripts.train_yolo import train_yolo
from scripts.predict_yolo import predict_yolo
from scripts.prepare_megapose import prepare_megapose

def main():
    """
    Hauptmenü zur Ausführung verschiedener Schritte der MegaPose-Pipeline.

    Optionen:
    ----------
    1. Kamera kalibrieren
       → Startet die OpenCV-basierte Kalibrierung und speichert 'camera_data.json'.
    
    2. YOLO trainieren
       → Führt das Training mit einem definierten YOLO-Modell aus.
    
    3. Pipeline starten
       → Führt folgende Schritte automatisiert durch:
           - Prüft, ob Kamera kalibriert ist
           - Führt YOLO-Inferenz durch
           - Bereitet MegaPose-Eingaben vor
           - Führt MegaPose-Skripte aus (Inferenz, Pose-Visualisierung)
           - Führt alle Ergebnisse in 'outputs/results/' zusammen

    Voraussetzungen:
    ----------------
    - Kamera ist kalibriert (camera_data.json vorhanden)
    - Meshes sind vorhanden unter: megapose6d/local_data/examples/<example>/meshes/
    - Bilder und YOLO-Ergebnisse liegen vor

    Hinweis:
    --------
    Der Ablauf verwendet feste Beispielnamen ('morobot') und geht davon aus,
    dass `megapose6d/src` das Modulverzeichnis ist (wird via PYTHONPATH gesetzt).
    """
    print("Was willst du tun?")
    print("1. Kamera kalibrieren")
    print("2. Yolo trainieren")
    print("3. Pipeline starten")

    choice = input("\n Bitte wähle eine Option! ").strip()

    if choice == "1":
        calibrate_camera()

    elif choice == "2":
        train_yolo()

    elif choice == "3":
        print("Starte Pipeline...")

        # Kamera-Kalibrierung prüfen
        print("Prüfe, ob Kamera kalibriert ist...")
        if not os.path.exists("outputs/camera_calibration/camera_data.json"):
            print("❌ Kamera ist nicht kalibriert. Bitte führe zuerst die Kamera-Kalibrierung durch.")
            return
        print("✅ Kamera ist kalibriert.")

        # YOLO starten
        print("Starte YOLO Inference...")
        predict_yolo()

        input("Drücke Enter, um die YOLO results für megapose6d vorzubereiten...")

        # Vorbereitung für MegaPose
        example_name = "morobot"
        prepare_megapose(example_name)

        inputs_dir = Path(f"megapose6d/local_data/examples/{example_name}/inputs")
        meshes_dir = Path(f"megapose6d/local_data/examples/{example_name}/meshes")

        if not inputs_dir.exists():
            print(f"❌ Eingabeverzeichnis fehlt: {inputs_dir}\n👉 Bitte führe zuerst ./setup/setup.sh aus.")
            return
        if not meshes_dir.exists():
            print(f"❌ Mesh-Verzeichnis fehlt: {meshes_dir}\n👉 Bitte führe zuerst ./setup/setup.sh aus.")
            return
        if not any(inputs_dir.iterdir()):
            print(f"❌ Eingabeverzeichnis ist leer: {inputs_dir}")
            return
        if not any(meshes_dir.iterdir()):
            print(f"❌ Mesh-Verzeichnis ist leer: {meshes_dir}")
            return

        print(f"✅ Megapose-Daten für '{example_name}' erfolgreich vorbereitet.")
        print("Starte Megapose Pipeline...")
        os.environ["PYTHONPATH"] = os.path.abspath("megapose6d/src")

        commands = [
            ("▶️ Starte Inference...", ["python", "-m", "megapose.scripts.run_infer_on", "morobot", "--run-inference"]),
            ("✅ Inference abgeschlossen.", None),
            ("▶️ Starte Pose Estimation...", ["python", "-m", "megapose.scripts.run_infer_on", "morobot", "--vis-outputs"]),
            (None, ["python", "-m", "megapose.scripts.run_infer_on", "morobot", "--draw-pose-bbox"]),
            ("✅ Pose Estimation abgeschlossen.", None),
        ]

        for message, cmd in commands:
            if message:
                print(message)
            if cmd:
                subprocess.run(cmd, check=True)

        print("▶️ Führe Ergebnisse zusammen...")
        merge_results("morobot")

        print("✅ Pipeline erfolgreich abgeschlossen.")
        print("Die Ergebnisse findest du in 'outputs/results/'.")

    else:
        print("❌ Ungültige Eingabe.")

if __name__ == "__main__":
    main()
