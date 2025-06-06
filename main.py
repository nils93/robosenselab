import os
import subprocess

from scripts.merge_results import merge_results
from scripts.camera_calibration import calibrate_camera
from scripts.train_yolo import train_yolo
from scripts.predict_yolo import predict_yolo
from scripts.prepare_megapose import prepare_megapose
from pathlib import Path

def main():
    print("Was willst du tun?")
    print("1. Kamera kalibrieren")
    print("2. Yolo trainieren")
    print("3. Pipeline starten")
    print("4. Results zusammenführen")

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

        # Vorbereitung für Megapose
        
        example_name = "morobot"
        prepare_megapose(example_name)
        

        inputs_dir = Path("megapose6d/local_data/examples/morobot/inputs")
        meshes_dir = Path("megapose6d/local_data/examples/morobot/meshes")

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
        print("▶️ Führe Ergbenisse zusammen...")
        
        merge_results("morobot")
        print("✅ Pipeline erfolgreich abgeschlossen.")

    elif choice == "4":
        print("▶️ Führe Ergbenisse zusammen...")
        merge_results("morobot")

    else:
        print("❌ Ungültige Eingabe.")

if __name__ == "__main__":
    main()