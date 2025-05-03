import os
import trimesh
import pyvista as pv

from scripts.process_all_models import process_all_models

# Hauptfunktion zum Starten des Prozesses
def main():
    model_directory = 'data/cad_models/'  # Ersetze dies mit dem Pfad zu deinem Verzeichnis der .obj-Dateien
    output_dir = './training_data'  # Zielverzeichnis für die gespeicherten Bilder

    # Sicherstellen, dass das Zielverzeichnis existiert
    os.makedirs(output_dir, exist_ok=True)

    # Verarbeite alle Modelle im angegebenen Verzeichnis
    process_all_models(model_directory, output_dir)

if __name__ == "__main__":
    main()


# Nächster Step: Die Farben werden aus den .obj-Dateien nicht korrekt extrahiert. Dementsprechend haben die Bilder eine falsche Farbe.