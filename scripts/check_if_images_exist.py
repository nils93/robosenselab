import os

# Funktion, um zu überprüfen, ob Bilder bereits existieren
def check_if_images_exist(output_dir, model_name):
    model_dir = os.path.join(output_dir, model_name)  # Der Pfad zum Modellordner
    # Wenn das Verzeichnis existiert und es bereits Bilder gibt, überspringen
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"Bilder für {model_name} existieren bereits. Überspringe Verarbeitung.")
        return True
    return False