import numpy as np
import trimesh

# Funktion zum Verschieben des Modells zum Ursprung
def move_model_to_origin(mesh):
    # Berechne den Mittelpunkt der Bounding Box
    model_center = (mesh.bounds[0] + mesh.bounds[1]) / 2

    # Verschiebe das Modell, sodass der Mittelpunkt im Ursprung liegt
    mesh.apply_translation(-model_center)  # Verschiebt das Modell zum Ursprung

    return mesh