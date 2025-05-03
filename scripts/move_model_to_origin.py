# Funktion zum Verschieben des Modells in den Ursprung
def move_model_to_origin(mesh):
    center = mesh.centroid  # Berechne den Mittelpunkt des Modells
    mesh.apply_translation(-center)  # Verschiebe das Modell zum Ursprung
    return mesh