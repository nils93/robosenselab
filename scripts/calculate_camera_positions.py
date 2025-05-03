import numpy as np
import trimesh

def calculate_camera_positions(mesh):
    # Berechne die maximalen Werte der Mesh-Bounds
    # mesh.bounds gibt die Grenzen des Meshes in der Form [x_min, x_max, y_min, y_max, z_min, z_max] zurück
    # Hier wird der Index des maximalen Wertes bestimmt, um die Kamera-Positionen zu berechnen

    max_value, max_index = max(
        (mesh.bounds[1], 0),  # (x_max, Index 0)
        (mesh.bounds[3], 1),  # (y_max, Index 1)
        (mesh.bounds[5], 2)   # (z_max, Index 2)
    )
    #print(f"Max value: {max_value}, Max index: {max_index}")

    # Berechne die Größe des Modells basierend auf den Extremen
    size = np.linalg.norm(mesh.bounds[max_index*2 + 1] - mesh.bounds[max_index*2])  # Entfernungen zwischen den Extremen

    # Bestimme eine geeignete Entfernung der Kamera basierend auf der Modellgröße
    camera_distance = int(size * 2.5)  # Kamera-Entfernung auf 2,5-fache Modellgröße setzen

    # Kamera-Positionen, dynamisch angepasst basierend auf dem Modell
    camera_positions = [
        (0, 0, -camera_distance),  # Vorne
        (0, 0, camera_distance),  # Hinten
        (-camera_distance, 0, 0),  # Links
        (camera_distance, 0, 0),  # Rechts
        (0, camera_distance, 0),  # Oben
        (0, -camera_distance, 0),  # Unten
    ]

    return camera_positions
