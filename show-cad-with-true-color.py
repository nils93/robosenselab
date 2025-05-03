import trimesh
import pyvista as pv
import numpy as np

# Lade das 3D-Modell mit Trimesh
def load_obj_with_trimesh(filename):
    # Lade das Modell mit Trimesh
    mesh = trimesh.load_mesh(filename)

    # Überprüfe, ob Farben im Modell definiert sind
    if mesh.visual.vertex_colors is not None:
        print(f"Farben wurden für das Modell geladen: {mesh.visual.vertex_colors}")
    else:
        print("Keine Farben im Modell gefunden!")

    # Um die Farben korrekt in PyVista darzustellen, sicherstellen, dass sie im [0, 1] Bereich liegen
    # Die Farben in Trimesh sind im Bereich von [0, 255], also müssen wir sie auf [0, 1] skalieren
    mesh.visual.vertex_colors = mesh.visual.vertex_colors / 255.0  # Farben in den Bereich [0, 1] konvertieren

    # PyVista Mesh erstellen
    pv_mesh = pv.wrap(mesh)
    
    # Setze die Farben als "point_data" in PyVista
    pv_mesh.point_data["colors"] = mesh.visual.vertex_colors  # Speichern der Farben für PyVista

    return pv_mesh

# Render das Modell
def render_obj(pv_mesh):
    # Rendern des Modells mit PyVista
    plotter = pv.Plotter()

    # Das Modell anzeigen und die Farben als RGB verwenden
    plotter.add_mesh(pv_mesh, scalars="colors", rgb=True)  # Verwenden der Farben in PyVista
    plotter.show()

# Main Loop
def main(obj_file):
    # Lade das Modell mit Trimesh und setze die Farbe in PyVista
    pv_mesh = load_obj_with_trimesh(obj_file)

    # Rendern des Modells
    render_obj(pv_mesh)

if __name__ == "__main__":
    obj_file = 'data/cad_models/morobot-s_Achse-1A_gray.obj'  # Ersetze dies mit dem Pfad zu deiner .obj-Datei
    main(obj_file)
