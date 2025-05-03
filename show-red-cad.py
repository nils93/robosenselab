import trimesh
import pyvista as pv  # PyVista importieren
import numpy as np

# Lade das 3D-Modell mit Trimesh
def load_obj_with_trimesh(filename):
    # Lade das Modell
    mesh = trimesh.load_mesh(filename)

    # PyVista Mesh erstellen
    pv_mesh = pv.wrap(mesh)
    
    return pv_mesh

# Render das Modell
def render_obj(pv_mesh):
    # Setze die globale Farbe auf Rot für das Modell
    pv.global_theme.color = [1.0, 1.0, 0.0]  # Setze die Farbe für alle Objekte auf Rot

    # Rendern des Modells mit PyVista
    plotter = pv.Plotter()

    # Das Modell anzeigen
    plotter.add_mesh(pv_mesh)
    plotter.show()

# Main Loop
def main(obj_file):
    # Lade das Modell mit Trimesh und setze die Farbe in PyVista
    pv_mesh = load_obj_with_trimesh(obj_file)

    # Rendern des Modells
    render_obj(pv_mesh)

if __name__ == "__main__":
    obj_file = 'data/cad_models/morobot-s_Achse-1A_yellow.obj'  # Ersetze dies mit dem Pfad zu deiner .obj-Datei
    main(obj_file)
