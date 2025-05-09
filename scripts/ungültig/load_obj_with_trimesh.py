import trimesh
import os
import pyvista as pv
import numpy as np

# Funktion zum Laden des Modells mit Trimesh
def load_obj_with_trimesh(filename):

    # Lade das Modell
    mesh = trimesh.load_mesh(filename)

    # Setze die Farbe des Modells auf Rot [255, 0, 0, 255] (RGBA)
    mesh.visual.vertex_colors = [255, 0, 0, 255]  # Rot

    # PyVista Mesh erstellen
    pv_mesh = pv.wrap(mesh)

    # Rendere das Modell mit PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, color='white')  # Setze eine allgemeine Farbe für den Hintergrund
    plotter.show()

# Beispielaufruf
load_obj_with_trimesh("dein_modell.obj")
