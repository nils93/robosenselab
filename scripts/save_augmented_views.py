import trimesh
import os
import pyvista as pv
import numpy as np
import random


def generate_camera_sphere_positions(n_views):
    """Generiert n gleichmäßig verteilte Punkte auf einer Einheitskugel."""
    indices = np.arange(0, n_views, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_views)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T  # shape: (n_views, 3)

def save_augmented_views(mesh, output_dir, model_name, n_views=20, apply_random_rotation=True, progress_bar=None, split_ratio=0.8):
    """
    Rendert n Bilder eines CAD-Modells aus verschiedenen Perspektiven mit optionaler Zufallsrotation.
    Die Bilder werden zufällig auf train/val verteilt.
    """
    if mesh.is_empty:
        print(f"[Fehler] Mesh '{model_name}' ist leer und wird übersprungen.")
        return

    # Erstelle das Zielverzeichnis
    #model_dir = os.path.join(output_dir, model_name)
    #os.makedirs(model_dir, exist_ok=True)

    # Optional: zufällige Rotation auf das Mesh anwenden
    mesh_copy = mesh.copy()
    if apply_random_rotation:
        R = trimesh.transformations.random_rotation_matrix()
        mesh_copy.apply_transform(R)

    # Mittelpunkt des Modells
    center = mesh_copy.bounds.mean(axis=0)

    # Größe des Modells und dynamischer Radius
    diameter = np.linalg.norm(mesh_copy.extents)
    radius = diameter * 1.5  # Kameraabstand = 1.5x Modellgröße

    # Kamerapositionen auf Kugel um das Modellzentrum
    directions = generate_camera_sphere_positions(n_views)
    camera_positions = directions * radius + center

    for i, cam_pos in enumerate(camera_positions):
        # Neuer Plotter für jede Ansicht (vermeidet Persistenzprobleme)
        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("white")
        plotter.window_size = [512, 512]

        # Konvertiere Trimesh zu PyVista PolyData
        pv_mesh = pv.wrap(mesh_copy)

        # Prüfe, ob Farben existieren
        if hasattr(mesh_copy.visual, "vertex_colors") and mesh_copy.visual.vertex_colors is not None:
            colors = mesh_copy.visual.vertex_colors
            # Trimesh gibt Farben als uint8, PyVista will float32 [0,1]
            if colors.max() > 1.0:
                colors = colors / 255.0
            # Setze Farben als Punktdaten
            pv_mesh.point_data["colors"] = colors[:, :3]  # RGB (ignoriere Alpha)
            plotter.add_mesh(pv_mesh, scalars="colors", rgb=True)
        else:
            plotter.add_mesh(pv_mesh, color="lightgray")


        # Dynamischer "up"-Vektor (falls Kamera direkt über/unter dem Modell ist)
        view_dir = np.array(center) - cam_pos
        up = np.array([0, 0, 1])
        if np.abs(np.dot(view_dir/np.linalg.norm(view_dir), up)) > 0.99:
            up = np.array([0, 1, 0])  # vermeidet degenerierten Up-Vektor

        plotter.camera_position = [cam_pos.tolist(), center.tolist(), up.tolist()]
        plotter.camera.zoom(0.9)  # z.B. 90 % Zoom = mehr Abstand
        plotter.render()

        subset = "train" if random.random() < split_ratio else "val"
        model_dir = os.path.join(output_dir, subset, model_name)
        os.makedirs(model_dir, exist_ok=True)

        output_path = os.path.join(model_dir, f"{model_name}_view_{i:02d}.png")
        plotter.screenshot(output_path)
        #print(f"[Gespeichert] {output_path}")

        plotter.close()

        if progress_bar:
            progress_bar.update(1)
