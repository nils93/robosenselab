import trimesh
import os
import pyvista as pv
import numpy as np
import random
import csv

from scripts.append_label_entry import append_label_entry
from scripts.generate_camera_sphere_positions import generate_camera_sphere_positions
from scripts.append_pose_label_entry import append_pose_label_entry
from scripts.camera_to_quaternion import camera_to_quaternion
from scripts.append_pose_quaternion_entry import append_pose_quaternion_entry

def save_augmented_views(mesh, output_dir, model_name, n_views=20, apply_random_rotation=True, progress_bar=None, split_ratio=0.8, start_index=0):
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
        plotter = None  # wichtig: existiert auch im finally-Block

        try:
            plotter = pv.Plotter(off_screen=True)
            plotter.set_background("white")
            plotter.window_size = [512, 512]

            pv_mesh = pv.wrap(mesh_copy)

            if hasattr(mesh_copy.visual, "vertex_colors") and mesh_copy.visual.vertex_colors is not None:
                colors = mesh_copy.visual.vertex_colors
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pv_mesh.point_data["colors"] = colors[:, :3]
                plotter.add_mesh(pv_mesh, scalars="colors", rgb=True)
            else:
                plotter.add_mesh(pv_mesh, color="lightgray")

            view_dir = np.array(center) - cam_pos
            up = np.array([0, 0, 1])
            if np.abs(np.dot(view_dir / np.linalg.norm(view_dir), up)) > 0.99:
                up = np.array([0, 1, 0])

            plotter.camera_position = [cam_pos.tolist(), center.tolist(), up.tolist()]
            plotter.camera.zoom(0.9)

            subset = "train" if random.random() < split_ratio else "val"
            model_dir = os.path.join(output_dir, subset, model_name)
            os.makedirs(model_dir, exist_ok=True)

            index = start_index + i
            filename = f"{model_name}_view_{index:04d}.png"
            output_path = os.path.join(model_dir, filename)

            plotter.screenshot(output_path)

            relative_path = os.path.join(subset, model_name, filename)
            csv_path = os.path.join(output_dir, "pose_labels.csv")
            quat = camera_to_quaternion(cam_pos, center, up)
            append_pose_quaternion_entry(csv_path, relative_path, model_name, cam_pos.tolist(), quat.tolist())

            if progress_bar:
                progress_bar.update(1)

        finally:
            if plotter:
                plotter.close()


