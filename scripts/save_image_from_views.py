import os
import pyvista as pv
from scripts.calculate_camera_positions import calculate_camera_positions
from scripts.camera_to_quaternion import camera_to_quaternion
from scripts.append_pose_quaternion_entry import append_pose_quaternion_entry

def save_image_from_views(mesh, output_dir, model_name):
    camera_positions = calculate_camera_positions(mesh)
    subset = "val"
    model_dir = os.path.join(output_dir, subset, model_name)
    os.makedirs(model_dir, exist_ok=True)

    view_names = ["front", "back", "left", "right", "top", "bottom"]
    saved_count = 0  # Zähle nur neu gespeicherte Bilder

    for i, camera_pos in enumerate(camera_positions):
        view_name = view_names[i]
        filename = f"{model_name}_{view_name}_view.png"
        output_path = os.path.join(model_dir, filename)

        if os.path.exists(output_path):
            continue

        scene = pv.Plotter(off_screen=True)
        scene.set_background("white")
        scene.window_size = [512, 512]
        scene.add_mesh(mesh, scalars=None, rgb=True)

        if i == 4:
            up = (0, 0, 1)
        elif i == 5:
            up = (0, 0, -1)
        else:
            up = (0, 1, 0)

        scene.camera_position = [camera_pos, (0, 0, 0), up]
        scene.screenshot(output_path)
        scene.close()

        # ➤ Berechne Quaternion aus Kameraausrichtung
        quat = camera_to_quaternion(camera_pos, (0, 0, 0), up)

        # ➤ Trage in CSV ein
        relative_path = os.path.join(subset, model_name, filename)
        csv_path = os.path.join(output_dir, "pose_labels.csv")
        append_pose_quaternion_entry(csv_path, relative_path, model_name, camera_pos, quat)

        saved_count += 1

    return saved_count
