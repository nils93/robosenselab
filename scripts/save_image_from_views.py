import os
import pyvista as pv
from scripts.calculate_camera_positions import calculate_camera_positions
from scripts.append_label_entry import append_label_entry

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

        # ➤ Überspringen, wenn Bild bereits existiert
        if os.path.exists(output_path):
            continue

        # Neu rendern und speichern
        scene = pv.Plotter(off_screen=True)
        scene.set_background("white")
        scene.window_size = [512, 512]
        scene.add_mesh(mesh, scalars=None, rgb=True)

        if i == 4:
            scene.camera_position = [camera_pos, (0, 0, 0), (0, 0, 1)]
        elif i == 5:
            scene.camera_position = [camera_pos, (0, 0, 0), (0, 0, -1)]
        else:
            scene.camera_position = [camera_pos, (0, 0, 0), (0, 1, 0)]

        scene.render()
        scene.screenshot(output_path)
        scene.close()

        # ➤ Label nur ergänzen, wenn Bild neu
        relative_path = os.path.join(subset, model_name, filename)
        csv_path = os.path.join(os.path.dirname(output_dir), "labels.csv")
        append_label_entry(csv_path, relative_path, model_name)

        saved_count += 1

    return saved_count  # → wichtig für tqdm.update(...)
