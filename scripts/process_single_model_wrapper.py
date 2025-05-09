import os

from scripts.process_pv_mesh import process_pv_mesh
from scripts.process_trimesh import process_trimesh
from scripts.save_image_from_views import save_image_from_views
from scripts.count_augmented_images import count_augmented_images
from scripts.save_augmented_views import save_augmented_views

def process_single_model_wrapper(args):
    obj_file, output_dir, n_views, generate_classic = args

    model_name = os.path.splitext(os.path.basename(obj_file))[0]
    saved_count = 0

    if generate_classic:
        pv_mesh, _ = process_pv_mesh(obj_file, output_dir)
        saved_count += save_image_from_views(pv_mesh, output_dir, model_name)

    trimesh_mesh, _ = process_trimesh(obj_file, output_dir)
    start_index = count_augmented_images(output_dir, model_name)

    save_augmented_views(
        trimesh_mesh,
        output_dir,
        model_name,
        n_views=n_views,
        split_ratio=0.8,
        start_index=start_index,
        progress_bar=None  # wichtig: nicht in Subprozessen
    )

    return saved_count + n_views  # für Fortschrittsanzeige
