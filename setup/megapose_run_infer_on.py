# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image, ImageDraw, ImageFont, Image as PILImage
from scipy.spatial.transform import Rotation as R

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import DetectionsType, ObservationTensor, PoseEstimatesType
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay

logger = get_logger(__name__)

#EXAMPLE_DIR = Path(__file__).parent
EXAMPLE_DIR = LOCAL_DATA_DIR / "examples" / "morobot"

DEFAULT_INPUT_DIR = EXAMPLE_DIR / "inputs"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "visualizations"

def load_single_observation(rgb_path: Path, camera_data: CameraData) -> np.ndarray:
    rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
    return rgb

def load_observation_tensor_from_file(
    rgb_path: Path,
    camera_data: CameraData,
    load_depth: bool = False
) -> ObservationTensor:
    rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
    depth = None  # Depth-Unterstützung könnte später ergänzt werden
    return ObservationTensor.from_numpy(rgb, depth, camera_data.K)

def load_object_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data

def load_detections_from_file(json_path: Path) -> DetectionsType:
    object_data = load_object_data(json_path)
    detections = make_detections_from_object_data(object_data).cuda()
    return detections

def make_object_dataset(meshes_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"

    for mesh_path in meshes_dir.glob("*.ply"):
        label = mesh_path.stem  # z. B. "morobot-s_Achse-1A_gray"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))

    return RigidObjectDataset(rigid_objects)

def make_detections_visualization_from_pair(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.png"))
    print(f"Found {len(image_paths)} RGB images in {input_dir}")
    for rgb_path in image_paths:
        image_stem = rgb_path.stem
        json_path = input_dir / f"{image_stem}.json"

        if not json_path.exists():
            logger.warning(f"Keine passende JSON-Datei gefunden für {image_stem}, überspringe.")
            continue

        rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
        detections = load_detections_from_file(json_path)

        plotter = BokehPlotter()
        fig_rgb = plotter.plot_image(rgb)
        fig_det = plotter.plot_detections(fig_rgb, detections=detections)

        vis_subdir = output_dir / image_stem
        vis_subdir.mkdir(parents=True, exist_ok=True)


        output_fn = vis_subdir / f"{image_stem}_detections.png"
        export_png(fig_det, filename=output_fn)
        logger.info(f"Wrote detections visualization: {output_fn}")

def save_predictions_for_image(
    output_dir: Path,
    image_stem: str,
    pose_estimates: PoseEstimatesType
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_stem}_pose.json"
    output_path.write_text(object_data_json)

    logger.info(f"Wrote predictions to: {output_path}")

def run_infer_on(
    example_dir: Path,
    model_name: str,
) -> None:
    logger.info(f"Starte Inferenzlauf im Verzeichnis {example_dir}")
    model_info = NAMED_MODELS[model_name]

    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())
    object_dataset = make_object_dataset(example_dir / "meshes")
    pose_estimator = load_named_model(model_name, object_dataset).cuda()

    input_dir = example_dir / "inputs"
    output_dir = example_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = sorted(input_dir.glob("*.json"))

    for json_path in pairs:
        image_stem = json_path.stem
        rgb_path = input_dir / f"{image_stem}.png"
        if not rgb_path.exists():
            logger.warning(f"Kein RGB-Bild gefunden für {image_stem}, überspringe.")
            continue

        logger.info(f"→ Verarbeite {image_stem}")
        observation = load_observation_tensor_from_file(rgb_path, camera_data).cuda()
        detections = load_detections_from_file(json_path).cuda()

        try:
            output, _ = pose_estimator.run_inference_pipeline(
                observation, detections=detections, **model_info["inference_parameters"]
            )
            save_predictions_for_image(output_dir, image_stem, output)
        except Exception as e:
            logger.warning(f"Fehler bei {image_stem}: {e}")
            continue

def make_output_visualization(example_dir: Path) -> None:
    input_dir = example_dir / "inputs"
    output_dir = example_dir / "outputs"
    vis_dir_base = example_dir / "visualizations"
    vis_dir_base.mkdir(parents=True, exist_ok=True)

    # Alle vorhandenen Predictions durchgehen
    for pose_json in sorted(output_dir.glob("*_pose.json")):
        image_stem = pose_json.stem.replace("_pose", "")
        image_path = input_dir / f"{image_stem}.png"

        if not image_path.exists():
            logger.warning(f"RGB-Bild fehlt für {image_stem}, überspringe.")
            continue

        rgb = np.array(Image.open(image_path), dtype=np.uint8)
        camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())
        camera_data.TWC = Transform(np.eye(4))

        object_datas = load_object_data(pose_json)
        object_dataset = make_object_dataset(example_dir / "meshes")

        renderer = Panda3dSceneRenderer(object_dataset)
        camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)

        light_datas = [
            Panda3dLightData(
                light_type="ambient",
                color=((1.0, 1.0, 1.0, 1)),
            ),
        ]

        renderings = renderer.render_scene(
            object_datas,
            [camera_data],
            light_datas,
            render_depth=False,
            render_binary_mask=False,
            render_normals=False,
            copy_arrays=True,
        )[0]

        plotter = BokehPlotter()
        fig_rgb = plotter.plot_image(rgb)
        fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
        contour_overlay = make_contour_overlay(
            rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
        )["img"]
        fig_contour_overlay = plotter.plot_image(contour_overlay)
        #fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)

        vis_dir = vis_dir_base / image_stem
        vis_dir.mkdir(parents=True, exist_ok=True)
        export_png(fig_mesh_overlay, filename=vis_dir / "mesh_overlay.png")
        export_png(fig_contour_overlay, filename=vis_dir / "contour_overlay.png")
        #export_png(fig_all, filename=vis_dir / "all_results.png")

        logger.info(f"Wrote visualizations to {vis_dir}")



def draw_pose_box_visualization(example_dir: Path) -> None:
    input_dir = example_dir / "inputs"
    output_dir = example_dir / "outputs"
    vis_base = example_dir / "visualizations"

    with open(example_dir / "camera_data.json", "r") as f:
        camera_data = json.load(f)
    K = np.array(camera_data["K"])
    fx, fy = K[0, 0], K[1, 1]

    axis_len = 0.05
    colors = {"x": (255, 0, 0), "y": (0, 255, 0), "z": (0, 0, 255)}

    # Farbzuweisung je Label
    LABEL_COLORS = {
        "morobot-s_Achse-1A_gray": (0, 255, 0),
        "morobot-s_Achse-1A_yellow": (255, 255, 0),
        "morobot-s_Achse-1B_yellow": (255, 0, 0),
        "morobot-s_Achse-3B_gray": (0, 255, 255),
    }

    font = ImageFont.truetype("DejaVuSans.ttf", 16)

    for rgb_path in sorted(input_dir.glob("*.png")):
        image_stem = rgb_path.stem
        pose_path = output_dir / f"{image_stem}_pose.json"
        bbox_path = input_dir / f"{image_stem}.json"

        if not pose_path.exists() or not bbox_path.exists():
            print(f"[WARN] Datei fehlt für {image_stem}, überspringe.")
            continue

        with open(pose_path, "r") as f:
            pose_data = json.load(f)
        with open(bbox_path, "r") as f:
            bbox_data = json.load(f)

        img = Image.open(rgb_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for pose_entry in pose_data:
            label = pose_entry["label"]
            quat = pose_entry["TWO"][0]
            trans = pose_entry["TWO"][1]
            R_cam = R.from_quat([*quat]).as_matrix()
            origin_cam = np.array(trans)

            matching_bbox = next((b for b in bbox_data if b["label"] == label), None)
            if not matching_bbox:
                print(f"[WARN] Keine bbox für {label}, überspringe.")
                continue

            x1, y1, x2, y2 = matching_bbox["bbox_modal"]
            bbox_width_px = x2 - x1
            bbox_height_px = y2 - y1
            z_depth = origin_cam[2]

            x_extent = (bbox_width_px * z_depth) / fx / 2.0
            y_extent = (bbox_height_px * z_depth) / fy / 2.0
            z_extent = 0.02  # heuristisch

            dx, dy, dz = x_extent, y_extent, z_extent
            corners_obj = np.array([
                [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
                [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz],
            ])
            corners_cam = (R_cam @ corners_obj.T).T + origin_cam

            def project(X):
                x = K @ X
                return (x[0]/x[2], x[1]/x[2])

            projected = [project(pt) for pt in corners_cam]
            edges = [
                (0,1),(1,2),(2,3),(3,0),
                (4,5),(5,6),(6,7),(7,4),
                (0,4),(1,5),(2,6),(3,7)
            ]

            box_color = LABEL_COLORS.get(label, (255, 255, 255))
            for i, j in edges:
                draw.line([projected[i], projected[j]], fill=box_color, width=3)

            # Achsen
            axes_obj = np.eye(3) * axis_len
            axes_cam = (R_cam @ axes_obj.T).T + origin_cam
            origin_2d = project(origin_cam)
            for i, name in enumerate(["x", "y", "z"]):
                tip_2d = project(axes_cam[i])
                draw.line([origin_2d, tip_2d], fill=colors[name], width=3)

        # Legende oben links einzeichnen
        legend_draw = ImageDraw.Draw(img)
        legend_x, legend_y = 20, 20
        for i, (legend_label, legend_color) in enumerate(LABEL_COLORS.items()):
            y = legend_y + i * 25
            legend_draw.rectangle([legend_x, y, legend_x + 20, y + 20], fill=legend_color)
            legend_draw.text((legend_x + 30, y), legend_label, font=font, fill=(255, 255, 255))

        out_dir = vis_base / image_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pose_overlay.png"
        img.save(out_path)
        print(f"[OK] {out_path} gespeichert.")

if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    parser.add_argument("--draw-pose-bbox", action="store_true")

    args = parser.parse_args()

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name

    if args.vis_detections:
        print("Visualisiere Detektionen...")
        make_detections_visualization_from_pair()

    if args.run_inference:
        print("Starte Inferenz...")
        run_infer_on(example_dir, args.model)

    if args.vis_outputs:
        print("Erstelle Visualisierungen der Ausgaben...")
        make_output_visualization(example_dir)

    if args.draw_pose_bbox:
        print("Zeichne Pose-Bounding-Box-Visualisierung...")
        draw_pose_box_visualization(example_dir)


