# Standardbibliotheken
import argparse
import json
import os
from pathlib import Path
from typing import List

# Drittanbieterbibliotheken
import numpy as np
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image, ImageDraw, ImageFont
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

# Beispielverzeichnis
EXAMPLE_DIR = LOCAL_DATA_DIR / "examples" / "morobot"
DEFAULT_INPUT_DIR = EXAMPLE_DIR / "inputs"
DEFAULT_OUTPUT_DIR = EXAMPLE_DIR / "visualizations"

def load_single_observation(rgb_path: Path, camera_data: CameraData) -> np.ndarray:
    """Lädt ein RGB-Bild als NumPy-Array."""
    return np.array(Image.open(rgb_path), dtype=np.uint8)

def load_observation_tensor_from_file(rgb_path: Path, camera_data: CameraData, load_depth: bool = False) -> ObservationTensor:
    """Lädt RGB-Bild + Kamera-Matrix als MegaPose-kompatiblen ObservationTensor."""
    rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
    depth = None  # derzeit nicht genutzt
    return ObservationTensor.from_numpy(rgb, depth, camera_data.K)

def load_object_data(data_path: Path) -> List[ObjectData]:
    """Lädt eine Liste von ObjectData-Instanzen aus einer JSON-Datei."""
    data = json.loads(data_path.read_text())
    return [ObjectData.from_json(d) for d in data]

def load_detections_from_file(json_path: Path) -> DetectionsType:
    """Konvertiert eine JSON-Datei mit Bounding Boxes in ein CUDA-kompatibles Detections-Objekt."""
    return make_detections_from_object_data(load_object_data(json_path)).cuda()


def make_object_dataset(meshes_dir: Path) -> RigidObjectDataset:
    """Erstellt ein RigidObjectDataset aus allen .ply-Dateien im Mesh-Verzeichnis."""
    objects = [
        RigidObject(label=mesh.stem, mesh_path=mesh, mesh_units="mm")
        for mesh in meshes_dir.glob("*.ply")
    ]
    return RigidObjectDataset(objects)

def make_detections_visualization_from_pair(input_dir: Path = DEFAULT_INPUT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Visualisiert Detektionen (aus JSON) über den Eingabebildern."""
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rgb_path in sorted(input_dir.glob("*.png")):
        image_stem = rgb_path.stem
        json_path = input_dir / f"{image_stem}.json"
        if not json_path.exists():
            logger.warning(f"Keine JSON für {image_stem}, überspringe.")
            continue

        rgb = np.array(Image.open(rgb_path), dtype=np.uint8)
        detections = load_detections_from_file(json_path)

        plotter = BokehPlotter()
        fig_det = plotter.plot_detections(plotter.plot_image(rgb), detections)

        vis_subdir = output_dir / image_stem
        vis_subdir.mkdir(parents=True, exist_ok=True)
        export_png(fig_det, filename=vis_subdir / f"{image_stem}_detections.png")
        logger.info(f"Wrote detections visualization: {vis_subdir}")


def save_predictions_for_image(output_dir: Path, image_stem: str, pose_estimates: PoseEstimatesType):
    """Speichert Posen als JSON (MegaPose-kompatibel)."""
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [ObjectData(label, Transform(pose)) for label, pose in zip(labels, poses)]
    json_data = json.dumps([x.to_json() for x in object_data])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{image_stem}_pose.json"
    path.write_text(json_data)
    logger.info(f"Wrote predictions to: {path}")

def run_infer_on(example_dir: Path, model_name: str):
    """Führt die Inferenz auf allen JSON/RGB-Paaren im Eingabeordner durch."""
    logger.info(f"Starte Inferenz im Verzeichnis {example_dir}")
    camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())
    object_dataset = make_object_dataset(example_dir / "meshes")
    model = load_named_model(model_name, object_dataset).cuda()

    input_dir = example_dir / "inputs"
    output_dir = example_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_path in sorted(input_dir.glob("*.json")):
        stem = json_path.stem
        rgb_path = input_dir / f"{stem}.png"
        if not rgb_path.exists():
            logger.warning(f"Fehlendes RGB-Bild für {stem}, überspringe.")
            continue

        observation = load_observation_tensor_from_file(rgb_path, camera_data).cuda()
        detections = load_detections_from_file(json_path)

        try:
            output, _ = model.run_inference_pipeline(
                observation, detections=detections, **NAMED_MODELS[model_name]["inference_parameters"]
            )
            save_predictions_for_image(output_dir, stem, output)
        except Exception as e:
            logger.warning(f"Fehler bei {stem}: {e}")
            continue

def make_output_visualization(example_dir: Path):
    """Visualisiert die berechneten Meshes als Overlay (mit Panda3D gerendert)."""
    input_dir = example_dir / "inputs"
    output_dir = example_dir / "outputs"
    vis_base = example_dir / "visualizations"
    vis_base.mkdir(parents=True, exist_ok=True)

    for pose_path in sorted(output_dir.glob("*_pose.json")):
        stem = pose_path.stem.replace("_pose", "")
        image_path = input_dir / f"{stem}.png"
        if not image_path.exists():
            logger.warning(f"RGB fehlt für {stem}, überspringe.")
            continue

        rgb = np.array(Image.open(image_path))
        camera_data = CameraData.from_json((example_dir / "camera_data.json").read_text())
        camera_data.TWC = Transform(np.eye(4))
        object_data = load_object_data(pose_path)
        object_dataset = make_object_dataset(example_dir / "meshes")
        renderer = Panda3dSceneRenderer(object_dataset)

        camera_data, object_data = convert_scene_observation_to_panda3d(camera_data, object_data)
        light = [Panda3dLightData(light_type="ambient", color=(1, 1, 1, 1))]
        render = renderer.render_scene(object_data, [camera_data], light)[0]

        plotter = BokehPlotter()
        fig_overlay = plotter.plot_overlay(rgb, render.rgb)
        contour_img = make_contour_overlay(rgb, render.rgb, dilate_iterations=1)["img"]
        fig_contour = plotter.plot_image(contour_img)

        vis_dir = vis_base / stem
        vis_dir.mkdir(parents=True, exist_ok=True)
        export_png(fig_overlay, filename=vis_dir / "mesh_overlay.png")
        export_png(fig_contour, filename=vis_dir / "contour_overlay.png")
        logger.info(f"Wrote visualizations to {vis_dir}")

def draw_pose_box_visualization(example_dir: Path):
    """Zeichnet 3D-Bounding-Boxen + Koordinatenachsen auf das Originalbild."""
    input_dir = example_dir / "inputs"
    output_dir = example_dir / "outputs"
    vis_base = example_dir / "visualizations"

    K = np.array(json.load(open(example_dir / "camera_data.json"))["K"])
    fx, fy = K[0, 0], K[1, 1]

    axis_len = 0.05
    colors = {"x": (255, 0, 0), "y": (0, 255, 0), "z": (0, 0, 255)}
    LABEL_COLORS = {
        "morobot-s_Achse-1A_gray": (0, 255, 0),
        "morobot-s_Achse-1A_yellow": (255, 255, 0),
        "morobot-s_Achse-1B_yellow": (255, 0, 0),
        "morobot-s_Achse-3B_gray": (0, 255, 255),
    }
    font = ImageFont.truetype("DejaVuSans.ttf", 16)

    for rgb_path in sorted(input_dir.glob("*.png")):
        stem = rgb_path.stem
        pose_path = output_dir / f"{stem}_pose.json"
        bbox_path = input_dir / f"{stem}.json"
        if not pose_path.exists() or not bbox_path.exists():
            print(f"[WARN] Datei fehlt für {stem}, überspringe.")
            continue

        pose_data = json.load(open(pose_path))
        bbox_data = json.load(open(bbox_path))
        img = Image.open(rgb_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for obj in pose_data:
            label = obj["label"]
            quat, trans = obj["TWO"]
            R_cam = R.from_quat(quat).as_matrix()
            origin = np.array(trans)

            bbox = next((b for b in bbox_data if b["label"] == label), None)
            if not bbox: continue

            x1, y1, x2, y2 = bbox["bbox_modal"]
            z = origin[2]
            dx = ((x2 - x1) * z) / fx / 2
            dy = ((y2 - y1) * z) / fy / 2
            dz = 0.02
            corners = np.array([
                [-dx, -dy, -dz], [ dx, -dy, -dz], [ dx,  dy, -dz], [-dx,  dy, -dz],
                [-dx, -dy,  dz], [ dx, -dy,  dz], [ dx,  dy,  dz], [-dx,  dy,  dz],
            ])
            corners = (R_cam @ corners.T).T + origin
            project = lambda X: tuple((K @ X)[:2] / X[2])
            projected = list(map(project, corners))

            for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                draw.line([projected[i], projected[j]], fill=LABEL_COLORS.get(label, (255,255,255)), width=3)

            # Achsen
            for i, axis in enumerate(np.eye(3) * axis_len):
                end = R_cam @ axis + origin
                draw.line([project(origin), project(end)], fill=colors["xyz"[i]], width=3)

        # Legende
        for i, (label, color) in enumerate(LABEL_COLORS.items()):
            y = 20 + i * 25
            draw.rectangle([20, y, 40, y + 20], fill=color)
            draw.text((50, y), label, font=font, fill=(255, 255, 255))

        out_dir = vis_base / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        img.save(out_dir / "pose_overlay.png")
        print(f"[OK] {out_dir}/pose_overlay.png gespeichert.")

# === Entry Point ===
if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name")
    parser.add_argument("--model", default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true")
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--vis-outputs", action="store_true")
    parser.add_argument("--draw-pose-bbox", action="store_true")
    args = parser.parse_args()

    example_dir = LOCAL_DATA_DIR / "examples" / args.example_name

    if args.vis_detections:
        make_detections_visualization_from_pair()

    if args.run_inference:
        run_infer_on(example_dir, args.model)

    if args.vis_outputs:
        make_output_visualization(example_dir)

    if args.draw_pose_bbox:
        draw_pose_box_visualization(example_dir)


