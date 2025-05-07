import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
import random
from scripts.append_pose_quaternion_entry import append_pose_quaternion_entry

def generate_backgrounds(output_dir, image_size=(512, 512), n_images=100):
    train_dir = os.path.join(output_dir, "train", "backgrounds")
    val_dir = os.path.join(output_dir, "val", "backgrounds")
    csv_path = os.path.join(output_dir, "pose_labels.csv")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Aktuelle maximale Indexnummer ermitteln
    existing_files = [f for f in os.listdir(train_dir) + os.listdir(val_dir) if f.startswith("background_") and f.endswith(".png")]
    existing_indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
    start_index = max(existing_indices, default=-1) + 1

    for i in tqdm(range(n_images), desc="🌟 Generiere Hintergründe"):
        noise = np.random.normal(loc=128, scale=40, size=(*image_size, 1)).clip(0, 255).astype(np.uint8)
        noise_rgb = np.repeat(noise, 3, axis=2)

        img = Image.fromarray(noise_rgb)
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 2)))

        if random.random() < 0.8:
            subset = "train"
            subdir = train_dir
        else:
            subset = "val"
            subdir = val_dir

        index = start_index + i
        filename = f"background_{index:04d}.png"
        rel_path = os.path.join(subset, "backgrounds", filename)
        full_path = os.path.join(subdir, filename)

        img.save(full_path)

        label = "background"
        cam_pos = [0.0, 0.0, 0.0]
        quat = [0.0, 0.0, 0.0, 1.0]
        append_pose_quaternion_entry(csv_path, rel_path, label, cam_pos, quat)

    print(f"\n🚀 {n_images} zusätzliche Hintergrundbilder wurden generiert und in pose_labels.csv eingetragen.")
