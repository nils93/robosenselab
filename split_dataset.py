import os
import shutil
import random

def split_dataset(source_dir, target_dir, train_ratio=0.8):
    classes = os.listdir(source_dir)
    for cls in classes:
        class_path = os.path.join(source_dir, cls)
        if not os.path.isdir(class_path):
            continue

        images = [f for f in os.listdir(class_path) if f.endswith(".png")]
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)

        for subset, subset_images in zip(["train", "val"], [images[:split_idx], images[split_idx:]]):
            subset_dir = os.path.join(target_dir, subset, cls)
            os.makedirs(subset_dir, exist_ok=True)

            for img in subset_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(subset_dir, img)
                shutil.copy(src, dst)

        print(f"[{cls}] → train: {split_idx}, val: {len(images)-split_idx}")
    print("[Fertig] Datensatz aufgeteilt.")