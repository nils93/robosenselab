import os
import re

def count_augmented_images(base_dir, model_name):
    pattern = re.compile(rf"{re.escape(model_name)}_view_(\d+)\.png")
    total = 0
    for subset in ["train", "val"]:
        model_dir = os.path.join(base_dir, subset, model_name)
        if os.path.isdir(model_dir):
            for fname in os.listdir(model_dir):
                if pattern.match(fname):
                    total += 1
    return total
