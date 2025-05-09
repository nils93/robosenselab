import os
import csv

def append_pose_label_entry(csv_path, filepath, label, camera_pos, up_vector):
    header = ["filepath", "label", "cam_x", "cam_y", "cam_z", "up_x", "up_y", "up_z"]
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            filepath,
            label,
            *camera_pos,
            *up_vector
        ])
