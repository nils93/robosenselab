import os
import csv

def append_pose_quaternion_entry(csv_path, filepath, label, translation, quaternion):
    header = ["filepath", "label", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            filepath,
            label,
            *translation,
            *quaternion
        ])
