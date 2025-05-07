import os
import csv

def append_label_entry(csv_path, relative_path, label):
    header = ["filepath", "label"]
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([relative_path, label])