import subprocess

def train_yolo(
    model="yolo11s.pt",
    data_yaml="yolo/ultralytics/data.yaml",
    epochs=500,
    imgsz=640,
    batch=8,
    device=0,
    project="outputs/yolo_runs",
    name="train_long_run"
):
    command = [
        "yolo",
        "detect",
        "train",
        f"model={model}",
        f"data={data_yaml}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"device={device}",
        f"project={project}",
        f"name={name}",
    ]

    print("Starte YOLO Training:")
    print(" ".join(command))

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("Fehler beim Training:", e)

# Falls du es direkt als Skript ausführst:
if __name__ == "__main__":
    train_yolo()
