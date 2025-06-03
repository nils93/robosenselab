# robosenselab

## 1. Vorbereitung der Entwicklungsumgebung
- Installiere die notwendigen Python-Bibliotheken:
  - `opencv-python` (für Bildverarbeitung)
  - `numpy` (für mathematische Operationen)
  - `pybullet` oder `pytransform3d` (für 3D-Transformationen und Pose-Schätzung)
  - `open3d` (optional für 3D-Visualisierung und Point-Cloud-Verarbeitung)
  - `scikit-image` (optional für zusätzliche Bildverarbeitungsfunktionen)
  - `Pillow` (optional für Bildbearbeitung)
  - `Trimesh`
  - `ultralytics` (yolo)
  - `pyvista`

  ```bash
  pip install opencv-python numpy pybullet pytransform3d open3d scikit-image Pillow trimesh ultralytics pyvista
  ```

## 2. Ultralytics YOLO als submodule einbinden

```bash
git submodule add https://github.com/ultralytics/ultralytics yolo
git submodule update --init --recursive
```

### I. data.yaml File einfügen
```bash
nano yolo/data.yaml
```
```yaml
# data.yaml
train: /home/focal/git/robosenselab/data/train
val: /home/focal/git/robosenselab/data/val
nc: 4
names:
  - morobot-s_Achse-1A_gray
  - morobot-s_Achse-1A_yellow
  - morobot-s_Achse-1B_yellow
  - morobot-s_Achse-3B_gray
```

## 3. YOLO trainieren
Sowohl der Trainings- als auch der Predict-Command sind in der main.py eingebunden!