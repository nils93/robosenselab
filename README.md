# robosenselab

## 0. Git-Repository klonen
```bash
git clone --recurse-submodules https://github.com/nils93/robosenselab.git robosenselab && cd robosenselab
```

## 1. Conda Environment erstellen
Hinweis: Dieser Schritt nimmt ca. 10 Minuten in Anspruch.
```bash
conda env create -f conda/environment.yaml
conda activate robosenselab
python -m pip install -r requirements.txt
```

## 2. Submodule `Ultralytics YOLO` und `meggapose6d` initialisieren

```bash
git submodule update --init --recursive
```

## 3. Pre-trained Pose-Estimation-Modelle herunterladen
Anmerkung: mit wget wird der SSL-Fehler ignoriert.  
Downloadgröße: 2,2 GB in 14m 41s (2,53 MB/s)
```bash
wget -r -np -nH --cut-dirs=3 --no-check-certificate https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/ -P megapose6d/local_data/megapose-models
```

## 4. `megapose6d` als Python-Modul verlinken
```bash
python -m pip install -e ./megapose6d
```

## 5. Beispieldaten herunterladen (optional)
Anmerkung: mit wget wird der SSL-Fehler ignoriert.
```bash
wget --no-check-certificate -r -np -nH --cut-dirs=3 https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/examples/ -P megapose6d/local_data/examples
```

## 6. `data`-Verzeichnis kopieren
Dieser Ordner wird aus lizenzrechtlichen Gründen separat zur Verfügung gestellt.

## 7. Setup-Skript ausführen
```bash
./setup/setup.sh
```

## 8. Starte das Hauptprogramm
```bash
python main.py
```

### I. Kalibriere Kamera -> Eingabe `1`

### II. Trainiere YOLO -> Eingabe `2`

### III. Starte die Pipeline (YOLO + megapose6d) -> Eingabe `3`

### IV. Ergebnisse liegen unter `outputs/results`


---


# Übersicht der MegaPose-Befehle

```bash
python -m megapose.scripts.run_infer_on morobot --vis-detections
python -m megapose.scripts.run_infer_on morobot --run-inference
python -m megapose.scripts.run_infer_on morobot --vis-outputs
python -m megapose.scripts.run_infer_on morobot --draw-pose-bbox
```
Die Ergebnisse findest du unter:
```xml
megapose6d/local_data/examples/morobot/outputs/
megapose6d/local_data/examples/morobot/visualizations/

```

