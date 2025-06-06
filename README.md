# robosenselab

## 1. Conda Environment erstellen
Hinweis: Dieser Schritt nimmt ca. 10 Minuten in Anspruch.
```bash
conda env create -f conda/environment.yaml
conda activate robosenselab
python -m pip install -r requirements.txt
```

## 2. Ultralytics YOLO als submodule (im Hauptverzeichnis) einbinden 

```bash
git submodule add https://github.com/ultralytics/ultralytics yolo
cd yolo && git submodule update --init --recursive
```

### 3. meggapose6d als submodule (im Hauptverzeichnis) einbinden
```bash
git submodule add https://github.com/megapose6d/megapose6d megapose6d
cd megapose6d && git submodule update --init --recursive
```

#### I. Pre-trained pose estimation models herunterladen
Anmerkung: mit wget wird der SSL-Fehler ignoriert.
```bash
wget -r -np -nH --cut-dirs=3 --no-check-certificate https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/ -P ./local_data/megapose-models
```


#### II. Verlinke Python-Paket aus megapose6d
```bash
python -m pip install -e .
```

#### Optional: Example data herunterladen
Anmerkung: mit wget wird der SSL-Fehler ignoriert.
```bash
wget --no-check-certificate -r -np -nH --cut-dirs=3 https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/examples/ -P ./local_data/examples
```


# Übersicht der Befehle

## megapose6d
```
python -m megapose.scripts.run_infer_on morobot --vis-detections
python -m megapose.scripts.run_infer_on morobot --run-inference
python -m megapose.scripts.run_infer_on morobot --vis-outputs
python -m megapose.scripts.run_infer_on morobot --draw-pose-bbox
```