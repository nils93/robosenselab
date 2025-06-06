#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/.."

echo "🔧 Starte Setup für YOLO und MegaPose..."

# 1. YOLO-Konfiguration kopieren
echo "📁 Kopiere YOLO-Konfigurationsdatei..."
if [ ! -f "$ROOT_DIR/yolo/data.yaml" ]; then
    cp "$SCRIPT_DIR/yolo_data.yaml" "$ROOT_DIR/yolo/data.yaml"
    echo "  ✅ data.yaml kopiert."
else
    echo "  ⚠️  yolo/data.yaml existiert bereits – überspringe Kopie."
fi

# 2. Verzeichnisse erstellen
echo "📁 Erstelle Verzeichnisse für MegaPose..."
mkdir -p \
  "$ROOT_DIR/megapose6d/local_data/examples/morobot" \
  "$ROOT_DIR/megapose6d/local_data/examples/morobot/inputs" \
  "$ROOT_DIR/megapose6d/local_data/examples/morobot/meshes"

# 3. .ply-Dateien kopieren
echo "📁 Kopiere .ply-Dateien..."
for file in "$ROOT_DIR/data/cad_models/"*.ply; do
    [ -e "$file" ] || { echo "⚠️  Keine .ply-Dateien gefunden."; break; }
    filename=$(basename "$file")
    target="$ROOT_DIR/megapose6d/local_data/examples/morobot/meshes/$filename"
    if [ ! -f "$target" ]; then
        cp "$file" "$target"
        echo "  ➕ $filename kopiert."
    else
        echo "  ⚠️  $filename existiert bereits – überspringe Kopie."
    fi
done

# 4. run_infer_on.py kopieren
echo "📁 Kopiere megapose_run_infer_on.py → run_infer_on.py für MegaPose..."
infer_target="$ROOT_DIR/megapose6d/src/megapose/scripts/run_infer_on.py"
mkdir -p "$(dirname "$infer_target")"

if [ ! -f "$infer_target" ]; then
    cp "$SCRIPT_DIR/megapose_run_infer_on.py" "$infer_target"
    echo "  ✅ run_infer_on.py kopiert."
else
    echo "  ⚠️  run_infer_on.py existiert bereits – überspringe Kopie."
fi




echo "✅ Setup abgeschlossen."
