import trimesh

# Lade das 3D-Modell mit Trimesh
def load_obj_with_trimesh(filename):
    mesh = trimesh.load_mesh(filename)
    return mesh

# Render das Modell
def render_obj(mesh):
    # Trimesh übernimmt die Visualisierung automatisch
    mesh.show()

# Main Loop
def main(obj_file):
    # Lade das Modell mit Trimesh
    mesh = load_obj_with_trimesh(obj_file)

    # Rendern des Modells
    render_obj(mesh)

if __name__ == "__main__":
    obj_file = 'data/cad_models/morobot-s_Achse-1A_gray.obj'  # Ersetze dies mit dem Pfad zu deiner .obj-Datei
    main(obj_file)
