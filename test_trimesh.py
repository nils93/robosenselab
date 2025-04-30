import trimesh

def visualize_cad_model(cad_model_path):
    # Lade das 3D-CAD-Modell
    cad_model = trimesh.load_mesh(cad_model_path)

    # Visualisiere das Modell
    cad_model.show()

# Beispielaufruf
cad_model_path = 'data/cad_models/morobot-s_Achse-1A_gray.obj'
visualize_cad_model(cad_model_path)
