import trimesh

cad_model_path = '/home/nifa/git/robosenselab/data/cad_models/morobot-s_Achse-2A_yellow.obj'
cad_model_mesh = trimesh.load_mesh(cad_model_path, force='mesh', skip_materials=True)

if cad_model_mesh.is_empty:
    print("Das CAD-Modell ist leer!")
else:
    print("Das CAD-Modell wurde erfolgreich geladen.")
    cad_model_image = cad_model_mesh.scene().save_image(resolution=(640, 480))
    if cad_model_image is not None:
        print("Das Bild des CAD-Modells wurde erfolgreich extrahiert.")
    else:
        print("Fehler: Das CAD-Modell konnte nicht als Bild extrahiert werden.")
