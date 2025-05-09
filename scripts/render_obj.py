import pyvista as pv

# Render das Modell
def render_obj(pv_mesh):
    # Rendern des Modells mit PyVista
    plotter = pv.Plotter()

    # Das Modell anzeigen und die Farben als RGB verwenden
    plotter.add_mesh(pv_mesh, scalars="colors", rgb=True)  # Verwenden der Farben in PyVista
    plotter.show()