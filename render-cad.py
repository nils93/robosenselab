import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from PIL import Image
import os

# Lade das OBJ-Modell
def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Nur Positionen der Vertices extrahieren
                # Nimm nur die ersten 3 Werte für x, y, z
                vertices.append(list(map(float, line.strip().split()[1:4])))  # Nimm nur die ersten 3 Werte
            elif line.startswith('f '):  # Gesichter (faces)
                face = line.strip().split()[1:]
                # Hole nur die Indizes der Vertices (ignoriere Textur und Normalen)
                face = [int(i.split('/')[0]) - 1 for i in face]  # OBJ-Index startet bei 1
                faces.append(face)
    return vertices, faces

# Render das Modell
def render_obj(vertices, faces):
    glBegin(GL_TRIANGLES)
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

# Initialisierung der OpenGL-Umgebung
def init_opengl(width, height):
    glClearColor(1, 1, 1, 1)
    glEnable(GL_DEPTH_TEST)
    gluPerspective(45, width / height, 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

# Bild speichern
def save_image(filename):
    # Bildschirm in ein Numpy-Array konvertieren
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, 800, 600, GL_RGB, GL_UNSIGNED_BYTE)
    pixels = np.frombuffer(pixels, dtype=np.uint8).reshape(600, 800, 3)

    # Bild speichern
    img = Image.fromarray(pixels)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL-Bild ist umgekehrt
    img.save(filename)

# Template generieren
def generate_template(obj_file, output_dir):
    vertices, faces = load_obj(obj_file)

    # Pygame und OpenGL initialisieren
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    init_opengl(display[0], display[1])

    # Modell rendern und speichern
    render_obj(vertices, faces)
    output_path = os.path.join(output_dir, 'template.png')
    save_image(output_path)
    pygame.quit()

if __name__ == "__main__":
    obj_file = 'data/cad_models/morobot-s_Achse-1A_gray.obj'  # Ersetze dies mit dem Pfad zu deiner .obj-Datei
    output_dir = './output'  # Zielverzeichnis für Templates
    os.makedirs(output_dir, exist_ok=True)
    
    generate_template(obj_file, output_dir)
    print("Template gespeichert als 'template.png'")
