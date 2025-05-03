# Funktioniert nicht. Problem: Beleuchtung ist nicht korrekt.
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import os

# Lade das OBJ-Modell
def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Nur Positionen der Vertices extrahieren
                vertex = list(map(float, line.strip().split()[1:4]))  # Nimm nur die ersten 3 Werte (x, y, z)
                vertices.append(vertex)
            elif line.startswith('f '):  # Gesichter (faces)
                face = line.strip().split()[1:]
                face = [int(i.split('/')[0]) - 1 for i in face]  # Nur die Vertex-Indizes verwenden
                if len(face) == 3:  # Sicherstellen, dass jedes Face 3 Vertices hat (d.h. dreieckig ist)
                    faces.append(face)
                else:
                    print(f"Ungültiges Face gefunden, überspringe: {face}")
    
    # Debugging: Überprüfen, ob alle Vertices korrekt geladen wurden
    print(f"Anzahl der geladenen Vertices: {len(vertices)}")
    print(f"Anzahl der geladenen Faces: {len(faces)}")
    
    center = np.mean(vertices, axis=0)
    vertices = [list(np.array(vertex) - center) for vertex in vertices]  # Alle Vertices verschieben, sodass der Mittelpunkt im Ursprung liegt
    return vertices, faces

# Render das Modell ohne Normalen
def render_obj(vertices, faces):
    for face in faces:
        glBegin(GL_TRIANGLES)
        for vertex in face:
            glVertex3fv(vertices[vertex])  # Rendere nur die Vertex-Koordinaten ohne Normalen
        glEnd()

# Funktion zum Zeichnen des Koordinatensystems
def draw_axes(scale=100):
    # X-Achse (rot)
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(scale, 0, 0)  # Skalierung der X-Achse
    glEnd()

    # Y-Achse (grün)
    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, scale, 0)  # Skalierung der Y-Achse
    glEnd()

    # Z-Achse (blau)
    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, scale)  # Skalierung der Z-Achse
    glEnd()

# Initialisierung der OpenGL-Umgebung
def init_opengl(width, height):
    glClearColor(1, 1, 1, 1)  # Weißer Hintergrund
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_LIGHTING)
    for i in range(6):  # Nur sechs Lichtquellen, eine von jeder Seite und in den Ecken
        glEnable(GL_LIGHT0 + i)

    positions = [
        (1.0, 0.0, 0.0, 0.0),  # Rechts
        (-1.0, 0.0, 0.0, 0.0),  # Links
        (0.0, 1.0, 0.0, 0.0),  # Oben
        (0.0, -1.0, 0.0, 0.0),  # Unten
        (0.0, 0.0, 1.0, 0.0)  # Vorne
        #(-1.0, 1.0, -1.0, 0.0)  # Oben hinten links
    ]

    for i, position in enumerate(positions):
        glLightfv(GL_LIGHT0 + i, GL_POSITION, position)
        glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, (0.3, 0.3, 0.3, 1.0))
        glLightfv(GL_LIGHT0 + i, GL_AMBIENT, (0.1, 0.1, 0.1, 1.0))

    glMaterialfv(GL_FRONT, GL_AMBIENT,  (0.5, 0.5, 0.5, 1.0))
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   (0.7, 0.7, 0.7, 1.0))

    gluPerspective(45, width / height, 0.1, 200.0)
    glTranslatef(0.0, 0.0, -150)  # Kamera weiter vom Modell entfernt

# Funktion zur Handhabung der Maus- und Tastatureingaben
def handle_input(rotation_speed=2.0, zoom_speed=1.0):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        glRotatef(rotation_speed, 0, 1, 0)
    if keys[pygame.K_RIGHT]:
        glRotatef(-rotation_speed, 0, 1, 0)
    if keys[pygame.K_UP]:
        glRotatef(rotation_speed, 1, 0, 0)
    if keys[pygame.K_DOWN]:
        glRotatef(-rotation_speed, 1, 0, 0)

    if keys[pygame.K_w]:
        glTranslatef(0.0, 0.0, zoom_speed)
    if keys[pygame.K_s]:
        glTranslatef(0.0, 0.0, -zoom_speed)

# Main Loop
def main(obj_file):
    vertices, faces = load_obj(obj_file)

    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    init_opengl(display[0], display[1])

    pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        handle_input()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_axes(scale=20)  # Koordinatensystem größer machen
        render_obj(vertices, faces)  # Rendern ohne Normalen

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    obj_file = 'data/cad_models/morobot-s_Achse-1A_gray.obj'  # Ersetze dies mit dem Pfad zu deiner .obj-Datei
    main(obj_file)
