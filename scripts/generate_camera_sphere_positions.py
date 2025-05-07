import numpy as np

def generate_camera_sphere_positions(n_views):
    """Generiert n gleichmäßig verteilte Punkte auf einer Einheitskugel."""
    indices = np.arange(0, n_views, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/n_views)
    theta = np.pi * (1 + 5**0.5) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.vstack((x, y, z)).T  # shape: (n_views, 3)