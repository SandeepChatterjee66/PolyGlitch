import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import random

def get_brightness(img, x, y):
    h, w = img.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    return img[y, x] / 255.0

def draw_brightness_voronoi(image_path, N=1000, save_path="voronoi_shaded_art.png"):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape

    # Generate N random seed points
    points = np.array([(random.uniform(0, w), random.uniform(0, h)) for _ in range(N)])
    vor = Voronoi(points)

    # Plot Voronoi with brightness-based edge thickness
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')

    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not region or -1 in region:
            continue  # skip open regions

        vertices = [vor.vertices[i] for i in region]
        num_vertices = len(vertices)

        for i in range(num_vertices):
            start = vertices[i]
            end = vertices[(i + 1) % num_vertices]
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            brightness = get_brightness(img, mid_x, mid_y)
            line_thickness = max(0.2, (1.0 - brightness) * 2.5)  # adjust scale for visual effect
            ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=line_thickness)

    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    print(f"Saved Voronoi shaded art to {save_path}")

# Example usage
draw_brightness_voronoi("ramanujan.jpg", N=15000, save_path="voronoi_shaded_output.png")
