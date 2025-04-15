import torch
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
import cv2
import os
import sys
from tqdm import tqdm

def load_grayscale_image(image_path):
    image = Image.open(image_path).convert("L")  # grayscale
    img_np = np.array(image)
    return image, img_np

def compute_darkness_density_map(gray_img):
    darkness = 255 - gray_img.astype(np.float32)  # darker → higher values
    norm_dark = darkness / (darkness.max() + 1e-6)
    density = norm_dark + 0.01  # avoid empty areas
    density /= density.sum()  # normalize to probabilities
    return density

def sample_points(density_map, num_points):
    h, w = density_map.shape
    flat = density_map.flatten()
    indices = np.random.choice(len(flat), size=num_points, replace=False, p=flat)
    ys, xs = np.unravel_index(indices, (h, w))
    return np.stack([xs, ys], axis=1)

def draw_triangulated_portrait(gray_img, triangles, points, device='cuda'):
    h, w = gray_img.shape
    result = torch.zeros((h, w), dtype=torch.uint8, device=device)
    edge_layer = np.zeros((h, w), dtype=np.uint8)

    gray_tensor = torch.from_numpy(gray_img).to(device)

    print("Rendering portrait triangles:")
    for tri_indices in tqdm(triangles.simplices, total=len(triangles.simplices), ncols=80):
        tri_pts = points[tri_indices]
        tri_pts_int = np.round(tri_pts).astype(np.int32)

        # Fill triangle with avg grayscale
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tri_pts_int, 1)
        mask_tensor = torch.from_numpy(mask).bool().to(device)

        r, c = mask_tensor.nonzero(as_tuple=True)
        values = gray_tensor[r, c]

        if values.size(0) > 0:
            mean_val = values.float().mean().to(dtype=torch.uint8)
            result[r, c] = mean_val

        # Draw triangle edges in soft gray
        for i in range(3):
            pt1 = tri_pts_int[i]
            pt2 = tri_pts_int[(i + 1) % 3]
            cv2.line(edge_layer, tuple(pt1), tuple(pt2), color=100, thickness=1)

    # Combine base and edge
    final = result.cpu().numpy()
    final[edge_layer > 0] = edge_layer[edge_layer > 0]
    return final

def process_portrait(image_path, p):
    image, gray = load_grayscale_image(image_path)
    h, w = gray.shape
    total_points = int(h * w * p / 100)

    density_map = compute_darkness_density_map(gray)
    points = sample_points(density_map, total_points)

    print(f"Sampling {len(points)} points from dark regions for triangulation...")
    tri = Delaunay(points)
    triangulated = draw_triangulated_portrait(gray, tri, points)

    out_path = os.path.splitext(image_path)[0] + str(p)+"_triangulated_portrait.png"
    Image.fromarray(triangulated).save(out_path)
    print(f"✅ Saved stylized portrait to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python triangulate_portrait.py <image_path> <percentage_of_pixels>")
        sys.exit(1)

    image_path = sys.argv[1]
    p = float(sys.argv[2])
    process_portrait(image_path, p)
