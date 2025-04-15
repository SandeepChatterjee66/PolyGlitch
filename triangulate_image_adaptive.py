import torch
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay
import cv2
import os
import sys
from tqdm import tqdm

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image, np.array(image)

def compute_density_map(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    norm_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-6)
    density_map = norm_magnitude + 0.05  # prevent total flattening
    density_map /= density_map.sum()  # normalize as probabilities
    return density_map

def sample_points_by_density(density_map, num_samples):
    h, w = density_map.shape
    flat_density = density_map.flatten()
    indices = np.random.choice(len(flat_density), size=num_samples, replace=False, p=flat_density)
    ys, xs = np.unravel_index(indices, (h, w))
    points = np.stack([xs, ys], axis=1)
    return points

def draw_triangulation(img_np, triangles, points, device='cuda'):
    h, w, _ = img_np.shape
    result = torch.zeros((h, w, 3), dtype=torch.uint8, device=device)
    edge_layer = np.zeros((h, w, 3), dtype=np.uint8)

    img_tensor = torch.from_numpy(img_np).to(device)

    print("Rendering triangles with smart-colored edges:")
    for tri_indices in tqdm(triangles.simplices, total=len(triangles.simplices), ncols=80):
        tri_pts = points[tri_indices]
        tri_pts_int = np.round(tri_pts).astype(np.int32)

        # Fill triangle with average color
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tri_pts_int, 1)
        mask_tensor = torch.from_numpy(mask).bool().to(device)

        r, c = mask_tensor.nonzero(as_tuple=True)
        colors = img_tensor[r, c]

        if colors.size(0) > 0:
            color_mean = colors.float().mean(dim=0).to(dtype=torch.uint8)
            result[r, c] = color_mean

        # Draw colored edges based on connected vertex colors
        for i in range(3):
            pt1 = tri_pts_int[i]
            pt2 = tri_pts_int[(i + 1) % 3]

            # Get pixel positions (clamp to image bounds)
            x1, y1 = np.clip(pt1[0], 0, w-1), np.clip(pt1[1], 0, h-1)
            x2, y2 = np.clip(pt2[0], 0, w-1), np.clip(pt2[1], 0, h-1)

            color1 = img_np[y1, x1].astype(np.float32)
            color2 = img_np[y2, x2].astype(np.float32)
            edge_color = ((color1 + color2) / 2).astype(np.uint8).tolist()

            cv2.line(edge_layer, tuple(pt1), tuple(pt2), color=edge_color, thickness=2)

    # Blend filled triangles with the edges
    final_img = result.cpu().numpy()
    edge_mask = (edge_layer.sum(axis=-1) > 0)
    final_img[edge_mask] = edge_layer[edge_mask]

    return final_img



def process(image_path, percent):
    image, img_np = load_image(image_path)
    density_map = compute_density_map(img_np)

    h, w = density_map.shape
    num_samples = int(h * w * percent / 100)
    sampled_points = sample_points_by_density(density_map, num_samples)

    print(f"Triangulating using {len(sampled_points)} adaptively sampled points...")
    tri = Delaunay(sampled_points)

    triangulated_img = draw_triangulation(img_np, tri, sampled_points)

    out_path = os.path.splitext(image_path)[0] + str(percent)+ "_adapt_triangulated.png"
    Image.fromarray(triangulated_img).save(out_path)
    print(f"✅ Saved triangulated image to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python triangulate_image.py <image_path> <percentage_of_pixels>")
        sys.exit(1)

    image_path = sys.argv[1]
    p = float(sys.argv[2])
    process(image_path, p)
