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

def sample_pixels(img, percent):
    h, w, _ = img.shape
    num_pixels = int(h * w * percent / 100)
    ys = torch.randint(0, h, (num_pixels,))
    xs = torch.randint(0, w, (num_pixels,))
    points = torch.stack([xs, ys], dim=1).cpu().numpy()
    return points

def draw_triangulation(img_np, triangles, points, device='cuda'):
    h, w, _ = img_np.shape
    result = torch.zeros((h, w, 3), dtype=torch.uint8, device=device)

    img_tensor = torch.from_numpy(img_np).to(device)

    print("Rendering triangles with representative colors:")
    for tri_indices in tqdm(triangles.simplices, total=len(triangles.simplices), ncols=80):
        tri_pts = points[tri_indices]
        tri_pts_int = np.round(tri_pts).astype(np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tri_pts_int, 1)
        mask_tensor = torch.from_numpy(mask).bool().to(device)

        r, c = mask_tensor.nonzero(as_tuple=True)
        colors = img_tensor[r, c]

        if colors.size(0) > 0:
            color_mean = colors.float().mean(dim=0).to(dtype=torch.uint8)
            result[r, c] = color_mean

        # White border for triangle edges
        cv2.polylines(result.cpu().numpy(), [tri_pts_int], True, color=(255, 255, 255), thickness=2)

    return result.cpu().numpy()

def process(image_path, percent):
    image, img_np = load_image(image_path)
    sampled_points = sample_pixels(img_np, percent)

    if len(sampled_points) < 3:
        raise ValueError("Not enough points to form triangles")

    print(f"Triangulating using {len(sampled_points)} sampled points...")
    tri = Delaunay(sampled_points)

    triangulated_img = draw_triangulation(img_np, tri, sampled_points)

    out_path = os.path.splitext(image_path)[0] + str(percent)+"_triangulated.png"
    Image.fromarray(triangulated_img).save(out_path)
    print(f"✅ Saved triangulated image to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python triangulate_image.py <image_path> <percentage_of_pixels>")
        sys.exit(1)

    image_path = sys.argv[1]
    p = float(sys.argv[2])
    process(image_path, p)
