import numpy as np
import torch
import cv2
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse


def lloyd_relaxation(points, width, height, iterations=3):
    device = points.device
    for _ in range(iterations):
        vor = Voronoi(points.cpu().numpy())
        new_points = []
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if not region or -1 in region:
                new_points.append([0, 0])
                continue
            polygon = np.array([vor.vertices[i] for i in region])
            if polygon.shape[0] < 3:
                new_points.append([0, 0])
                continue
            centroid = polygon.mean(axis=0)
            centroid[0] = np.clip(centroid[0], 0, width)
            centroid[1] = np.clip(centroid[1], 0, height)
            new_points.append(centroid)
        points = torch.tensor(new_points, device=device, dtype=torch.float32)
    return points


def point_in_poly(xy, poly):
    n = poly.shape[0]
    inside = torch.zeros(xy.shape[0], dtype=torch.bool, device=xy.device)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > xy[:, 1]) != (yj > xy[:, 1])) & \
               (xy[:, 0] < (xj - xi) * (xy[:, 1] - yi) / (yj - yi + 1e-6) + xi)
        inside ^= cond
        j = i
    return inside


def rasterize_polygon_torch(polygon, width, height):
    min_x = torch.clamp(torch.floor(polygon[:, 0].min()).int(), 0, width - 1)
    min_y = torch.clamp(torch.floor(polygon[:, 1].min()).int(), 0, height - 1)
    max_x = torch.clamp(torch.ceil(polygon[:, 0].max()).int(), 0, width - 1)
    max_y = torch.clamp(torch.ceil(polygon[:, 1].max()).int(), 0, height - 1)

    ys, xs = torch.meshgrid(torch.arange(min_y, max_y, device=polygon.device),
                            torch.arange(min_x, max_x, device=polygon.device),
                            indexing='ij')
    coords = torch.stack([xs.flatten(), ys.flatten()], dim=1).float()
    inside_mask = point_in_poly(coords, polygon)

    mask = torch.zeros((height, width), dtype=torch.bool, device=polygon.device)
    ys_inside = coords[inside_mask, 1].long()
    xs_inside = coords[inside_mask, 0].long()
    mask[ys_inside, xs_inside] = True
    return mask


def draw_voronoi_torch(image_path, N=3000, downscale_factor=4, lloyd_steps=2,
                        save_path="voronoi_torch.png", thickness_scale=2.0):
    print("Loading and downsampling image...")
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h_full, w_full = img.shape[:2]
    h, w = h_full // downscale_factor, w_full // downscale_factor
    img_small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img_tensor = torch.tensor(img_small).permute(2, 0, 1).to("cuda")  # C, H, W

    print("Sampling and relaxing points...")
    points = torch.rand(N, 2, device='cuda') * torch.tensor([w, h], device='cuda')
    points = lloyd_relaxation(points, w, h, iterations=lloyd_steps)
    points_np = points.cpu().numpy()
    vor = Voronoi(points_np)

    canvas = torch.zeros((3, h, w), dtype=torch.float32, device='cuda')

    print("Filling Voronoi cells with solid average color...")
    for region_index in tqdm(vor.point_region, desc="Filling regions"):
        region = vor.regions[region_index]
        if not region or -1 in region:
            continue
        polygon = np.array([vor.vertices[i] for i in region])
        if polygon.shape[0] < 3:
            continue
        polygon_torch = torch.tensor(polygon, dtype=torch.float32, device='cuda')
        try:
            mask = rasterize_polygon_torch(polygon_torch, w, h)
            if mask.sum() == 0:
                continue
            # Compute the average color of the polygon area
            avg_color = img_tensor[:, mask].mean(dim=1)
            for c in range(3):
                canvas[c][mask] = avg_color[c]
        except Exception:
            continue

    print("Upscaling canvas to full resolution...")
    canvas_np = canvas.cpu().numpy()
    canvas_img = np.moveaxis(canvas_np, 0, -1)
    canvas_upsampled = cv2.resize(canvas_img, (w_full, h_full), interpolation=cv2.INTER_LINEAR)

    print("Drawing edges over upscaled result...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(canvas_upsampled)
    ax.set_xlim(0, w_full)
    ax.set_ylim(h_full, 0)
    ax.axis('off')

    scale_x = w_full / w
    scale_y = h_full / h
    for region_index in tqdm(vor.point_region, desc="Drawing edges"):
        region = vor.regions[region_index]
        if not region or -1 in region:
            continue
        polygon = np.array([vor.vertices[i] for i in region])
        if polygon.shape[0] < 3:
            continue
        polygon = polygon * np.array([scale_x, scale_y])

        for i in range(len(polygon)):
            x0, y0 = polygon[i]
            x1, y1 = polygon[(i + 1) % len(polygon)]
            mx, my = int((x0 + x1) / 2), int((y0 + y1) / 2)
            if 0 <= mx < w_full and 0 <= my < h_full:
                brightness = np.mean(img[my, mx])
                thickness = max(0.3, (1.0 - brightness) * thickness_scale)
                color = img[my, mx]
                ax.plot([x0, x1], [y0, y1], color=color, linewidth=thickness)

    print("Saving final output...")
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Done. Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input .jpg or .png image")
    parser.add_argument("--N", type=int, default=10000, help="Number of Voronoi seed points")
    parser.add_argument("--save", default="voronoi_art.png", help="Path to save output")
    args = parser.parse_args()
    draw_voronoi_torch(args.image_path, N=args.N, save_path=args.save)