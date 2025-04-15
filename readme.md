# 🎨 PolyGlitch: Artistic Geometric Image Stylizers

Bring your photos to life with mathematical elegance. **PolyGlitch** is a suite of 3 cutting-edge tools that transform images into art using triangulation, Voronoi diagrams, and perceptual geometry. Perfect for generative art, visual storytelling, and creative portraiture.

---

## ✨ Tools Included

### 1. 🔺 Triangulated Portrait Generator

Transform grayscale portraits into stylized low-poly artworks. This tool:

- Samples more points in **darker regions** to capture detail.
- Leaves lighter areas with **sparser triangles** for an airy feel.
- Draws triangle edges using soft grayscale lines for a natural look.
- Preserves facial contours, shadows, and expressions with abstract flair.

> **Input:** Grayscale portrait  
> **Output:** Artistic triangle mesh portrait

```bash
python triangulate_portrait.py portrait.jpg 3
```

---

### 2. 🧩 Adaptive Voronoi Mosaic Tessellation

Convert your image into a **Voronoi mosaic** with region-based abstraction:

- Regions are colored by their average pixel value.
- Sampling density is adaptive: **more regions in complex areas**, fewer in flat zones.
- Looks like stained glass or mosaic tiling — bold, geometric, artistic.

> **Input:** Any image  
> **Output:** Adaptive Voronoi mosaic

```bash
python adaptive_voronoi_mosaic.py image.jpg 2
```

---

### 3. 🐝 Voronoi Combs Colorizer

A modern generative twist on Voronoi art:

- Generates **Voronoi comb patterns** with customizable orientation.
- Each region is colorized using nearby pixel context.
- Beautiful for backgrounds, abstract designs, or AI-augmented art.

> **Input:** Any image  
> **Output:** Stylized hex/Voronoi-tiled artwork

```bash
python voronoi_combs.py image.jpg
```

---

## 🧠 Underlying Mathematics

Each tool leverages mathematical ideas from computational geometry and perceptual modeling:

- **Delaunay Triangulation**: Produces triangle meshes from points such that no point lies inside the circumcircle of any triangle.
- **Voronoi Diagrams**: Partitions space into regions based on proximity to sampled points.
- **Adaptive Sampling**: Uses grayscale intensity, edge detection, or entropy maps to determine where to place more points.
- **Color Mapping**: Triangles or regions are filled using mean colors, gradients, or sampling functions.
- **CUDA Acceleration (via PyTorch)**: For fast image processing and region computations, especially on large images.

---

## 💡 Use Cases

- AI Art & Generative Posters  
- Abstract Self-Portraits  
- Data-driven Mosaic Design  
- Profile Pictures with Flair  
- Educational Tools for Geometry + Art

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- Pillow
- SciPy

```bash
pip install torch numpy opencv-python pillow scipy
```

---

## 📷 Gallery

Coming soon: curated artworks from the community — or feel free to share yours!

---

## 🧑‍🎨 Author

Crafted with math, code, and love by [Your Name / GitHub handle]  
For research, art, and everything in between 💫

---

## 📜 License

MIT — use freely, remix creatively.
