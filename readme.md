# 🎨 PolyGlitch: Artistic Geometric Image Stylizers

Bring your photos to life with mathematical elegance. **PolyGlitch** is a suite of 3 cutting-edge tools that transform images into art using triangulation, Voronoi diagrams, and perceptual geometry. Perfect for generative art, visual storytelling, and creative portraiture.

---
Try out demo at https://huggingface.co/spaces/SChatterjee-huggingface/delaunay-portrait
![huggingface co_spaces_SChatterjee-huggingface_delaunay-portrait(Asus Zenbook Fold) (3)](https://github.com/user-attachments/assets/3d9d215a-9042-4bf4-8ae0-84b4678bbfad)

To understand the theory, you may refer to this lecture by me
https://github.com/SandeepChatterjee66/PolyGlitch/blob/main/Vornoi%20Computation%20SChatterjee_compressed.pdf

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
![Marilyn_Monroe1 0_triangulated_portrait](https://github.com/user-attachments/assets/4c590029-06c7-4e85-a43a-ad8165d44095)
![Srinivasa_Ramanujan-Add _MS_a94_version2_(cropped)6 0_triangulated_portrait](https://github.com/user-attachments/assets/ac2347a8-fb97-42f5-880a-653ce3188f44)
![Screenshot 2025-04-15 1501511 0_triangulated_portrait](https://github.com/user-attachments/assets/7980f531-0577-4335-a458-bd3eb25016ca)
![Screenshot 2025-04-15 1503241 0_triangulated_portrait](https://github.com/user-attachments/assets/21112706-eed3-491e-b4c6-cf5795dd7dee)
![photo_2025-02-21_10-24-331 0_triangulated_portrait](https://github.com/user-attachments/assets/969f0010-46b2-4c3c-801a-5c7b24c90bcd)

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


---![isi-lake5 0_adapt_triangulated](https://github.com/user-attachments/assets/f4a48ea6-9a3c-4278-a984-dd285c044179)
![isi-lake1 5_triangulated](https://github.com/user-attachments/assets/9618af22-128f-45eb-aaae-c21452567ec6)


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
![isi_colordots_voronoi5](https://github.com/user-attachments/assets/7d061f15-5098-448b-b5b3-8ade94df2191)


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

## 📷 Algorithm 


# Triangulation of Images with Adaptive Sampling: Algorithm and Mathematical Analysis

This document provides a rigorous mathematical description and analysis of an algorithm designed for triangulation-based image decomposition. The algorithm is centered around adaptive sampling of image pixels and Delaunay triangulation, with a focus on minimizing distortion and ensuring efficient decomposition based on image content.

## 1. Problem Definition

Given an input image \(I: \mathbb{R}^2 \to \mathbb{R}^3\), where each pixel \(p(x, y) \in I\) has an associated RGB color value, the objective is to partition the image into a set of non-overlapping triangles \(T_i\) such that:
- The number of triangles varies adaptively based on the color contrast and content of the image.
- The triangles in high-detail regions (i.e., areas with significant color gradients or texture) should be smaller, while in smooth areas, they should be larger.
- Each triangle should be filled with a single representative color, and the edges of the triangles should be emphasized with thick white borders.

## 2. Algorithm Overview

### Step 1: Image Preprocessing

1. Convert the image to grayscale \(I_g(x, y)\), representing the intensity of each pixel:
   \[
   I_g(x, y) = \frac{R(x, y) + G(x, y) + B(x, y)}{3}
   \]
   where \(R(x, y)\), \(G(x, y)\), and \(B(x, y)\) are the RGB channels of the image.

2. Compute the gradient magnitude \( \nabla I_g(x, y) \) to detect edges in the image. The gradient magnitude at a point \((x, y)\) is given by:
   \[
   \nabla I_g(x, y) = \sqrt{\left(\frac{\partial I_g}{\partial x}\right)^2 + \left(\frac{\partial I_g}{\partial y}\right)^2}
   \]
   This gives the local edge intensity, which will be used to determine the level of detail in different regions of the image.

### Step 2: Adaptive Sampling

1. Define the **sampling density** function \(D(x, y)\) at each pixel \(p(x, y)\). This function should increase in regions with high gradient magnitude (edges or details) and decrease in smoother regions. A possible form for \(D(x, y)\) is:
   \[
   D(x, y) = \frac{1}{1 + \gamma \nabla I_g(x, y)}
   \]
   where \(\gamma\) is a constant controlling the sensitivity to gradients.

2. Randomly sample a set of points \(P = \{p_i\}\) from the image based on the density function \(D(x, y)\). The probability of selecting a point \(p_i = (x_i, y_i)\) is proportional to the density function \(D(x_i, y_i)\):
   \[
   \text{Prob}(p_i) \propto D(x_i, y_i)
   \]

### Step 3: Delaunay Triangulation

1. Perform **Delaunay triangulation** on the set of sampled points \(P\). The Delaunay criterion ensures that no point lies inside the circumcircle of any triangle in the triangulation. This can be formulated as:
   \[
   \forall T = \{p_i, p_j, p_k\} \in \mathcal{T}, \text{ the circumcenter of } T \text{ lies outside or on the convex hull of } P
   \]
   where \(\mathcal{T}\) is the set of triangles in the triangulation.

2. Let \(\mathcal{T} = \{T_1, T_2, \dots, T_n\}\) represent the set of triangles resulting from the Delaunay triangulation.

### Step 4: Color Assignment and Border Definition

1. For each triangle \(T_i = \{p_a, p_b, p_c\}\), compute the **average color** of the pixels inside the triangle:
   \[
   C(T_i) = \frac{1}{|T_i|} \sum_{p_j \in T_i} I(p_j)
   \]
   where \(I(p_j)\) is the color at pixel \(p_j\) and \(|T_i|\) is the number of pixels in \(T_i\).

2. Assign the color \(C(T_i)\) to the interior of the triangle. To create a **thick white border**, we define the border \(B(T_i)\) as a set of pixels along the edges of the triangle, typically along the pixels that are within one or two pixels of the edge.

3. Finally, the output image is constructed by filling each triangle \(T_i\) with the color \(C(T_i)\) and applying the thick white border to the edges of the triangles.

## 3. Mathematical Analysis

### 3.1 Complexity of Adaptive Sampling

Let \(N\) be the total number of pixels in the image, and let \(N_s\) be the number of points sampled from the image.

- The **sampling process** involves selecting each point based on its density, which takes \(O(N)\) time. The total time complexity for sampling is therefore \(O(N)\).

### 3.2 Complexity of Delaunay Triangulation

The Delaunay triangulation algorithm (using incremental insertion or divide-and-conquer approaches) has a complexity of \(O(N_s \log N_s)\), where \(N_s\) is the number of sampled points. Thus, the complexity for the triangulation step is \(O(N_s \log N_s)\).

### 3.3 Complexity of Color Assignment

For each triangle \(T_i\), the color \(C(T_i)\) is computed by averaging the color of all pixels inside the triangle. In the worst case, this requires examining all pixels, leading to a complexity of \(O(N)\) for computing the color of all triangles.

### 3.4 Overall Time Complexity

The total time complexity of the algorithm is dominated by the Delaunay triangulation step, which is \(O(N_s \log N_s)\), where \(N_s\) is the number of sampled points. Since \(N_s \leq N\), the overall time complexity is:
\[
O(N \log N)
\]
This complexity ensures that the algorithm scales efficiently with the size of the image.

## 4. Conclusion

The proposed algorithm for triangulating images with adaptive sampling based on image gradients is efficient and scalable. By adjusting the sampling density based on edge strength, the algorithm ensures that regions with more details have smaller triangles, while smooth regions are represented with larger triangles. The Delaunay triangulation ensures that the resulting triangles do not overlap and are as equiangular as possible, minimizing distortion.

The overall time complexity of \(O(N \log N)\) ensures that the algorithm can handle large images efficiently, making it suitable for applications in image stylization and artistic rendering.



---

## 🧑‍🎨 Author

Crafted with math, code, and love by Sandeep Chatterjee
For research, art, and everything in between 💫

---

## 📜 License
MIT licenese

MIT — use freely, remix creatively.
