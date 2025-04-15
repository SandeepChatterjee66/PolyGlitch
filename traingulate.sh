for percent in 0.5 1 5; do
    python3 triangulate_image_adaptive.py isi-lake.jpg "$percent"
done