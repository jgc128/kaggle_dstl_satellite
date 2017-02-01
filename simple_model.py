import logging
import os

from utils.data import load_grid_sizes, load_polygons, load_images
from utils.matplotlib import matplotlib_setup
from config import DATA_DIR, GRID_SIZES_FILENAME, POLYGONS_FILENAME, IMAGES_DIR


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    grid_sizes = load_grid_sizes(GRID_SIZES_FILENAME)
    polygons = load_polygons(POLYGONS_FILENAME)

    images = load_images(IMAGES_DIR, nb_images=-1)

    shapes = set([img.shape for img in images.values()])
    print(shapes)

if __name__ == '__main__':
    main()
