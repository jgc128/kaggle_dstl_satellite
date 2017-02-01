import os
import logging
import csv

import numpy as np

import shapely
import shapely.wkt
import shapely.affinity

from utils.data import load_grid_sizes, load_polygons, load_images
from utils.matplotlib import matplotlib_setup, plot_image
from config import DATA_DIR, GRID_SIZES_FILENAME, POLYGONS_FILENAME, IMAGES_DIR, DEBUG, DEBUG_IMAGE


def generate_images_metadata(grid_sizes, images, polygons):
    image_ids = sorted(images.keys())
    image_ids_set = set(image_ids)

    images_metadata = []

    for image_id, row in polygons.iterrows():
        if image_id not in image_ids_set:
            continue

        grid_size = grid_sizes.loc[image_id]
        x_max = grid_size['x_max']
        y_min = grid_size['y_min']

        width = images[image_id].shape[1]
        height = images[image_id].shape[0]

        width_prime = width * (width / (width + 1))
        height_prime = height * (height / (height + 1))

        x_scaler = width_prime / x_max
        y_scaler = height_prime / y_min

        polygon = row['polygons']
        poly = shapely.wkt.loads(polygon)
        ploy_scaled = shapely.affinity.scale(poly, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

        class_type = row['class_type']
        image_md = {
            'image_id': image_id,
            'width': width,
            'height': height,
            'x_max': x_max,
            'y_min': y_min,
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
            'poly': poly.wkt,
            'ploy_scaled': ploy_scaled.wkt,
            'class_type': class_type,
        }

        images_metadata.append(image_md)

    return images_metadata

def plot_polygons(image, image_metadata):
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.path import Path
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection

    # http://matplotlib.org/users/path_tutorial.html
    # http://matplotlib.org/examples/api/patch_collection.html

    # def create_plt_patch(im, coords):
    #     patch = np.array(coords)
    #     patch[:, 0] /= im['height']
    #     patch[:, 1] /= im['width']
    #     plt_poly = Polygon(patch, True)
    #
    #     return plt_poly

    int_coords = lambda x: np.array(x).round().astype(np.int32)

    taget_type = 1

    image_metadata = [im for im in image_metadata if im['class_type'] == taget_type]

    # create polygons
    plt_polygons = []
    plt_colors = []
    for im in image_metadata:
        multipoly = shapely.wkt.loads(im['ploy_scaled'])

        exteriors = [int_coords(poly.exterior.coords) for poly in multipoly]
        interiors = [int_coords(pi.coords) for poly in multipoly for pi in poly.interiors]
        for poly in exteriors:
            plt_poly = patches.Polygon(poly, closed=True)
            plt_polygons.append(plt_poly)
            plt_colors.append('black')

        for poly in interiors:
            plt_poly = patches.Polygon(poly, closed=True)
            plt_polygons.append(plt_poly)
            plt_colors.append('red')


    # plt_patches = PatchCollection(plt_polygons)
    # plt_patches.set_array(np.full((len(plt_polygons),), 1))
    # ax.add_collection(plt_patches)

    fig, ax = plt.subplots()
    plt_patches = PatchCollection(plt_polygons, facecolors=plt_colors)
    ax.add_collection(plt_patches)

    ax.set_xlim(0, image_metadata[0]['width'])
    ax.set_ylim(0, image_metadata[0]['height'])
    ax.set_axis_off()

    plt.show()

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    grid_sizes = load_grid_sizes(GRID_SIZES_FILENAME)
    polygons = load_polygons(POLYGONS_FILENAME)
    images = load_images(IMAGES_DIR)

    images_metadata = generate_images_metadata(grid_sizes, images, polygons)

    if DEBUG:
        image = images[DEBUG_IMAGE]
        # plot_image(image[2900:3200,2000:2300]) #

        image_metadata = [im for im in images_metadata if im['image_id'] == DEBUG_IMAGE]
        plot_polygons(image, image_metadata)

    pass

if __name__ == '__main__':
    main()
