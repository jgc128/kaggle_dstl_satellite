import os
import logging
import csv
import pickle

import numpy as np

import shapely
import shapely.wkt
import shapely.affinity

from utils.data import load_grid_sizes, load_polygons, load_images
from utils.matplotlib import matplotlib_setup, plot_image, plot_mask
from config import DATA_DIR, GRID_SIZES_FILENAME, POLYGONS_FILENAME, IMAGES_DIR, DEBUG, DEBUG_IMAGE, \
    IMAGES_METADATA_MASKS_FILENAME
from utils.polygon import create_mask_from_metadata


def create_images_metadata(grid_sizes, images, polygons):
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

    logging.info('Metadata: %s', len(images_metadata))

    return images_metadata


def create_classes_masks(images_metadata):
    masks = {}

    for im in images_metadata:
        image_id = im['image_id']
        if image_id not in masks:
            masks[image_id] = {}

        mask = create_mask_from_metadata(im)
        class_type = im['class_type']

        masks[image_id][class_type] = mask

    logging.info('Masks: %s', len(masks))

    return masks

def save_data(filename, images, images_metadata, images_masks):
    data_list = [images, images_metadata, images_masks]

    with open(filename, 'wb') as f:
        pickle.dump(data_list, f)

    logging.info('Saved: %s', os.path.basename(filename))


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    grid_sizes = load_grid_sizes(GRID_SIZES_FILENAME)
    polygons = load_polygons(POLYGONS_FILENAME)
    images = load_images(IMAGES_DIR)

    images_metadata = create_images_metadata(grid_sizes, images, polygons)
    images_masks = create_classes_masks(images_metadata)

    # if DEBUG:
    #     image = images[DEBUG_IMAGE]
    #     mask = images_masks[DEBUG_IMAGE][1]
    #     plot_image(image[2900:3200,2000:2300])
    #     plot_mask(mask[2900:3200,2000:2300])

    # save everything into a pickled file
    save_data(IMAGES_METADATA_MASKS_FILENAME, images, images_metadata, images_masks)

if __name__ == '__main__':
    main()
