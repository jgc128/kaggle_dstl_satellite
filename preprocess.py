import os
import logging
import csv
import pickle

import numpy as np

import shapely
import shapely.wkt
import shapely.affinity

from utils.data import load_grid_sizes, load_polygons, load_images, save_pickle
from utils.matplotlib import matplotlib_setup, plot_image, plot_mask
from config import DATA_DIR, GRID_SIZES_FILENAME, POLYGONS_FILENAME, IMAGES_DIR, DEBUG, DEBUG_IMAGE, \
    IMAGES_METADATA_MASKS_FILENAME
from utils.polygon import create_mask_from_metadata


def create_images_metadata(grid_sizes, images_data, polygons):
    image_ids = sorted(images_data.keys())
    image_ids_set = set(image_ids)

    images_metadata = []

    for image_id, row in polygons.iterrows():
        if image_id not in image_ids_set:
            continue

        grid_size = grid_sizes.loc[image_id]
        x_max = grid_size['x_max']
        y_min = grid_size['y_min']

        width = images_data[image_id].shape[1]
        height = images_data[image_id].shape[0]

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

def normalize_images(images_data):
    nb_channels = images_data[list(images_data.keys())[0]].shape[2]

    channels_mean = []

    for img_id, img_data in images_data.items():
        current_mean = np.mean(img_data, axis=(0,1))
        channels_mean.append(current_mean)

    channels_mean = np.array(channels_mean).mean(axis=0)

    images_data_normalized = {}
    for img_id, img_data in images_data.items():
        images_data_normalized[img_id] = img_data - channels_mean

    return images_data_normalized, channels_mean


def save_data(filename, images, images_metadata, images_masks):
    data_list = [images, images_metadata, images_masks]

    save_pickle(filename, data_list)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    grid_sizes = load_grid_sizes(GRID_SIZES_FILENAME)
    polygons = load_polygons(POLYGONS_FILENAME)

    train_images = sorted(polygons.index)
    images_data = load_images(IMAGES_DIR, target_images=train_images)


    images_metadata = create_images_metadata(grid_sizes, images_data, polygons)
    images_masks = create_classes_masks(images_metadata)

    images_data_normalized, channels_mean = normalize_images(images_data)

    if DEBUG:
        img = images_data[DEBUG_IMAGE]
        img_normalized = images_data_normalized[DEBUG_IMAGE]
        mask = images_masks[DEBUG_IMAGE][1]
        plot_image(img[2900:3200,2000:2300])
        plot_image(img_normalized[2900:3200,2000:2300])
        plot_mask(mask[2900:3200,2000:2300])

    # save everything into a pickled file
    save_data(IMAGES_METADATA_MASKS_FILENAME, images_data_normalized, images_metadata, images_masks, channels_mean)


if __name__ == '__main__':
    main()
