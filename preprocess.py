import os
import sys
import logging
import csv
import pickle

import numpy as np

import shapely
import shapely.wkt
import shapely.affinity

from utils.data import load_grid_sizes, load_polygons, load_images, save_pickle, get_images_sizes
from utils.matplotlib import matplotlib_setup, plot_image, plot_mask
from utils.polygon import create_mask_from_metadata
from config import GRID_SIZES_FILENAME, POLYGONS_FILENAME, IMAGES_THREE_BAND_DIR
from config import IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME
from config import IMAGES_NORMALIZED_FILENAME, IMAGES_MASKS_FILENAME


def create_images_metadata(grid_sizes, images_sizes, polygons):
    images_metadata = {}
    images_metadata_polygons = {}

    for img_id, row in grid_sizes.iterrows():
        x_max = row['x_max']
        y_min = row['y_min']

        if img_id not in images_sizes:
            continue

        width = images_sizes[img_id][1]
        height = images_sizes[img_id][0]

        width_prime = width * (width / (width + 1))
        height_prime = height * (height / (height + 1))

        x_scaler = width_prime / x_max
        y_scaler = height_prime / y_min

        image_md = {
            'image_id': img_id,
            'width': width,
            'height': height,
            'x_max': x_max,
            'y_min': y_min,
            'x_scaler': x_scaler,
            'y_scaler': y_scaler,
        }
        images_metadata[img_id] = image_md

        if img_id in polygons.index:
            img_polygons = polygons.loc[img_id]
            images_metadata_polygons[img_id] = {}

            for _, row in img_polygons.iterrows():
                polygon = row['polygons']
                poly = shapely.wkt.loads(polygon)
                ploy_scaled = shapely.affinity.scale(poly, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

                class_type = row['class_type']

                image_md_poly = {
                    'poly': poly.wkt,
                    'ploy_scaled': ploy_scaled.wkt,
                    'class_type': class_type,
                }
                images_metadata_polygons[img_id][class_type] = image_md_poly

    return images_metadata, images_metadata_polygons


def create_classes_masks(images_metadata, images_metadata_polygons):
    masks = {}

    for img_id, img_polygons in images_metadata_polygons.items():
        masks[img_id] = {}

        for class_type, polygon_metadata in img_polygons.items():
            img_metadata = images_metadata[img_id]

            mask = create_mask_from_metadata(img_metadata, polygon_metadata)
            masks[img_id][class_type] = mask

    return masks


def normalize_images(images_data):
    nb_channels = images_data[list(images_data.keys())[0]].shape[2]

    channel_data = [[] for i in range(nb_channels)]
    for img_id, img_data in images_data.items():
        for i in range(nb_channels):
            img_channel_data = img_data[:, :, i].flatten()
            channel_data[i].append(img_channel_data)

    channel_data = np.array([np.concatenate(chds, axis=0) for chds in channel_data])

    channels_mean = channel_data.mean(axis=1).astype(np.float32)
    channels_std = channel_data.std(axis=1).astype(np.float32)

    images_data_normalized = {}
    for img_id, img_data in images_data.items():
        images_data_normalized[img_id] = ((img_data - channels_mean) / channels_std).astype(np.float32)

    return images_data_normalized, channels_mean, channels_std


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    grid_sizes = load_grid_sizes(GRID_SIZES_FILENAME)
    polygons = load_polygons(POLYGONS_FILENAME)

    all_images = sorted(set(grid_sizes.index))
    train_images = sorted(set(polygons.index))
    test_images = sorted(set(all_images) - set(train_images))
    logging.info('Train: %s, Test: %s, All: %s', len(train_images), len(test_images), len(all_images))

    # create images metadata
    images_sizes = get_images_sizes(IMAGES_THREE_BAND_DIR, target_images=all_images)
    images_metadata, images_metadata_polygons = create_images_metadata(grid_sizes, images_sizes, polygons)
    logging.info('Metadata: %s, polygons metadata: %s', len(images_metadata), len(images_metadata_polygons))

    # load train images, create masks and calc mean and std
    images_data = load_images(IMAGES_THREE_BAND_DIR, target_images=train_images)
    images_masks = create_classes_masks(images_metadata, images_metadata_polygons)
    images_data_normalized, channels_mean, channels_std = normalize_images(images_data)

    # save everything
    save_pickle(IMAGES_METADATA_FILENAME, [images_metadata, channels_mean, channels_std])
    save_pickle(IMAGES_METADATA_POLYGONS_FILENAME, images_metadata_polygons)

    save_pickle(IMAGES_MASKS_FILENAME, images_masks)
    save_pickle(IMAGES_NORMALIZED_FILENAME, images_data_normalized)


if __name__ == '__main__':
    main()
