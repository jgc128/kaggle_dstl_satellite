import logging
from datetime import datetime

import os
import pickle

import numpy as np
import pandas as pd
import shapely
import tifffile as tiff

from config import DEBUG, DEBUG_IMAGE


def load_pickle(filename):
    data_start = datetime.now()

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    data_end = datetime.now()
    logging.info('Loaded: %s in %s', os.path.basename(filename), data_end - data_start)

    return data


def save_pickle(filename, obj):
    data_start = datetime.now()

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

    data_end = datetime.now()
    logging.info('Saved: %s in %s', os.path.basename(filename), data_end - data_start)


def load_grid_sizes(filename):
    data = pd.read_csv(filename, header=0, index_col=0, names=['image_id', 'x_max', 'y_min'])
    logging.info('Grid sizes: %s', data.shape)

    return data


def load_polygons(filename):
    data = pd.read_csv(filename, header=0, index_col=0, names=['image_id', 'class_type', 'polygons'])
    logging.info('Polygons: %s', data.shape)

    return data


def _get_images_filenames(directory, target_images):
    if target_images is None:
        target_images = sorted([f[:-4] for f in os.listdir(directory)])  # :-4 to discard .tif

    if DEBUG:
        target_images = [DEBUG_IMAGE, ]

    logging.info('Images: %s', len(target_images))

    images_filenames = [os.path.join(directory, i + '.tif') for i in target_images]

    return images_filenames, target_images


def load_images(directory, target_images=None):
    images_filenames, target_images = _get_images_filenames(directory, target_images)

    # transpose to get the (height, width, channels) shape
    images_data = {image_id: tiff.imread(images_filenames[i]).transpose([1, 2, 0]) for i, image_id in
                   enumerate(target_images)}

    logging.info('Images data: %s', len(images_data))
    return images_data


def get_images_sizes(directory, target_images=None):
    images_filenames, target_images = _get_images_filenames(directory, target_images)

    images_sizes = {}
    for i, img_id in enumerate(target_images):
        img = tiff.imread(images_filenames[i]).transpose([1, 2, 0])
        images_sizes[img_id] = img.shape[:2]

    logging.info('Image sizes: %s', len(images_sizes))
    return images_sizes
