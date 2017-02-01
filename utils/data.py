import logging
import os

import numpy as np
import pandas as pd
import shapely
import tifffile as tiff

from config import DEBUG, DEBUG_IMAGE


def load_grid_sizes(filename):
    data = pd.read_csv(filename, header=0, index_col=0, names=['image_id', 'x_max', 'y_min'])
    logging.info('Grid sizes: %s', data.shape)

    return data


def load_polygons(filename):
    data = pd.read_csv(filename, header=0, index_col=0, names=['image_id', 'class_type', 'polygons'])
    logging.info('Polygons: %s', data.shape)

    return data


def load_images(directory):
    images = sorted([f[:-4] for f in os.listdir(directory)])  # :-4 to discard .tif

    if DEBUG:
        images = [DEBUG_IMAGE, ]
    logging.info('Images: %s', len(images))

    images_filenames = [os.path.join(directory, i + '.tif') for i in images]

    # transpose to get the (height, width, channels) shape
    image_files = {image_id: tiff.imread(images_filenames[i]).transpose([1, 2, 0]) for i, image_id in enumerate(images)}

    return image_files
