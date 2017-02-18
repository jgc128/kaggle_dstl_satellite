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


def load_image(img_filename):
    img_data = tiff.imread(img_filename)

    if len(img_data.shape) > 2:
        img_data = img_data.transpose([1, 2, 0])
    else:
        img_data = np.expand_dims(img_data, -1)

    return img_data

def load_images(directory, target_images=None):
    images_filenames, target_images = _get_images_filenames(directory, target_images)

    # transpose to get the (height, width, channels) shape
    images_data = {}
    for i, img_id in enumerate(target_images):
        img_filename = images_filenames[i]
        images_data[img_id] = load_image(img_filename)

        if (i + 1) % 10 == 0:
            logging.info('Loaded: %s/%s [%.2f]', (i + 1), len(target_images), 100 * (i + 1) / len(target_images))

    logging.info('Images data: %s', len(images_data))
    return images_data



def get_images_sizes(directory, target_images=None):
    images_filenames, target_images = _get_images_filenames(directory, target_images)

    images_sizes = {}
    for i, img_id in enumerate(target_images):
        img_filename = images_filenames[i]
        img = load_image(img_filename)
        images_sizes[img_id] = img.shape[:2]

    logging.info('Image sizes: %s', len(images_sizes))
    return images_sizes


def load_sample_submission(filename):
    data = pd.read_csv(filename)
    logging.info('Sample submission: %s', data.shape)

    return data


def convert_mask_to_one_hot(Y):
    batch_size, img_height, img_width, nb_classes = Y.shape

    # Classes:
    # 1 - Buildings
    # 2 - Misc. Manmade structures
    # 3 - Road
    # 4 - Track
    # 5 - Trees
    # 6 - Crops
    # 7 - Waterway
    # 8 - Standing water
    # 9 - Vehicle Large
    # 10 - Vehicle Small

    classes_mul = {
        1: 5,
        2: 10,
        3: 5,
        4: 5,
        5: 10,
        6: 1,
        7: 5,
        8: 5,
        9: 10,
        10: 10,
    }
    classes_mul = [classes_mul[c+1] for c in range(nb_classes)]

    Y_prior = Y * classes_mul

    Y_prior *= 2
    no_class = np.full((batch_size, img_height, img_width), 1, dtype=np.uint8)
    Y_one_hot = np.insert(Y_prior, 0, no_class, axis=3)
    Y_one_hot = Y_one_hot.argmax(axis=-1)

    return Y_one_hot


def convert_softmax_to_masks(Y_probs):
    nb_classes = Y_probs.shape[2] - 1
    mask_height = Y_probs.shape[0]
    mask_width = Y_probs.shape[1]

    Y_classes = np.argmax(Y_probs, axis=-1)

    masks = []
    for i in range(1, nb_classes + 1):
        m = np.zeros((mask_height, mask_width), dtype=np.uint8)
        m[np.where(Y_classes == i)] = 1
        masks.append(m)

    mask_joined = np.stack(masks, axis=-1)
    return mask_joined
