import logging
from datetime import datetime

import os
import pickle

import numpy as np
import pandas as pd
import shapely
import tifffile as tiff
import skimage.transform
import skimage.color

from config import DEBUG, DEBUG_IMAGE, GRID_SIZES_FILENAME, POLYGONS_FILENAME


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


def _get_images_filenames(directory, target_images, target_format=None):
    # TODO: as for now it works only with three band dir
    if target_images is None:
        target_images = sorted([f[:-4] for f in os.listdir(directory)])  # :-4 to discard .tif

    logging.info('Images: %s', len(target_images))

    if target_format is None:
        filename_format = '{0}.tif'
    else:
        filename_format = '{0}_{1}.tif'

    images_filenames = [os.path.join(directory, filename_format.format(img_id, target_format)) for img_id in
                        target_images]

    return images_filenames, target_images


def load_image(img_filename):
    img_data = tiff.imread(img_filename)

    if len(img_data.shape) > 2:
        # transpose to get the (height, width, channels) shape
        img_data = img_data.transpose([1, 2, 0])
    else:
        # if the image does not have channels - add one
        img_data = np.expand_dims(img_data, -1)

    return img_data


def load_images(directory, target_images=None, target_format=None):
    images_filenames, target_images = _get_images_filenames(directory, target_images, target_format)

    images_data = {}
    for i, img_id in enumerate(target_images):
        img_filename = images_filenames[i]
        images_data[img_id] = load_image(img_filename)

        if (i + 1) % 10 == 0:
            logging.info('Loaded: %s/%s [%.2f]', (i + 1), len(target_images), 100 * (i + 1) / len(target_images))

    logging.info('Images data: %s - %s', target_format or '*', len(images_data))
    return images_data


def get_images_sizes(directory, target_images=None, target_format=None):
    images_filenames, target_images = _get_images_filenames(directory, target_images, target_format)

    images_sizes = {}
    for i, img_id in enumerate(target_images):
        img_filename = images_filenames[i]
        img = load_image(img_filename)
        images_sizes[img_id] = img.shape[:2]

        if (i + 1) % 10 == 0:
            logging.info('Loaded: %s/%s [%.2f]', (i + 1), len(target_images), 100 * (i + 1) / len(target_images))

    logging.info('Image sizes: %s - %s', target_format or '*', len(images_sizes))
    return images_sizes


def load_sample_submission(filename):
    data = pd.read_csv(filename)
    logging.info('Sample submission: %s', data.shape)

    return data


def convert_masks_to_softmax(Y, needed_classes=None):
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
    classes_mul = [classes_mul[c + 1] for c in range(nb_classes)]

    Y_prior = Y * classes_mul
    Y_prior *= 2

    # select only needed classes
    if needed_classes is not None:
        Y_prior = Y_prior[:, :, :, needed_classes]

    no_class = np.full((batch_size, img_height, img_width), 1, dtype=np.uint8)
    Y_no_class = np.insert(Y_prior, 0, no_class, axis=3)
    Y_softmax = Y_no_class.argmax(axis=-1)

    return Y_softmax


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


def pansharpen(m, pan, method='browley', W=0.1, all_data=False):
    """From https://www.kaggle.com/resolut/dstl-satellite-imagery-feature-detection/panchromatic-sharpening-simple/discussion"""

    # m.shape - (837, 849, 8)
    # pan.shape - (3348, 3396, 1)

    # M bands order:
    # 0 - Coastal
    # 1 - Blue
    # 2 - Green
    # 3 - Yellow
    # 4 - Red
    # 5 - Red Edge
    # 6 - Near IR1
    # 7 - Near IR2

    # reshape pan
    pan = np.reshape(pan, (pan.shape[0], pan.shape[1]))

    # get M bands
    rgbn = np.empty((m.shape[0], m.shape[1], 4))
    rgbn[:, :, 0] = m[:, :, 4]  # red
    rgbn[:, :, 1] = m[:, :, 2]  # green
    rgbn[:, :, 2] = m[:, :, 1]  # blue
    rgbn[:, :, 3] = m[:, :, 6]  # NIR-1

    rest_m = np.empty((m.shape[0], m.shape[1], 4))
    rest_m[:, :, 0] = m[:, :, 0]  # Coastal
    rest_m[:, :, 1] = m[:, :, 3]  # Yellow
    rest_m[:, :, 2] = m[:, :, 5]  # Red Edge
    rest_m[:, :, 3] = m[:, :, 7]  # NIR-2

    # scaled them
    rgbn_scaled = np.empty((m.shape[0] * 4, m.shape[1] * 4, 4))
    rest_m_scaled = np.empty((m.shape[0] * 4, m.shape[1] * 4, 4))

    for i in range(4):
        img = rgbn[:, :, i]
        scaled = skimage.transform.rescale(img, (4, 4))
        rgbn_scaled[:, :, i] = scaled

        img_rest = rest_m[:, :, i]
        scaled_rest = skimage.transform.rescale(img_rest, (4, 4))
        rest_m_scaled[:, :, i] = scaled_rest

    # check size and crop for pan band
    if pan.shape[0] < rgbn_scaled.shape[0]:
        rgbn_scaled = rgbn_scaled[:pan.shape[0], :, :]
        rest_m_scaled = rest_m_scaled[:pan.shape[0], :, :]
    else:
        pan = pan[:rgbn_scaled.shape[0], :]

    if pan.shape[1] < rgbn_scaled.shape[1]:
        rgbn_scaled = rgbn_scaled[:, :pan.shape[1], :]
        rest_m_scaled = rest_m_scaled[:, :pan.shape[1], :]
    else:
        pan = pan[:, :rgbn_scaled.shape[1]]

    R = rgbn_scaled[:, :, 0]
    G = rgbn_scaled[:, :, 1]
    B = rgbn_scaled[:, :, 2]
    I = rgbn_scaled[:, :, 3]

    image = None

    if method == 'simple_browley':
        all_in = R + G + B
        prod = np.multiply(all_in, pan)

        r = np.multiply(R, pan / all_in)[:, :, np.newaxis]
        g = np.multiply(G, pan / all_in)[:, :, np.newaxis]
        b = np.multiply(B, pan / all_in)[:, :, np.newaxis]

        image = np.concatenate([r, g, b], axis=2)

    if method == 'sample_mean':
        r = 0.5 * (R + pan)[:, :, np.newaxis]
        g = 0.5 * (G + pan)[:, :, np.newaxis]
        b = 0.5 * (B + pan)[:, :, np.newaxis]

        image = np.concatenate([r, g, b], axis=2)

    if method == 'esri':
        ADJ = pan - rgbn_scaled.mean(axis=2)
        r = (R + ADJ)[:, :, np.newaxis]
        g = (G + ADJ)[:, :, np.newaxis]
        b = (B + ADJ)[:, :, np.newaxis]
        i = (I + ADJ)[:, :, np.newaxis]

        image = np.concatenate([r, g, b, i], axis=2)

    if method == 'browley':
        DNF = (pan - W * I) / (W * R + W * G + W * B)

        r = (R * DNF)[:, :, np.newaxis]
        g = (G * DNF)[:, :, np.newaxis]
        b = (B * DNF)[:, :, np.newaxis]
        i = (I * DNF)[:, :, np.newaxis]

        image = np.concatenate([r, g, b, i], axis=2)

    if method == 'hsv':
        hsv = skimage.color.rgb2hsv(rgbn_scaled[:, :, :3])
        hsv[:, :, 2] = pan - I * W
        image = skimage.color.hsv2rgb(hsv)

    # scale the rest by using the simple_browley method
    all_in_rest = np.sum(rest_m_scaled, axis=-1)
    prod = pan / all_in_rest
    image_rest = rest_m_scaled * np.expand_dims(prod, -1)

    if all_data:
        return rgbn_scaled, image, I, image_rest
    else:
        return image, image_rest


def get_train_test_images_ids():
    grid_sizes = load_grid_sizes(GRID_SIZES_FILENAME)
    polygons = load_polygons(POLYGONS_FILENAME)

    all_images = sorted(set(grid_sizes.index))
    train_images = sorted(set(polygons.index))
    test_images = sorted(set(all_images) - set(train_images))

    return all_images, train_images, test_images
