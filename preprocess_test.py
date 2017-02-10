import os
import sys
import logging
import csv
import pickle

import numpy as np

from utils.data import load_pickle, load_images
from utils.matplotlib import matplotlib_setup
from config import IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME, IMAGES_THREE_BAND_DIR, \
    IMAGES_NORMALIZED_DATA_DIR


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    # load images metadata
    images_metadata, channels_mean, channels_std = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Images metadata: %s, mean: %s, std: %s', len(images_metadata), channels_mean.shape,
                 channels_std.shape)

    images_metadata_polygons = load_pickle(IMAGES_METADATA_POLYGONS_FILENAME)
    logging.info('Polygons metadata: %s', len(images_metadata_polygons))

    images_all = list(images_metadata.keys())
    images_train = list(images_metadata_polygons.keys())
    images_test = sorted(set(images_all) - set(images_train))
    logging.info('Train: %s, test: %s, all: %s', len(images_train), len(images_test), len(images_all))

    # let's try to read all images!
    images_data = load_images(IMAGES_THREE_BAND_DIR, target_images=images_all)

    for i, (img_id, img_data) in enumerate(images_data.items()):
        img_normalized = ((img_data - channels_mean) / channels_std).astype(np.float32)
        img_filename = os.path.join(IMAGES_NORMALIZED_DATA_DIR, img_id + '.npy')
        np.save(img_filename, img_normalized)

        if (i + 1) % 50 == 0:
            logging.info('Processed: %s/%s [%.2f]', (i+1), len(images_data), 100 * (i + 1) / len(images_data))


if __name__ == '__main__':
    main()
