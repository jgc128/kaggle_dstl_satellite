import os
import logging

from joblib import Parallel, delayed

import numpy as np

from preprocess import pansharpen_images
from utils.data import load_pickle, load_images, save_pickle
from utils.matplotlib import matplotlib_setup
from config import IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME, IMAGES_THREE_BAND_DIR, \
    IMAGES_NORMALIZED_DATA_DIR, IMAGES_MEANS_STDS_FILENAME, IMAGES_SIXTEEN_BAND_DIR


def normalize_image(img_id, mean_m, std_m, mean_sharpened, std_sharpened):
    img_data_m = load_images(IMAGES_SIXTEEN_BAND_DIR, target_images=[img_id], target_format='M')
    img_data_p = load_images(IMAGES_SIXTEEN_BAND_DIR, target_images=[img_id], target_format='P')

    img_data_sharpened = pansharpen_images(img_data_m, img_data_p)

    img_normalized_m = ((img_data_m[img_id] - mean_m) / std_m).astype(np.float32)
    img_normalized_sharpened = ((img_data_sharpened[img_id] - mean_sharpened) / std_sharpened).astype(np.float32)

    img_filename = os.path.join(IMAGES_NORMALIZED_DATA_DIR, img_id + '.pkl')
    save_pickle(img_filename, [img_normalized_sharpened, img_normalized_m])


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    # load images metadata
    images_metadata = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Metadata: %s', len(images_metadata))

    mean_m, std_m, mean_sharpened, std_sharpened = load_pickle(IMAGES_MEANS_STDS_FILENAME)
    logging.info('Mean & Std: %s - %s, %s - %s', mean_m.shape, std_m.shape, mean_sharpened.shape, std_sharpened.shape)

    images_metadata_polygons = load_pickle(IMAGES_METADATA_POLYGONS_FILENAME)
    logging.info('Polygons metadata: %s', len(images_metadata_polygons))

    images_all = list(images_metadata.keys())
    images_train = list(images_metadata_polygons.keys())
    images_test = sorted(set(images_all) - set(images_train))
    logging.info('Train: %s, test: %s, all: %s', len(images_train), len(images_test), len(images_all))

    target_images = images_all
    logging.info('Target images: %s', len(target_images))

    Parallel(n_jobs=8, verbose=5)(
        delayed(normalize_image)(img_id, mean_m, std_m, mean_sharpened, std_sharpened)
        for img_id in target_images
    )


if __name__ == '__main__':
    main()
