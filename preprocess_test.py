import os
import logging

from joblib import Parallel, delayed

import numpy as np

from preprocess import pansharpen_images, normalize_images
from utils.data import load_pickle, load_images, save_pickle, get_train_test_images_ids
from utils.matplotlib import matplotlib_setup
from config import IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME, IMAGES_THREE_BAND_DIR, \
    IMAGES_NORMALIZED_DATA_DIR, IMAGES_MEANS_STDS_FILENAME, IMAGES_SIXTEEN_BAND_DIR


def load_and_normalize_image(img_id, mean_sharpened, std_sharpened):
    img_data_m = load_images(IMAGES_SIXTEEN_BAND_DIR, target_images=[img_id], target_format='M')
    img_data_p = load_images(IMAGES_SIXTEEN_BAND_DIR, target_images=[img_id], target_format='P')

    img_data_sharpened = pansharpen_images(img_data_m, img_data_p)

    img_normalized_sharpened = normalize_images(img_data_sharpened, mean_sharpened, std_sharpened)[img_id]

    img_filename = os.path.join(IMAGES_NORMALIZED_DATA_DIR, img_id + '.pkl')
    save_pickle(img_filename, img_normalized_sharpened)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    mean_sharpened, std_sharpened = load_pickle(IMAGES_MEANS_STDS_FILENAME)
    logging.info('Mean: %s, Std: %s', mean_sharpened.shape, std_sharpened.shape)

    images_all, images_train, images_test = get_train_test_images_ids()

    logging.info('Train: %s, test: %s, all: %s', len(images_train), len(images_test), len(images_all))

    target_images = images_all
    logging.info('Target images: %s', len(target_images))

    Parallel(n_jobs=8, verbose=5)(
        delayed(load_and_normalize_image)(img_id, mean_sharpened, std_sharpened)
        for img_id in target_images
    )


if __name__ == '__main__':
    main()
