import logging
import os

import numpy as np

from config import IMAGES_METADATA_FILENAME, IMAGES_PREDICTION_MASK_DIR, IMAGES_NORMALIZED_FILENAME, \
    IMAGES_MASKS_FILENAME
from config import IMAGES_METADATA_POLYGONS_FILENAME
from utils.data import load_pickle
from utils.matplotlib import matplotlib_setup, plot_masks_predictions


def main(kind):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    # load images
    images_data = load_pickle(IMAGES_NORMALIZED_FILENAME)
    images_masks = load_pickle(IMAGES_MASKS_FILENAME)
    logging.info('Images: %s, masks: %s', len(images_data), len(images_masks))

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

    if kind == 'test':
        target_images = images_test
    elif kind == 'train':
        target_images = images_train
    else:
        raise ValueError('Unknown kind: {}'.format(kind))

    nb_target_images = len(target_images)
    logging.info('Target images: %s - %s', kind, nb_target_images)

    nb_classes = len(images_masks[images_train[0]])
    classes = np.arange(1, nb_classes + 1)
    images_masks_stacked = {
        img_id: np.stack([images_masks[img_id][target_class] for target_class in classes], axis=-1)
        for img_id in target_images
        }

    for img_idx, img_id in enumerate(target_images):
        mask_filename = os.path.join(IMAGES_PREDICTION_MASK_DIR, img_id + '.npy')
        if not os.path.isfile(mask_filename):
            logging.warning('Cannot find masks for image: %s', img_id)
            continue

        Y_pred = np.load(mask_filename)
        Y_true = images_masks_stacked[img_id]
        X_test = images_data[img_id]
        plot_masks_predictions(X_test, Y_true, Y_pred, channels_mean, channels_std)



if __name__ == '__main__':
    kind = 'train'
    main(kind)