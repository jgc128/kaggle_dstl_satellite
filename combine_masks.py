import os
import logging

import numpy as np

from config import IMAGES_PREDICTION_MASK_DIR
from utils.data import get_train_test_images_ids, load_prediction_mask, save_prediction_mask
from utils.matplotlib import matplotlib_setup


def main(model_names, output_name):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    logging.info('Combining masks:')
    for mn in model_names:
        logging.info(' - %s', mn)

    logging.info('Output masks: %s', output_name)

    images_all, images_train, images_test = get_train_test_images_ids()
    logging.info('Train: %s, test: %s, all: %s', len(images_train), len(images_test), len(images_all))

    target_images = images_test
    logging.info('Target images: %s', len(target_images))
    for img_number, img_id in enumerate(target_images):
        # if img_id != '6060_2_3':
        #     continue

        img_masks = [load_prediction_mask(IMAGES_PREDICTION_MASK_DIR, img_id, model_name) for model_name in model_names]
        img_masks = np.array(img_masks)

        img_masks_combined = np.sum(img_masks, axis=0)

        save_prediction_mask(IMAGES_PREDICTION_MASK_DIR, img_masks_combined, img_id, output_name)

        logging.info('Combined: %s/%s [%.2f]',
                     img_number + 1, len(target_images), 100 * (img_number + 1) / len(target_images))


if __name__ == '__main__':
    model_names = [
        'softmax_pansharpen_tiramisu_big_objects',
        'softmax_pansharpen_tiramisu_small_objects',
    ]
    output_name = 'softmax_pansharpen_tiramisu'

    main(model_names, output_name)
