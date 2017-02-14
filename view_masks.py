import logging
import os

import numpy as np

from config import IMAGES_METADATA_FILENAME, IMAGES_PREDICTION_MASK_DIR, IMAGES_NORMALIZED_FILENAME, \
    IMAGES_MASKS_FILENAME, IMAGES_NORMALIZED_DATA_DIR
from config import IMAGES_METADATA_POLYGONS_FILENAME
from create_submission import create_image_polygons
from utils.data import load_pickle
from utils.matplotlib import matplotlib_setup, plot_image, plot_polygons
from utils.polygon import jaccard_coef, create_mask_from_metadata


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

    images_masks_stacked = None
    if kind == 'train':
        images_masks_stacked = {
            img_id: np.stack([images_masks[img_id][target_class] for target_class in classes], axis=-1)
            for img_id in target_images
            }

    jaccards_reconstructed = []
    jaccards = []
    for img_idx, img_id in enumerate(target_images):
        if img_id != '6110_1_2':
            continue

        mask_filename = os.path.join(IMAGES_PREDICTION_MASK_DIR, img_id + '.npy')
        if not os.path.isfile(mask_filename):
            logging.warning('Cannot find masks for image: %s', img_id)
            continue

        img_data = None
        if kind == 'train':
            img_data = images_data[img_id] * channels_std + channels_mean
        if kind == 'test':
            img_filename = os.path.join(IMAGES_NORMALIZED_DATA_DIR, img_id + '.npy')
            img_data = np.load(img_filename)

        img_metadata = images_metadata[img_id]
        img_mask_pred = np.load(mask_filename)

        if kind == 'train':
            img_poly_true = images_metadata_polygons[img_id]
            img_mask_true = images_masks_stacked[img_id]
        else:
            img_poly_true = None


        img_poly_pred = create_image_polygons(img_mask_pred, img_metadata, scale=False)
        plot_polygons(img_data, img_metadata, img_poly_pred, img_poly_true, title=img_id, show=False)

        # if kind == 'train':
        #     # convert predicted polygons to mask
        #     img_mask_reconstructed = []
        #     for class_type in sorted(img_poly_pred.keys()):
        #         ploy_metadata = {'ploy_scaled': img_poly_pred[class_type].wkt}
        #         img_class_mask_reconstructed = create_mask_from_metadata(img_metadata, ploy_metadata)
        #         img_mask_reconstructed.append(img_class_mask_reconstructed)
        #     img_mask_reconstructed = np.stack(img_mask_reconstructed, axis=-1)
        #
        #
        #     jaccard = jaccard_coef(img_mask_pred, img_mask_true)
        #     jaccard_reconstructed = jaccard_coef(img_mask_reconstructed, img_mask_true)
        #     jaccards.append(jaccard)
        #     jaccards_reconstructed.append(jaccard_reconstructed)
        #     logging.info('Image: %s, jaccard: %s, jaccard reconstructed: %s', img_id, jaccard, jaccard_reconstructed)

    if kind == 'train':
        logging.info('Mean jaccard: %s, Mean jaccard reconstructed: %s', np.mean(jaccards), np.mean(jaccards_reconstructed))

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    kind = 'train'
    main(kind)
