import logging
import os

import numpy as np
import cv2

from config import IMAGES_METADATA_FILENAME, IMAGES_PREDICTION_MASK_DIR, \
    IMAGES_MASKS_FILENAME, IMAGES_NORMALIZED_DATA_DIR, IMAGES_NORMALIZED_M_FILENAME, \
    IMAGES_NORMALIZED_SHARPENED_FILENAME, IMAGES_MEANS_STDS_FILENAME, CLASSES_NAMES
from config import IMAGES_METADATA_POLYGONS_FILENAME
from create_submission import create_image_polygons
from utils.data import load_pickle, get_train_test_images_ids
from utils.matplotlib import matplotlib_setup, plot_image, plot_polygons, plot_two_masks
from utils.polygon import jaccard_coef, create_mask_from_polygons, simplify_mask, stack_masks


def main(kind):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    images_data_m = load_pickle(IMAGES_NORMALIZED_M_FILENAME)
    images_data_sharpened = load_pickle(IMAGES_NORMALIZED_SHARPENED_FILENAME)
    logging.info('Images: %s, %s', len(images_data_m), len(images_data_sharpened))

    images_masks = load_pickle(IMAGES_MASKS_FILENAME)
    logging.info('Masks: %s', len(images_masks))

    images_metadata = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Metadata: %s', len(images_metadata))

    images_metadata_polygons = load_pickle(IMAGES_METADATA_POLYGONS_FILENAME)
    logging.info('Polygons metadata: %s', len(images_metadata_polygons))

    mean_m, std_m, mean_sharpened, std_sharpened = load_pickle(IMAGES_MEANS_STDS_FILENAME)
    logging.info('Mean & Std: %s - %s, %s - %s', mean_m.shape, std_m.shape, mean_sharpened.shape, std_sharpened.shape)

    images_all, images_train, images_test = get_train_test_images_ids()
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
        images_masks_stacked = stack_masks(target_images, images_masks, classes)
        logging.info('Masks stacked: %s', len(images_masks_stacked))

    jaccards = []
    model_name = 'combined_model_jaccard_softmax_without_small'
    for img_idx, img_id in enumerate(target_images):
        if img_id != '6040_4_4':
            continue

        mask_filename = os.path.join(IMAGES_PREDICTION_MASK_DIR, '{0}_{1}.npy'.format(img_id, model_name))
        if not os.path.isfile(mask_filename):
            logging.warning('Cannot find masks for image: %s', img_id)
            continue

        img_data = None
        if kind == 'train':
            img_data = images_data_sharpened[img_id] * std_sharpened + mean_sharpened
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
            img_mask_true = None

        # plot_image(img_data)

        img_mask_pred_simplified = simplify_mask(img_mask_pred, kernel_size=5)

        # if kind == 'train':
        #     for i, class_name in enumerate(CLASSES_NAMES):
        #         if img_mask_true[:,:,i].sum() > 0:
        #             plot_two_masks(img_mask_true[:,:,i], img_mask_pred[:,:,i],
        #                 titles=['Ground Truth - {}'.format(class_name), 'Prediction - {}'.format(class_name)])
        #             # plot_two_masks(img_mask_pred[:,:,i], img_mask_pred_simplified[:,:,i],
        #             #     titles=['Ground Truth - {}'.format(class_name), 'Prediction - {}'.format(class_name)])

        img_poly_pred = create_image_polygons(img_mask_pred, img_metadata, scale=False)
        plot_polygons(img_data, img_metadata, img_poly_pred, img_poly_true, title=img_id, show=False)

        if kind == 'train':
            # convert predicted polygons to mask
            jaccard = jaccard_coef(img_mask_pred, img_mask_true)
            jaccards.append(jaccard)
            logging.info('Image: %s, jaccard: %s', img_id, jaccard)

    if kind == 'train':
        logging.info('Mean jaccard: %s', np.mean(jaccards))

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':
    kind = 'train'
    main(kind)
