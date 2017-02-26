import logging

import numpy as np

from config import IMAGES_NORMALIZED_FILENAME, IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME, \
    IMAGES_MASKS_FILENAME
from utils.data import load_pickle, convert_masks_to_softmax, convert_softmax_to_masks, get_train_test_images_ids
from utils.matplotlib import matplotlib_setup, plot_image, plot_mask, plot_two_masks
from utils.polygon import stack_masks


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    # load images
    images_data = load_pickle(IMAGES_NORMALIZED_FILENAME)
    images_masks = load_pickle(IMAGES_MASKS_FILENAME)
    logging.info('Images: %s, masks: %s', len(images_data), len(images_masks))

    images_all, images_train, images_test = get_train_test_images_ids()
    logging.info('Train: %s, test: %s, all: %s', len(images_train), len(images_test), len(images_all))

    nb_classes = len(images_masks[images_train[0]])
    classes = np.arange(1, nb_classes + 1)

    images_masks_stacked = stack_masks(images_train, images_masks, classes)
    logging.info('Masks stacked: %s', len(images_masks_stacked))

    # # plot_image(images_data[test_img_id] * channels_std + channels_mean)
    # for img_id in images_train:
    #     if images_masks[img_id][9].sum() > 0:
    #         logging.info('Image: %s, Vehicle: %s - %s, Crops: %s, Trees: %s',
    #                      img_id,
    #                      images_masks[img_id][9].sum(), images_masks[img_id][10].sum(),
    #                      images_masks[img_id][6].sum(), images_masks[img_id][5].sum(),
    #                      )

    # test on trees/crops
    test_img_id = '6120_2_2'

    # convert ot one hot and back
    masks_softmax = convert_masks_to_softmax(np.expand_dims(images_masks_stacked[test_img_id], 0))[0]
    masks_softmax_flattened = np.reshape(masks_softmax, (-1))
    mask_probs = np.zeros((masks_softmax_flattened.shape[0], 11))
    mask_probs[np.arange(masks_softmax_flattened.shape[0]), masks_softmax_flattened] = 1
    mask_probs = np.reshape(mask_probs, (masks_softmax.shape[0], masks_softmax.shape[1], 11))
    masks_converted = convert_softmax_to_masks(mask_probs)

    classes_to_plot = [
        [5, 'Trees'],
        [6, 'Crops'],
        # [3, 'Roads'],
        # [4, 'Track'],
    ]
    for class_type, class_title in classes_to_plot:
        titles = [class_title + ' - ' + t for t in ['True', 'Reconstructed']]
        plot_two_masks(images_masks[test_img_id][class_type][:750,:750], masks_converted[:750,:750, class_type - 1], titles=titles)



if __name__ == '__main__':
    main()
