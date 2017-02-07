import os
import sys
import logging
import csv
import pickle

from joblib import Parallel, delayed

import numpy as np
from random import shuffle

from utils.data import load_grid_sizes, load_polygons, load_images, save_pickle, get_images_sizes, load_pickle
from utils.matplotlib import matplotlib_setup, plot_image, plot_mask
from config import IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME, TRAIN_PATCHES_COORDINATES_FILENAME
from config import IMAGES_NORMALIZED_FILENAME, IMAGES_MASKS_FILENAME


def sample_patch(img_id, mask_data, patch_size, threshold=0.1, nb_masks=64, nb_tries=99):
    img_height = mask_data.shape[0]
    img_width = mask_data.shape[1]

    masks = []

    for i in range(nb_masks):
        # try to sample a patch with the given threshold 99 times
        for try_number in range(nb_tries):
            img_c1 = (
                np.random.randint(0, img_height - patch_size[0]),
                np.random.randint(0, img_width - patch_size[1])
            )
            img_c2 = (img_c1[0] + patch_size[0], img_c1[1] + patch_size[1])

            # img_patch = img_data[img_c1[0]:img_c2[0], img_c1[1]:img_c2[1], :]
            img_mask = mask_data[img_c1[0]:img_c2[0], img_c1[1]:img_c2[1], :]

            # add only samples with some amount of target class
            mask_fraction = img_mask.sum() / (patch_size[0] * patch_size[1])
            if mask_fraction >= threshold:
                masks.append((img_id, img_c1[0], img_c1[1], img_c2[0], img_c2[1]))
                break

        else:
            logging.warning('Cannot sample patch from image %s', img_id)

    return masks


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    images_data = load_pickle(IMAGES_NORMALIZED_FILENAME)
    images_masks = load_pickle(IMAGES_MASKS_FILENAME)
    logging.info('Images: %s, masks: %s', len(images_data), len(images_masks))

    images_metadata, channels_mean, channels_std = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Images metadata: %s, mean: %s, std: %s',
                 len(images_metadata), channels_mean.shape, channels_std.shape)

    patch_size = (256, 256,)
    nb_channels = 3
    nb_classes = 10

    nb_patches = 1000000
    mask_threshold = 0.15

    images = np.array(list(images_data.keys()))
    classes = np.arange(1, nb_classes + 1)

    images_masks_stacked = {
        img_id: np.stack([images_masks[img_id][target_class] for target_class in classes], axis=-1)
        for img_id in images
        }

    train_patches_coordinates = []
    while len(train_patches_coordinates) < nb_patches:
        try:
            img_id = np.random.choice(images)
            img_mask_data = images_masks_stacked[img_id]

            # # sample 8*32 masks from the same image
            # masks_batch = Parallel(n_jobs=4, verbose=10)(delayed(sample_patch)(
            #     img_id, img_mask_data, patch_size, mask_threshold, 32, 99) for i in range(64))
            #
            # for masks in masks_batch:
            #     train_patches_coordinates.extend(masks)
            masks = sample_patch(img_id, img_mask_data, patch_size, threshold=mask_threshold, nb_masks=32)
            train_patches_coordinates.extend(masks)

            nb_sampled = len(train_patches_coordinates)
            if nb_sampled % 50 == 0:
                logging.info('Sampled %s/%s [%.2f]', nb_sampled, nb_patches, 100 * nb_sampled / nb_patches)

        except KeyboardInterrupt:
            break

    shuffle(train_patches_coordinates)
    logging.info('Sampled patches: %s', len(train_patches_coordinates))

    save_pickle(TRAIN_PATCHES_COORDINATES_FILENAME, train_patches_coordinates)
    logging.info('Saved: %s', os.path.basename(TRAIN_PATCHES_COORDINATES_FILENAME))


if __name__ == '__main__':
    main()
