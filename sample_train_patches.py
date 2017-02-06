import os
import sys
import logging
import csv
import pickle

import numpy as np

from utils.data import load_grid_sizes, load_polygons, load_images, save_pickle, get_images_sizes, load_pickle
from utils.matplotlib import matplotlib_setup, plot_image, plot_mask
from config import IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME, TRAIN_PATCHES_FILENAME
from config import IMAGES_NORMALIZED_FILENAME, IMAGES_MASKS_FILENAME


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

    nb_patches = 100000
    mask_threshold = 0.15

    images = np.array(list(images_data.keys()))
    classes = np.arange(1, nb_classes+1)

    train_patches = []
    train_mask = []
    while len(train_patches) < nb_patches:
        try:
            img_id = np.random.choice(images)

            img_data = images_data[img_id]
            img_mask_data = np.stack([images_masks[img_id][target_class] for target_class in classes], axis=-1)

            img_height = img_data.shape[0]
            img_width = img_data.shape[1]

            img_c1 = (
                np.random.randint(0, img_height - patch_size[0]),
                np.random.randint(0, img_width - patch_size[1])
            )
            img_c2 = (img_c1[0] + patch_size[0], img_c1[1] + patch_size[1])

            img_patch = img_data[img_c1[0]:img_c2[0], img_c1[1]:img_c2[1], :]
            img_mask = img_mask_data[img_c1[0]:img_c2[0], img_c1[1]:img_c2[1], :]

            # add only samples with some amount of target class
            mask_fraction = img_mask.sum() / (patch_size[0] * patch_size[1])
            if mask_fraction >= mask_threshold:
                train_patches.append(img_patch)
                train_mask.append(img_mask)

                if len(train_patches) % 50 == 0:
                    logging.info('Sampled %s/%s [%.2f]', len(train_patches), nb_patches, 100 * len(train_patches) / nb_patches)

        except KeyboardInterrupt:
            break

    train_patches = np.array(train_patches)
    train_mask = np.array(train_mask)
    logging.info('Patches: %s, mask: %s', train_patches.shape, train_mask.shape)

    np.savez_compressed(TRAIN_PATCHES_FILENAME, train_patches=train_patches, train_mask=train_mask)
    logging.info('Saved: %s', os.path.basename(TRAIN_PATCHES_FILENAME))

if __name__ == '__main__':
    main()
