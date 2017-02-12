import logging
import itertools

import numpy as np

from config import IMAGES_MASKS_FILENAME, IMAGES_METADATA_FILENAME
from config import IMAGES_NORMALIZED_FILENAME
from utils.data import load_pickle
from utils.matplotlib import matplotlib_setup


def get_intersection_area(mask1, mask2, mode='union'):
    intersection = np.bitwise_and(mask1, mask2)
    union = np.bitwise_or(mask1, mask2)

    if mode == 'union':
        denominator = union.sum()
    elif mode == 'mask1':
        denominator = mask1.sum()
    else:
        raise ValueError('Mode unknown: {}'.format(mode))

    intersection_fraction = intersection.sum() / denominator

    return intersection_fraction

def plot_intersections(intersection_data, id2class):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(7, 7))

    img = ax.imshow(intersection_data, interpolation='nearest', cmap=plt.get_cmap('Reds'),
                    vmin=intersection_data.min(), vmax=intersection_data.max())
    fig.colorbar(img)

    ticks = range(len(id2class))
    labels = [id2class[i+1] for i in ticks]

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(False)
    #     ax.set_title(title)

    fig.tight_layout(pad=2)

    plt.show()

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    images_masks = load_pickle(IMAGES_MASKS_FILENAME)
    logging.info('Masks: %s', len(images_masks))

    images = sorted(images_masks.keys())
    logging.info('Images: %s', len(images))

    classes = [
        'Buildings',
        'Misc. Manmade structures ',
        'Road ',
        'Track',
        'Trees',
        'Crops',
        'Waterway ',
        'Standing water',
        'Vehicle Large',
        'Vehicle Small',
    ]
    id2class = {i+1:c for i, c in enumerate(classes)}
    nb_classes = len(classes)
    logging.info('Classes: %s', nb_classes)

    intersection_data = np.zeros((nb_classes, nb_classes), dtype=np.float32)
    for i, j in itertools.permutations(range(1, nb_classes+1), 2):
        logging.info('Comparing classes %s and %s', i, j)

        i_masks = [images_masks[img_id][i] for img_id in images]
        j_masks = [images_masks[img_id][j] for img_id in images]

        intersections = []
        for k, img_id in enumerate(images):
            intersection = get_intersection_area(i_masks[k],j_masks[k], mode='mask1')
            intersections.append(intersection)

        mean_intersection = np.mean(np.nan_to_num(intersections))
        intersection_data[i-1,j-1] = mean_intersection

    plot_intersections(intersection_data, id2class)


if __name__ == '__main__':
    main()
