import logging
import os
import csv

from joblib import Parallel, delayed

import numpy as np
import pandas as pd


from config import SAMPLE_SUBMISSION_FILENAME, IMAGES_PREDICTION_MASK_DIR, IMAGES_METADATA_FILENAME, SUBMISSION_DIR
from utils.data import load_sample_submission, load_pickle
from utils.matplotlib import matplotlib_setup, plot_mask
from utils.polygon import create_polygons_from_mask


def save_submission(polygons, submission_order, filename):
    fieldnames = ['ImageId', 'ClassType', 'MultipolygonWKT']

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for img_id, class_type in submission_order:
            if img_id in polygons and class_type in polygons[img_id]:
                poly_str = polygons[img_id][class_type].wkt
            else:
                poly_str = 'MULTIPOLYGON EMPTY'

            writer.writerow((img_id, class_type, poly_str))

    logging.info('Submission saved: %s', os.path.basename(filename))


def create_image_polygons(img_masks, img_metadata, scale=True):
    if img_masks is None:
        return {}

    nb_classes = img_masks.shape[2]

    polygons = Parallel(n_jobs=4)(
        delayed(create_polygons_from_mask)(img_masks[:, :, class_id], img_metadata, scale) for class_id in range(nb_classes)
    )

    polygons_dict = {class_id+1:polygons[class_id] for class_id in range(nb_classes)}
    return polygons_dict

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    # load images metadata
    images_metadata, _, _ = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Images metadata: %s', len(images_metadata))

    sample_submission = load_sample_submission(SAMPLE_SUBMISSION_FILENAME)
    submission_order = [(row['ImageId'], row['ClassType']) for i, row in sample_submission.iterrows()]

    target_images = sorted(set([r[0] for r in submission_order]))
    logging.info('Target images: %s', len(target_images))

    polygons = {}
    for i, img_id in enumerate(target_images):
        img_metadata = images_metadata[img_id]

        mask_filename = os.path.join(IMAGES_PREDICTION_MASK_DIR, img_id + '.npy')
        if os.path.isfile(mask_filename):
            img_mask = np.load(mask_filename)
        else:
            img_mask = None

        img_polygons = create_image_polygons(img_mask, img_metadata, scale=True)
        polygons[img_id] = img_polygons

        if (i + 1) % 10 == 0:
            logging.info('Processed images: %s/%s [%.2f]', i + 1, len(target_images), 100 * (i + 1) / len(target_images))


    submission_filename = os.path.join(SUBMISSION_DIR, 'simple_model_submission.csv')
    save_submission(polygons, submission_order, submission_filename)


if __name__ == '__main__':
    main()
