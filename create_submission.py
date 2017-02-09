import logging
import os
import csv

import numpy as np
import pandas as pd

from config import SAMPLE_SUBMISSION_FILENAME, IMAGES_TEST_PREDICTION_MASK_DIR, IMAGES_METADATA_FILENAME, SUBMISSION_DIR
from utils.data import load_sample_submission, load_pickle
from utils.matplotlib import matplotlib_setup, plot_mask
from utils.polygon import create_polygons_from_mask


def save_submission(submission, filename):
    fieldnames = ['ImageId', 'ClassType', 'MultipolygonWKT']

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        writer.writerows(submission)

    logging.info('Submission saved: %s', os.path.basename(filename))


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    # load images metadata
    images_metadata, _, _ = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Images metadata: %s', len(images_metadata))

    sample_submission = load_sample_submission(SAMPLE_SUBMISSION_FILENAME)

    img_id = ''
    img_mask_data = None
    submission = []
    for i, row in sample_submission.iterrows():
        target_img_id = row['ImageId']
        target_class = row['ClassType']
        target_class_id = int(target_class) - 1

        if target_img_id != img_id:
            mask_filename = os.path.join(IMAGES_TEST_PREDICTION_MASK_DIR, target_img_id + '.npy')
            if os.path.isfile(mask_filename):
                img_mask_data = np.load(mask_filename)
                img_id = target_img_id
            else:
                img_mask_data = None
                img_id = ''

        if img_mask_data is not None:
            mask = img_mask_data[target_class_id, :, :]

            poly = create_polygons_from_mask(mask, images_metadata[img_id])
            submission.append((img_id, target_class, poly.wkt))
        else:
            submission.append((img_id, target_class, 'MULTIPOLYGON EMPTY'))

        if (i + 1) % 10 == 0:
            logging.info('Processed: %s/%s [%.2f]', i + 1, len(sample_submission), 100 * (i + 1) / len(sample_submission))

    save_submission(submission, os.path.join(SUBMISSION_DIR, 'simple_model_submission.csv'))


if __name__ == '__main__':
    main()
