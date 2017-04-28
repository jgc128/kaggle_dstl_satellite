import os
import sys
import csv
import logging

import numpy as np
import pandas as pd

def main(base_dir, source_submission_filename, target_classes):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    logging.info('Source submission: %s', os.path.basename(source_submission_filename))
    source_submission = pd.read_csv(os.path.join(base_dir, source_submission_filename))

    for target_class in target_classes:
        target_submission_filename = os.path.join(
            base_dir, '{}_class_{}.csv'.format(os.path.splitext(source_submission_filename)[0], target_class))

        target_submission = source_submission.copy()
        rows_to_empty = target_submission['ClassType'] != target_class
        target_submission.loc[rows_to_empty, ['MultipolygonWKT']] = 'GEOMETRYCOLLECTION EMPTY'

        target_submission.to_csv(target_submission_filename, index=False)

        logging.info('Class %s processed: %s', target_class, os.path.basename(target_submission_filename))


if __name__ == '__main__':
    base_dir = '/home/aromanov/projects/kaggle_dstl_satellite/data/paper_data/kaggle_separate_submissions/'
    source_submission_filename = 'submission_softmax_pansharpen.csv'
    target_classes = np.arange(1, 10 + 1)

    main(base_dir, source_submission_filename, target_classes)
