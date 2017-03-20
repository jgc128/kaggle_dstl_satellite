import os
import sys
import csv
import logging

import shapely
import shapely.wkt
import shapely.affinity

from config import SUBMISSION_DIR


def fix_polygon(poly):
    polygons = shapely.wkt.loads(poly).buffer(0.00001)
    return shapely.wkt.dumps(polygons)


def load_submission(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)

        polygons = [r for r in reader]

    return polygons


def save_submission(filename, submission):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(submission)


def get_key(row):
    return row[0] + '_' + row[1]


def main(model_name, polygons_to_fix):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    csv.field_size_limit(sys.maxsize)

    input_submission_filename = os.path.join(SUBMISSION_DIR, 'submission_{}.csv'.format(model_name))
    submission = load_submission(input_submission_filename)
    logging.info('Submission loaded: %s', os.path.basename(input_submission_filename))

    keys_to_fix = set(get_key(r) for r in polygons_to_fix)
    for row in submission:
        row_key = get_key(row)

        if row_key in keys_to_fix:
            poly = row[2]
            poly_fixed = fix_polygon(poly)

            row[2] = poly_fixed

            logging.info('Polygon fixed: %s', row_key)

    output_submission_filename = os.path.join(SUBMISSION_DIR, 'submission_{}_fixed.csv'.format(model_name))
    save_submission(output_submission_filename, submission)
    logging.info('Submission saved: %s', os.path.basename(output_submission_filename))


if __name__ == '__main__':
    model_name = 'softmax_pansharpen_vgg_for_ws'

    polygons_to_fix = [
        ('6060_0_1', '4',),
        ('6030_1_4', '4',),
    ]

    main(model_name, polygons_to_fix)
