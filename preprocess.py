import logging

import numpy as np

import shapely
import shapely.wkt
import shapely.affinity

from utils.data import load_grid_sizes, load_polygons, load_images, save_pickle, get_images_sizes, pansharpen
from utils.matplotlib import matplotlib_setup
from utils.polygon import create_mask_from_polygons
from config import GRID_SIZES_FILENAME, POLYGONS_FILENAME, IMAGES_THREE_BAND_DIR, IMAGES_SIXTEEN_BAND_DIR, \
    IMAGES_NORMALIZED_SHARPENED_FILENAME, IMAGES_NORMALIZED_M_FILENAME, IMAGES_MEANS_STDS_FILENAME, \
    IMAGES_MASKS_FILENAME, IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME


def get_x_scaler(width, x_max):
    width_prime = width * (width / (width + 1))
    x_scaler = width_prime / x_max
    return x_scaler


def get_y_scaler(height, y_min):
    height_prime = height * (height / (height + 1))
    y_scaler = height_prime / y_min
    return y_scaler


def create_images_metadata(grid_sizes, polygons, images_sizes_rgb, images_sizes_m, images_sizes_p):
    images_metadata = {}
    images_metadata_polygons = {}

    for img_id, row in grid_sizes.iterrows():
        x_max = row['x_max']
        y_min = row['y_min']

        if img_id not in images_sizes_rgb or img_id not in images_sizes_m or img_id not in images_sizes_p:
            logging.warning('Skipping image: %s', img_id)
            continue

        width_rgb = images_sizes_rgb[img_id][1]
        height_rgb = images_sizes_rgb[img_id][0]
        x_rgb_scaler = get_x_scaler(width_rgb, x_max)
        y_rgb_scaler = get_y_scaler(height_rgb, y_min)

        width_m = images_sizes_m[img_id][1]
        height_m = images_sizes_m[img_id][0]
        x_m_scaler = get_x_scaler(width_m, x_max)
        y_m_scaler = get_y_scaler(height_m, y_min)

        width_p = images_sizes_p[img_id][1]
        height_p = images_sizes_p[img_id][0]
        x_p_scaler = get_x_scaler(width_p, x_max)
        y_p_scaler = get_y_scaler(height_p, y_min)

        image_md = {
            'image_id': img_id,
            'x_max': x_max,
            'y_min': y_min,

            'width_rgb': width_rgb,
            'height_rgb': height_rgb,
            'x_rgb_scaler': x_rgb_scaler,
            'y_rgb_scaler': y_rgb_scaler,

            'width_m': width_m,
            'height_m': height_m,
            'x_m_scaler': x_m_scaler,
            'y_m_scaler': y_m_scaler,

            'width_p': width_p,
            'height_p': height_p,
            'x_p_scaler': x_p_scaler,
            'y_p_scaler': y_p_scaler,
        }
        images_metadata[img_id] = image_md

        if img_id in polygons.index:
            img_polygons = polygons.loc[img_id]
            images_metadata_polygons[img_id] = {}

            for _, row in img_polygons.iterrows():
                polygon = row['polygons']
                poly = shapely.wkt.loads(polygon)
                ploy_rgb_scaled = shapely.affinity.scale(poly, xfact=x_rgb_scaler, yfact=y_rgb_scaler, origin=(0, 0, 0))
                ploy_m_scaled = shapely.affinity.scale(poly, xfact=x_m_scaler, yfact=y_m_scaler, origin=(0, 0, 0))
                ploy_p_scaled = shapely.affinity.scale(poly, xfact=x_p_scaler, yfact=y_p_scaler, origin=(0, 0, 0))

                class_type = row['class_type']

                image_md_poly = {
                    'poly': poly.wkt,
                    'ploy_rgb_scaled': ploy_rgb_scaled.wkt,
                    'ploy_m_scaled': ploy_m_scaled.wkt,
                    'ploy_p_scaled': ploy_p_scaled.wkt,
                    'class_type': class_type,
                }
                images_metadata_polygons[img_id][class_type] = image_md_poly

    return images_metadata, images_metadata_polygons


def create_classes_masks(images_metadata, images_metadata_polygons):
    masks = {}

    for img_id, img_polygons in images_metadata_polygons.items():
        masks[img_id] = {}

        for class_type, polygon_metadata in img_polygons.items():
            img_metadata = images_metadata[img_id]

            img_size = (img_metadata['height_rgb'], img_metadata['width_rgb'])
            polygons = shapely.wkt.loads(polygon_metadata['ploy_rgb_scaled'])
            mask = create_mask_from_polygons(img_size, polygons)

            masks[img_id][class_type] = mask

    return masks


def normalize_images(images_data):
    nb_channels = images_data[list(images_data.keys())[0]].shape[2]

    channel_data = [[] for _ in range(nb_channels)]
    for img_id, img_data in images_data.items():
        for i in range(nb_channels):
            img_channel_data = img_data[:, :, i].flatten()
            channel_data[i].append(img_channel_data)

    channel_data = np.array([np.concatenate(chds, axis=0) for chds in channel_data])

    channels_mean = channel_data.mean(axis=1).astype(np.float32)
    channels_std = channel_data.std(axis=1).astype(np.float32)

    images_data_normalized = {}
    for img_id, img_data in images_data.items():
        images_data_normalized[img_id] = ((img_data - channels_mean) / channels_std).astype(np.float32)

    return images_data_normalized, channels_mean, channels_std


def pansharpen_images(images_data_m, images_data_p, method='browley', W=0.3):
    images = sorted(images_data_m.keys())

    images_data_sharpened = {}
    for img_id in images:
        img_m = images_data_m[img_id]
        img_pan = images_data_p[img_id]
        img_rgbn_sharpened = pansharpen(img_m, img_pan, method=method, W=W)

        images_data_sharpened[img_id] = img_rgbn_sharpened

    return images_data_sharpened


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    grid_sizes = load_grid_sizes(GRID_SIZES_FILENAME)
    polygons = load_polygons(POLYGONS_FILENAME)

    all_images = sorted(set(grid_sizes.index))
    train_images = sorted(set(polygons.index))
    test_images = sorted(set(all_images) - set(train_images))
    logging.info('Train: %s, Test: %s, All: %s', len(train_images), len(test_images), len(all_images))

    # create images metadata
    images_sizes_rgb = get_images_sizes(IMAGES_THREE_BAND_DIR, target_images=all_images)
    images_sizes_m = get_images_sizes(IMAGES_SIXTEEN_BAND_DIR, target_images=all_images, target_format='M')
    images_sizes_p = get_images_sizes(IMAGES_SIXTEEN_BAND_DIR, target_images=all_images, target_format='P')
    images_metadata, images_metadata_polygons = create_images_metadata(
        grid_sizes, polygons, images_sizes_rgb, images_sizes_m, images_sizes_p)
    logging.info('Metadata: %s, polygons metadata: %s', len(images_metadata), len(images_metadata_polygons))

    # load train images
    images_data_m = load_images(IMAGES_SIXTEEN_BAND_DIR, target_images=train_images, target_format='M')
    images_data_p = load_images(IMAGES_SIXTEEN_BAND_DIR, target_images=train_images, target_format='P')

    # pansharpen to get (R,G,B,NIR) scaled images
    images_data_sharpened = pansharpen_images(images_data_m, images_data_p)
    logging.info('Images sharpened: %s', len(images_data_sharpened))

    # create masks using RGB sizes
    images_masks = create_classes_masks(images_metadata, images_metadata_polygons)

    # normalize all the data
    images_data_m_normalized, mean_m, std_m = normalize_images(images_data_m)
    images_data_sharpened_normalized, mean_sharpened, std_sharpened = normalize_images(images_data_sharpened)

    # save everything
    save_pickle(IMAGES_METADATA_FILENAME, images_metadata)
    save_pickle(IMAGES_METADATA_POLYGONS_FILENAME, images_metadata_polygons)

    save_pickle(IMAGES_MASKS_FILENAME, images_masks)
    save_pickle(IMAGES_NORMALIZED_M_FILENAME, images_data_m_normalized)
    save_pickle(IMAGES_NORMALIZED_SHARPENED_FILENAME, images_data_sharpened_normalized)

    save_pickle(IMAGES_MEANS_STDS_FILENAME, [mean_m, std_m, mean_sharpened, std_sharpened])


if __name__ == '__main__':
    main()
