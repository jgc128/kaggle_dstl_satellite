from collections import defaultdict

import cv2
import shapely
import shapely.wkt
import shapely.affinity
from shapely.geometry import MultiPolygon, Polygon
import rasterio
import rasterio.features
import numpy as np


# https://mapbox.github.io/rasterio/api/rasterio.features.html
# rasterio.features.rasterize()

# def helper(poly):
#     polygons = shapely.wkt.loads(poly).buffer(0.00001)
#     return shapely.wkt.dumps(polygons)



def round_coords(coords):
    return np.array(coords).round().astype(np.int32)


def create_mask_from_metadata(img_metadata, polygon_metadata):
    """Adopted from https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly"""
    image_size = (img_metadata['height'], img_metadata['width'])
    multipolygon = shapely.wkt.loads(polygon_metadata['ploy_scaled'])

    image_mask = np.zeros(image_size, np.uint8)

    if not multipolygon:
        return image_mask

    exteriors = [round_coords(poly.exterior.coords) for poly in multipolygon]
    interiors = [round_coords(pi.coords) for poly in multipolygon for pi in poly.interiors]

    cv2.fillPoly(image_mask, exteriors, 1)
    cv2.fillPoly(image_mask, interiors, 0)

    return image_mask




def mask_to_polygons(mask, epsilon=5, min_area=1.0):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()

    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])

    if all_polygons.is_valid:
        return all_polygons
    else:
        return MultiPolygon()

def mask_to_polygons_v2(mask):
    all_polygons=[]
    for shape, value in rasterio.features.shapes(mask.astype(np.int16), mask = (mask==1), transform = rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):

        all_polygons.append(shapely.geometry.shape(shape))

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        #Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        #need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])

    return all_polygons

def create_polygons_from_mask(mask, image_metadata, scale=True):
    # poly = mask_to_polygons(mask, min_area=1.0)
    poly = mask_to_polygons_v2(mask)

    if scale:
        x_scaler = image_metadata['x_scaler']
        y_scaler = image_metadata['y_scaler']
        poly = shapely.affinity.scale(poly, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))

    return poly


def sample_random_corner(img_height, img_width, patch_size, kind, val_size):
    if kind == 'train':
        min_height = 0
        min_width = val_size
        max_height = img_height - patch_size[0]
        max_width = img_width - patch_size[1]
    elif kind == 'val':
        min_height = 0
        min_width = 0
        max_height = img_height - patch_size[0]
        max_width = val_size
    elif kind == 'all':
        min_height = 0
        min_width = 0
        max_height = img_height - patch_size[0]
        max_width = img_width - patch_size[1]
    else:
        raise ValueError('Kind {} is not valid'.format(kind))

    img_c1 = (
        np.random.randint(min_height, max_height),
        np.random.randint(min_width, max_width)
    )
    img_c2 = (img_c1[0] + patch_size[0], img_c1[1] + patch_size[1])

    return img_c1, img_c2


def sample_patch(img_data, img_mask_data, patch_size, kind='train', val_size = 256):
    img_height = img_mask_data.shape[0]
    img_width = img_mask_data.shape[1]

    img_c1, img_c2 = sample_random_corner(img_height, img_width, patch_size, kind, val_size)

    img_patch = img_data[img_c1[0]:img_c2[0], img_c1[1]:img_c2[1], :]
    img_mask = img_mask_data[img_c1[0]:img_c2[0], img_c1[1]:img_c2[1], :]

    return img_patch, img_mask



def sample_patches(images, images_data, images_masks_stacked, patch_size, nb_samples, kind='train', val_size = 256):
    nb_channels = images_data[images[0]].shape[2]
    nb_classes = images_masks_stacked[images[0]].shape[2]

    X = np.zeros((nb_samples, patch_size[0], patch_size[1], nb_channels))
    Y = np.zeros((nb_samples, patch_size[0], patch_size[1], nb_classes), dtype=np.uint8)

    for i in range(nb_samples):
        img_id = np.random.choice(images)
        img_data = images_data[img_id]
        img_mask_data = images_masks_stacked[img_id]

        img_patch, img_mask = sample_patch(img_data, img_mask_data, patch_size, kind=kind, val_size=val_size)

        X[i] = img_patch
        Y[i] = img_mask

    return X, Y


def split_image_to_patches(image_data, patch_size, leftovers=True, add_random=0):
    patches = []
    patches_coord = []

    nb_patches_height = int(image_data.shape[0] / patch_size[0])
    nb_patches_width = int(image_data.shape[1] / patch_size[1])

    # TODO: deal with leftovers
    leftover_height = image_data.shape[0] - nb_patches_height * patch_size[0]
    leftover_width = image_data.shape[1] - nb_patches_width * patch_size[1]

    # tile
    for i in range(nb_patches_height):
        for j in range(nb_patches_width):
            c1 = (i * patch_size[0], j * patch_size[1])
            patch = image_data[c1[0]:c1[0] + patch_size[0], c1[1]:c1[1] + patch_size[1], :]

            patches.append(patch)
            patches_coord.append(c1)


    # deal with leftovers
    if leftovers:
        for i in range(nb_patches_height):
            c1 = (i * patch_size[0], image_data.shape[1] - patch_size[1])
            patch = image_data[c1[0]:c1[0] + patch_size[0], c1[1]:c1[1] + patch_size[1], :]

            patches.append(patch)
            patches_coord.append(c1)

        for j in range(nb_patches_width):
            c1 = (image_data.shape[0] - patch_size[0], j * patch_size[0])
            patch = image_data[c1[0]:c1[0] + patch_size[0], c1[1]:c1[1] + patch_size[1], :]

            patches.append(patch)
            patches_coord.append(c1)

    # add random patches
    for i in range(add_random):
        c1, _ = sample_random_corner(image_data.shape[0], image_data.shape[1], patch_size, kind='all', val_size=0)
        patch = image_data[c1[0]:c1[0] + patch_size[0], c1[1]:c1[1] + patch_size[1], :]

        patches.append(patch)
        patches_coord.append(c1)

    return patches, patches_coord


def join_mask_patches(patches, patches_coord, image_height, image_width, softmax_needed=False):
    def softmax(x, axis=-1):
        original_shape = x.shape
        nb_classes = original_shape[axis]

        x_reshaped = np.reshape(x, (-1, nb_classes))
        x_sotfmaxed = np.exp(x_reshaped) / np.sum(np.exp(x_reshaped), axis=-1, keepdims=True)

        x_res = np.reshape(x_sotfmaxed, original_shape)

        return x_res

    nb_channels = patches[0].shape[2]
    patch_size = patches[0].shape[:2]

    image_data = np.zeros((image_height, image_width, nb_channels), dtype=patches.dtype)

    for i, c1 in enumerate(patches_coord):
        image_data[c1[0]:c1[0] + patch_size[0], c1[1]:c1[1] + patch_size[1], :] += patches[i]

    if softmax_needed:
        image_data = softmax(image_data)

    return image_data


def jaccard_coef(y_pred, y_true, mean=True):
    # inspired by https://www.kaggle.com/drn01z3/dstl-satellite-imagery-feature-detection/end-to-end-baseline-with-u-net-keras

    epsilon = 0.00001

    intersection = np.sum(y_pred * y_true, axis=(0,1,2))
    sum_tmp = np.sum(y_pred + y_true, axis=(0,1,2))
    union = sum_tmp - intersection

    jaccard = (intersection + epsilon) / (union + epsilon)
    jaccard_mean = np.mean(jaccard)

    return jaccard_mean