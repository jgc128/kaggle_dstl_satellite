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

# def rotateImage(image, angle):
#     image_center = tuple(np.array(image.shape[:2])/2)
#     rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
#     result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_LINEAR)
#     return result


def round_coords(coords):
    return np.array(coords).round().astype(np.int32)


def create_mask_from_polygons(img_size, polygons):
    """Adopted from https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly"""

    image_mask = np.zeros(img_size, np.uint8)

    if polygons is None:
        return image_mask

    exteriors = [round_coords(poly.exterior.coords) for poly in polygons]
    interiors = [round_coords(pi.coords) for poly in polygons for pi in poly.interiors]

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


def mask_to_polygons_v2(mask, min_area=1.0):
    all_polygons = []
    discarded = 0
    for shape, value in rasterio.features.shapes(mask.astype(np.int16), mask=(mask == 1),
                                                 transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        poly = shapely.geometry.shape(shape)
        if poly.area > min_area:
            all_polygons.append(poly)
        else:
            discarded += 1

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)

    all_polygons = all_polygons.simplify(tolerance=0.5, preserve_topology=False)

    if all_polygons.type == 'Polygon':
        all_polygons = shapely.geometry.MultiPolygon([all_polygons])

    return all_polygons


def create_polygons_from_mask(mask, image_metadata, scale=True):
    # poly = mask_to_polygons(mask, min_area=1.0)
    poly = mask_to_polygons_v2(mask, min_area=20)

    if scale:
        x_scaler = image_metadata['x_rgb_scaler']
        y_scaler = image_metadata['y_rgb_scaler']
        poly = shapely.affinity.scale(poly, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))

    return poly


def sample_random_corner(img_height, img_width, patch_size, kind, val_size):
    if kind == 'train':
        min_height = 0
        min_width = val_size
        max_height = img_height - patch_size[0] - 4  # just in case
        max_width = img_width - patch_size[1] - 4  # just in case
    elif kind == 'val':
        min_height = 0
        min_width = 0
        max_height = img_height - patch_size[0]
        max_width = val_size - patch_size[1]
    elif kind == 'all':
        min_height = 0
        min_width = 0
        max_height = img_height - patch_size[0]
        max_width = img_width - patch_size[1]
    elif kind == 'test':
        min_height = 0
        min_width = 0
        max_height = 10
        max_width = 10
    else:
        raise ValueError('Kind {} is not valid'.format(kind))

    img_c1 = (
        np.random.randint(min_height, max_height),
        np.random.randint(min_width, max_width)
    )

    return img_c1


def get_scaled_patches(base_c1, patch_sizes):
    scales = [(patch_sizes[0][0] / p[0], patch_sizes[0][1] / p[1],) for p in patch_sizes]

    img_c1 = [(int(base_c1[0] / scales[i][0]), int(base_c1[1] / scales[i][1])) for i, p in enumerate(patch_sizes)]
    img_c2 = [(img_c1[i][0] + patch_sizes[i][0], img_c1[i][1] + patch_sizes[i][1]) for i, p in enumerate(patch_sizes)]

    return img_c1, img_c2


def sample_patch(img_data, patch_sizes, kind='train', val_size=256):
    img_height = img_data[0].shape[0]
    img_width = img_data[0].shape[1]
    patch_size = patch_sizes[0]

    base_c1 = sample_random_corner(img_height, img_width, patch_size, kind, val_size)
    img_c1, img_c2 = get_scaled_patches(base_c1, patch_sizes)

    img_patch = [d[img_c1[i][0]:img_c2[i][0], img_c1[i][1]:img_c2[i][1], :] for i, d in enumerate(img_data)]

    return img_patch


def sample_patches(images, data, patch_sizes, nb_samples, kind='train', val_size=256):
    nb_data = len(data)
    nb_data_channels = [d[images[0]].shape[2] for d in data]
    data_sampled = [
        np.zeros((nb_samples, patch_sizes[i][0], patch_sizes[i][1], nb_data_channels[i]), dtype=d[images[0]].dtype)
        for i, d in enumerate(data)
        ]

    for i in range(nb_samples):
        img_id = np.random.choice(images)
        img_data = [d[img_id] for d in data]

        img_patch = sample_patch(img_data, patch_sizes, kind=kind, val_size=val_size)

        for k in range(nb_data):
            data_sampled[k][i] = img_patch[k]

    return data_sampled


def split_image_to_patches(data, patch_sizes, overlap=0.5):
    img_height = data[0].shape[0] - 4  # just in case
    img_width = data[0].shape[1] - 4  # just in case
    patch_size = patch_sizes[0]

    step_size = (int(patch_size[0] * overlap), int(patch_size[1] * overlap))
    nb_steps_height = (img_height - patch_size[0]) // step_size[0] + 1
    nb_steps_width = (img_width - patch_size[1]) // step_size[1] + 1

    patches = [[] for _ in range(len(data))]
    patches_coordinates = [[] for _ in range(len(data))]

    # tile
    for i in range(nb_steps_height):
        for j in range(nb_steps_width):
            base_c1 = (i * step_size[0], j * step_size[1])
            img_c1, img_c2 = get_scaled_patches(base_c1, patch_sizes)

            for k, img_data in enumerate(data):
                img_patch = img_data[img_c1[k][0]:img_c2[k][0], img_c1[k][1]:img_c2[k][1], :]
                patches[k].append(img_patch)
                patches_coordinates[k].append(img_c1[k])

    # leftovers - width
    for i in range(nb_steps_height):
        base_c1 = (i * step_size[0], img_width - patch_size[1])
        img_c1, img_c2 = get_scaled_patches(base_c1, patch_sizes)

        for k, img_data in enumerate(data):
            img_patch = img_data[img_c1[k][0]:img_c2[k][0], img_c1[k][1]:img_c2[k][1], :]
            patches[k].append(img_patch)
            patches_coordinates[k].append(img_c1[k])

    # leftovers - height
    for j in range(nb_steps_width):
        base_c1 = (img_height - patch_size[0], j * step_size[1])
        img_c1, img_c2 = get_scaled_patches(base_c1, patch_sizes)

        for k, img_data in enumerate(data):
            img_patch = img_data[img_c1[k][0]:img_c2[k][0], img_c1[k][1]:img_c2[k][1], :]
            patches[k].append(img_patch)
            patches_coordinates[k].append(img_c1[k])

    # leftovers - bottom right corner
    base_c1 = (img_height - patch_size[0], img_width - patch_size[1])
    img_c1, img_c2 = get_scaled_patches(base_c1, patch_sizes)
    for k, img_data in enumerate(data):
        img_patch = img_data[img_c1[k][0]:img_c2[k][0], img_c1[k][1]:img_c2[k][1], :]
        patches[k].append(img_patch)
        patches_coordinates[k].append(img_c1[k])

    return patches, patches_coordinates


def join_mask_patches(patches, patches_coord, image_height, image_width, softmax=False, normalization=False):
    def np_softmax(x, axis=-1):
        original_shape = x.shape
        nb_classes = original_shape[axis]

        x_reshaped = np.reshape(x, (-1, nb_classes))
        x_sotfmaxed = np.exp(x_reshaped) / np.sum(np.exp(x_reshaped), axis=-1, keepdims=True)

        x_res = np.reshape(x_sotfmaxed, original_shape)

        return x_res

    nb_channels = patches[0].shape[2]
    patch_size = patches[0].shape[:2]

    image_data = np.zeros((image_height, image_width, nb_channels), dtype=patches.dtype)
    counts = np.zeros((image_height, image_width, nb_channels), dtype=np.float32)
    counts += 0.000001

    for i, c1 in enumerate(patches_coord):
        image_data[c1[0]:c1[0] + patch_size[0], c1[1]:c1[1] + patch_size[1], :] += patches[i]
        counts[c1[0]:c1[0] + patch_size[0], c1[1]:c1[1] + patch_size[1], :] += np.ones_like(patches[i],
                                                                                            dtype=np.float32)

    if softmax:
        image_data = np_softmax(image_data)

    if normalization:
        image_data = image_data / counts

    return image_data


def jaccard_coef(y_pred, y_true, mean=True):
    # inspired by https://www.kaggle.com/drn01z3/dstl-satellite-imagery-feature-detection/end-to-end-baseline-with-u-net-keras

    epsilon = 0.00001

    intersection = np.sum(y_pred * y_true, axis=(0, 1, 2))
    sum_tmp = np.sum(y_pred + y_true, axis=(0, 1, 2))
    union = sum_tmp - intersection

    jaccard = (intersection + epsilon) / (union + epsilon)

    if mean:
        jaccard = np.mean(jaccard)

    return jaccard


def simplify_mask(mask, kernel_size=5):
    nb_classes = mask.shape[2]

    mask_simplified = []
    for i in range(nb_classes):
        msk = mask[:, :, i]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        msk_handled = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel)

        mask_simplified.append(msk_handled)

    mask_simplified = np.stack(mask_simplified, axis=-1)

    return mask_simplified

def stack_masks(images, images_masks, classes):
    images_masks_stacked = {
        img_id: np.stack([images_masks[img_id][target_class] for target_class in classes], axis=-1)
        for img_id in images
        }

    return images_masks_stacked


