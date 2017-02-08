import cv2
import shapely
import shapely.wkt
import shapely.affinity
import numpy as np

# https://mapbox.github.io/rasterio/api/rasterio.features.html
# rasterio.features.rasterize()

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


def split_image_to_patches(image_data, patch_size):
    patches = []
    patches_coord = []

    nb_patches_height = int(image_data.shape[0] / patch_size[0])
    nb_patches_width = int(image_data.shape[1] / patch_size[1])

    # TODO: deal with leftovers
    leftover_height = image_data.shape[0] - nb_patches_height * patch_size[0]
    leftover_width = image_data.shape[1] - nb_patches_width * patch_size[1]

    for i in range(nb_patches_height):
        for j in range(nb_patches_width):
            c1 = (i * patch_size[0], j * patch_size[1])
            patch = image_data[c1[0]:c1[0] + patch_size[0], c1[1]:c1[1] + patch_size[1], :]

            patches.append(patch)
            patches_coord.append(c1)

    return patches, patches_coord


def join_patches_to_image(patches, patches_coord, image_height, image_width):
    nb_channels = patches[0].shape[2]
    patch_size = patches[0].shape[:2]

    image_data = np.zeros((image_height, image_width, nb_channels), dtype=patches.dtype)

    for i, c1 in enumerate(patches_coord):
        image_data[c1[0]:c1[0] + patch_size[0], c1[1]:c1[1] + patch_size[1], :] = patches[i]

    return image_data