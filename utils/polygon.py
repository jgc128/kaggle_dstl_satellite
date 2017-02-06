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
