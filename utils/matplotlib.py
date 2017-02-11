import os
import logging

import shapely
import shapely.wkt
import shapely.affinity

import numpy as np
from scipy.misc import bytescale


def matplotlib_setup():
    import matplotlib

    display = os.environ.get('DISPLAY', '')
    if display == '':
        logging.info('Display is not set')
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')

    logging.info('Matplotlib backend: %s', matplotlib.get_backend())


def plot_image(image_data, figure=None, subplot=111):
    import matplotlib.pyplot as plt
    import tifffile as tiff

    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(image_data)
    # fig.tight_layout()
    # plt.show()

    tiff.imshow(image_data, figure=figure, subplot=subplot)
    plt.show()


def plot_mask(mask_data, figure=None, subplot=111):
    """Adopted from https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly"""
    import matplotlib.pyplot as plt
    import tifffile as tiff

    mask_plot_data = 255 * np.stack([mask_data, mask_data, mask_data])

    tiff.imshow(mask_plot_data, figure=figure, subplot=subplot)
    plt.show()


def plot_polygons(img_data, img_metadata, img_poly_pred, img_poly_true=None, title=None, show=True):
    import matplotlib.pyplot as plt

    img_data_scaled = bytescale(img_data, low=0, high=255, cmin=0, cmax=2047)

    if isinstance(img_poly_pred[list(img_poly_pred.keys())[0]], str):
        img_poly_pred = {i: shapely.wkt.loads(p) for i, p in img_poly_pred.items()}

    img_poly_true = {i: shapely.wkt.loads(p['ploy_scaled']) for i, p in img_poly_true.items()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax_true = axes[0]
    ax_pred = axes[1]

    # plot images
    ax_true.imshow(img_data_scaled, interpolation='none')
    ax_pred.imshow(img_data_scaled, interpolation='none')

    # plot polygons
    plt_patches_pred = create_matplotlib_patches_from_polygons(img_poly_pred)
    ax_pred.add_collection(plt_patches_pred)

    plt_patches_true = create_matplotlib_patches_from_polygons(img_poly_true)
    ax_true.add_collection(plt_patches_true)

    # set attributes
    ax_true.set_xlim(0, img_metadata['width'])
    ax_true.set_ylim(0, img_metadata['height'])
    ax_true.set_axis_off()

    ax_pred.set_xlim(0, img_metadata['width'])
    ax_pred.set_ylim(0, img_metadata['height'])
    ax_pred.set_axis_off()

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()

    if show:
        plt.show()


def create_matplotlib_patches_from_polygons(img_poly):
    from matplotlib.path import Path
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection
    import seaborn as sns

    int_coords = lambda x: np.array(x).round().astype(np.int32)

    class_colormap = sns.color_palette('deep', len(img_poly.keys()))

    plt_polygons = []
    plt_colors = []
    for class_type in sorted(img_poly.keys()):
        multipoly = img_poly[class_type]

        exteriors = [int_coords(poly.exterior.coords) for poly in multipoly]
        interiors = [int_coords(pi.coords) for poly in multipoly for pi in poly.interiors]
        for poly in exteriors:
            plt_poly = patches.Polygon(poly, closed=True)
            plt_polygons.append(plt_poly)
            plt_colors.append(class_colormap[class_type - 1])

            # for poly in interiors:
            #     plt_poly = patches.Polygon(poly, closed=True)
            #     plt_polygons.append(plt_poly)
            #     plt_colors.append('red')
            #
            # break

    plt_patches = PatchCollection(plt_polygons, facecolors=plt_colors)

    return plt_patches
