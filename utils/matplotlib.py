import os
import logging

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


def plot_test_predictions(X_test, Y_true, Y_pred, channels_mean, channels_std, show=True, title=None, filename=None):
    import matplotlib.pyplot as plt

    nb_test_images = len(X_test)

    fig, axes = plt.subplots(nb_test_images, 3, figsize=(10, 10))
    for j in range(nb_test_images):
        ax_img = axes[j, 0]
        ax_true = axes[j, 1]
        ax_pred = axes[j, 2]

        X_test_sacled = bytescale(X_test[j] * channels_std + channels_mean, low=0, high=2047, cmin=0, cmax=2047)

        ax_img.imshow(X_test_sacled, interpolation='none')
        ax_true.imshow(Y_true[j], interpolation='none', cmap='gray')
        ax_pred.imshow(Y_pred[j], interpolation='none', cmap='gray')

    for ax in axes.flatten():
        ax.set_axis_off()

    for ax, ax_title in zip(axes[0], ['Image', 'Ground Truth', 'Prediction', ]):
        ax.set_title(ax_title)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout(pad=2)

    if show:
        plt.show()

    if filename is not None:
        plt.savefig(filename)
        logging.info('Image saved: %s', os.path.basename(filename))
