import os
import logging
import numpy as np

def matplotlib_setup():
    import matplotlib

    display = os.environ.get('DISPLAY', '')
    if display == '':
        logging.info('Display is not set')
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')

    logging.info('Matplotlib backend: %s', matplotlib.get_backend())


def plot_image(image_data):
    import matplotlib.pyplot as plt
    import tifffile as tiff

    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(image_data)
    # fig.tight_layout()
    # plt.show()

    tiff.imshow(image_data)
    plt.show()

def plot_mask(mask_data):
    """Adopted from https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly"""
    import matplotlib.pyplot as plt
    import tifffile as tiff

    mask_plot_data = 255 * np.stack([mask_data, mask_data, mask_data])

    tiff.imshow(mask_plot_data)
    plt.show()
