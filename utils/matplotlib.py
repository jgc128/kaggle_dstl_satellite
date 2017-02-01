import os
import logging

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
