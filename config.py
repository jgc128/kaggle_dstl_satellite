import os

# DEBUG = True
DEBUG = False
DEBUG_IMAGE = '6120_2_2'

DATA_DIR = './data/'

GRID_SIZES_FILENAME = os.path.join(DATA_DIR, 'grid_sizes.csv')
POLYGONS_FILENAME = os.path.join(DATA_DIR, 'train_wkt_v4.csv')
IMAGES_THREE_BAND_DIR = os.path.join(DATA_DIR, 'three_band/')

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed/')

IMAGES_METADATA_FILENAME = os.path.join(DATA_DIR, 'images_metadata.pkl')
IMAGES_METADATA_POLYGONS_FILENAME = os.path.join(DATA_DIR, 'images_metadata_polygons.pkl')
IMAGES_NORMALIZED_FILENAME = os.path.join(DATA_DIR, 'images_normalized.pkl')
IMAGES_MASKS_FILENAME = os.path.join(DATA_DIR, 'images_masks.pkl')

MODELS_DIR = os.path.join(DATA_DIR, 'models/')
FIGURES_DIR = os.path.join(DATA_DIR, 'figures/')
TENSORBOARD_DIR = os.path.join(DATA_DIR, 'tensorboard/')
