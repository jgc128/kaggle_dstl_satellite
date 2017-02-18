import os

# DEBUG = True
DEBUG = False
DEBUG_IMAGE = '6120_2_2'

DATA_DIR = './data/'

SAMPLE_SUBMISSION_FILENAME = os.path.join(DATA_DIR, 'sample_submission.csv')

GRID_SIZES_FILENAME = os.path.join(DATA_DIR, 'grid_sizes.csv')
POLYGONS_FILENAME = os.path.join(DATA_DIR, 'train_wkt_v4.csv')
IMAGES_THREE_BAND_DIR = os.path.join(DATA_DIR, 'three_band/')
IMAGES_SIXTEEN_BAND_DIR = os.path.join(DATA_DIR, 'sixteen_band/')

IMAGES_METADATA_FILENAME = os.path.join(DATA_DIR, 'images_metadata.pkl')
IMAGES_METADATA_POLYGONS_FILENAME = os.path.join(DATA_DIR, 'images_metadata_polygons.pkl')
IMAGES_NORMALIZED_M_FILENAME = os.path.join(DATA_DIR, 'images_normalized_m.pkl')
IMAGES_NORMALIZED_SHARPENED_FILENAME = os.path.join(DATA_DIR, 'images_normalized_sharpened.pkl')
IMAGES_MEANS_STDS_FILENAME = os.path.join(DATA_DIR, 'images_means_stds.pkl')
IMAGES_MASKS_FILENAME = os.path.join(DATA_DIR, 'images_masks.pkl')

IMAGES_NORMALIZED_DATA_DIR = os.path.join(DATA_DIR, 'images_test_normalized/')
IMAGES_PREDICTION_MASK_DIR = os.path.join(DATA_DIR, 'images_test_prediction_mask/')

MODELS_DIR = os.path.join(DATA_DIR, 'models/')
FIGURES_DIR = os.path.join(DATA_DIR, 'figures/')
TENSORBOARD_DIR = os.path.join(DATA_DIR, 'tensorboard/')

TRAIN_PATCHES_COORDINATES_FILENAME = os.path.join(DATA_DIR, 'train_pathes.npz')

SUBMISSION_DIR = os.path.join(DATA_DIR, 'submissions/')

CLASSES_NAMES = [
    'Buildings',
    'Misc. Manmade structures',
    'Road',
    'Track',
    'Trees',
    'Crops',
    'Waterway',
    'Standing water',
    'Vehicle Large',
    'Vehicle Small',
]
