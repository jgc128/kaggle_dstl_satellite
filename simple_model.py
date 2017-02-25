import logging
from collections import Counter
import sys
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from random import shuffle

from tensorflow_helpers.models.base_model import BaseModel

from utils.data import load_pickle, convert_masks_to_softmax, convert_softmax_to_masks
from utils.matplotlib import matplotlib_setup
from config import IMAGES_MASKS_FILENAME, TENSORBOARD_DIR, MODELS_DIR, \
    IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME, \
    IMAGES_NORMALIZED_DATA_DIR, IMAGES_PREDICTION_MASK_DIR, CLASSES_NAMES, IMAGES_NORMALIZED_M_FILENAME, \
    IMAGES_NORMALIZED_SHARPENED_FILENAME, IMAGES_MEANS_STDS_FILENAME
from utils.polygon import split_image_to_patches, join_mask_patches, sample_patches, jaccard_coef
import utils.tf as tf_utils


# https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation
# https://github.com/shekkizh/FCN.tensorflow
# https://github.com/warmspringwinds/tf-image-segmentation


# def jaccard_coef_loss(y_true, y_pred):
#     return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)
# https://www.kaggle.com/resolut/dstl-satellite-imagery-feature-detection/panchromatic-sharpening-simple
# NDVI, NIR, NDWI
# (NIR-RED)/(NIR+RED)
# dice loss, more weights
# shapely.wkt.dumps(polygons, rounding_precision=12)
# (RGB[0] - M[7] ) / (RGB[0] + M[7] + epsilon)
# Polyak Averaging
# RGB + M[0] + M[5:8]
# tree, water - excluding


class SimpleModel(BaseModel):
    def __init__(self, **kwargs):
        super(SimpleModel, self).__init__()
        logging.info('Using model: %s', type(self).__name__)

        self.nb_classes = kwargs.get('nb_classes')

    def build_model(self):
        input = self.input_dict['X']
        input_shape = tf.shape(input)

        targets = self.input_dict['Y']

        net = input

        # VGG-16
        batch_norm_params = {'is_training': self.is_train, 'decay': 0.999, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.005)
                            ):
            net = slim.conv2d(net, 64, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool1_1')

            net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2_1')

            net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3_1')
            pool3 = net  # save pool3 reference for FCN-8s

            net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool4_1')
            pool4 = net  # save pool4 reference for FCN-16s

            net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool5_1')

            net = slim.conv2d(net, 512, [1, 1], scope='conv6')
            net = slim.conv2d(net, 512, [1, 1], scope='conv7')

            # upsampling

            # first, upsample x2 and add scored pool4
            net = slim.conv2d_transpose(net, 512, [4, 4], stride=2, scope='upsample_1')
            pool4_scored = slim.conv2d(pool4, 512, [1, 1], scope='pool4_scored', activation_fn=None)
            net = net + pool4_scored

            # second, upsample x2 and add scored pool3
            net = slim.conv2d_transpose(net, 256, [4, 4], stride=2, scope='upsample_2')
            pool3_scored = slim.conv2d(pool3, 256, [1, 1], scope='pool3_scored', activation_fn=None)
            net = net + pool3_scored

            # finally, upsample x8
            net = slim.conv2d_transpose(net, 32, [16, 16], stride=8, scope='upsample_3')

            # # add a few conv layers as the output
            # net = slim.conv2d(net, 64, [3, 3], scope='conv_final_1')
            # net = slim.conv2d(net, 32, [3, 3], scope='conv_final_2')

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=slim.l2_regularizer(0.005)
                            ):
            net = slim.conv2d(net, self.nb_classes, [1, 1], scope='conv_final_classes', activation_fn=None)

        ########
        # Logits

        with tf.name_scope('prediction'):
            classes_probs = tf.nn.sigmoid(net)

            self.op_predict = classes_probs

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(net, targets)
            cross_entropy_classes = tf.reduce_sum(cross_entropy, axis=[1, 2])
            cross_entropy_batch = tf.reduce_sum(cross_entropy_classes, axis=-1)
            loss_ce = tf.reduce_mean(cross_entropy_batch)

            loss_jaccard = tf_utils.jaccard_coef(targets, classes_probs)

            loss_weighted = loss_ce - 1000 * tf.log(loss_jaccard)

            slim.losses.add_loss(loss_weighted)
            self.op_loss = slim.losses.get_total_loss(add_regularization_losses=True)


class CombinedModel(BaseModel):
    def __init__(self, **kwargs):
        super(CombinedModel, self).__init__()
        logging.info('Using model: %s', type(self).__name__)

        self.nb_classes = kwargs.get('nb_classes')

    def build_model(self):
        input_sharpened = self.input_dict['X_sharpened']
        input_sharpened_shape = tf.shape(input_sharpened)

        input_m = self.input_dict['X_m']
        input_m_shape = tf.shape(input_m)

        targets = self.input_dict['Y']
        targets_one_hot = tf.one_hot(targets, self.nb_classes + 1)

        # VGG-16
        batch_norm_params = {'is_training': self.is_train, 'decay': 0.999, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.005)
                            ):
            net = slim.conv2d(input_sharpened, 64, [3, 3], scope='conv1_1')
            net = slim.conv2d(net, 64, [3, 3], scope='conv1_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool1_1')
            pool1 = net

            net = slim.conv2d(net, 128, [3, 3], scope='conv2_1')
            net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2_1')
            pool2 = net

            # combine pool2 and M bands
            net_m = slim.conv2d(input_m, 64, [3, 3], scope='conv1_1_m')
            net_m = slim.conv2d(net_m, 128, [3, 3], scope='conv1_2_m')
            net = tf.concat_v2([net, net_m], axis=3)

            net = slim.conv2d(net, 256, [3, 3], scope='conv3_1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3_1')
            pool3 = net  # save pool3 reference for FCN-8s

            net = slim.conv2d(net, 512, [3, 3], scope='conv4_1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_2')
            net = slim.conv2d(net, 512, [3, 3], scope='conv4_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool4_1')
            pool4 = net  # save pool4 reference for FCN-16s

            net = slim.conv2d(net, 512, [3, 3], scope='conv5_1')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_2')
            net = slim.conv2d(net, 512, [3, 3], scope='conv5_3')
            net = slim.max_pool2d(net, [2, 2], scope='pool5_1')
            pool5 = net

            net = slim.conv2d(net, 512, [1, 1], scope='conv6')
            net = slim.conv2d(net, 512, [1, 1], scope='conv7')

            # upsampling

            # first, upsample x2 and add scored pool4
            net = slim.conv2d_transpose(net, 512, [4, 4], stride=2, scope='upsample_1')
            pool4_scored = slim.conv2d(pool4, 512, [1, 1], scope='pool4_scored', activation_fn=None)
            net = net + pool4_scored

            # second, upsample x2 and add scored pool3
            net = slim.conv2d_transpose(net, 256, [4, 4], stride=2, scope='upsample_2')
            pool3_scored = slim.conv2d(pool3, 256, [1, 1], scope='pool3_scored', activation_fn=None)
            net = net + pool3_scored

            # finally, upsample x8
            net = slim.conv2d_transpose(net, 32, [16, 16], stride=8, scope='upsample_3')

            # # add a few conv layers as the output
            # net = slim.conv2d(net, 64, [3, 3], scope='conv_final_1')
            # net = slim.conv2d(net, 32, [3, 3], scope='conv_final_2')

        ########
        # Logits
        with slim.arg_scope([slim.conv2d, ], weights_regularizer=slim.l2_regularizer(0.005)):
            net = slim.conv2d(net, self.nb_classes + 1, [1, 1], scope='conv_final_classes', activation_fn=None)

        with tf.name_scope('prediction'):
            classes_probs = tf.nn.softmax(net)

            self.op_predict = classes_probs

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net, targets_one_hot)
            loss_ce = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=[1, 2]))

            slim.losses.add_loss(loss_ce)
            self.op_loss = slim.losses.get_total_loss(add_regularization_losses=True)


def main(model_name):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    classes_names = [c.strip().lower().replace(' ', '_').replace('.', '') for c in CLASSES_NAMES]
    nb_classes = len(classes_names)
    classes = np.arange(1, nb_classes + 1)
    logging.info('Classes: %s', nb_classes)

    # load images data
    images_data_m = load_pickle(IMAGES_NORMALIZED_M_FILENAME)
    images_data_sharpened = load_pickle(IMAGES_NORMALIZED_SHARPENED_FILENAME)
    logging.info('Images: %s, %s', len(images_data_m), len(images_data_sharpened))

    # load masks
    images_masks = load_pickle(IMAGES_MASKS_FILENAME)
    logging.info('Masks: %s', len(images_masks))

    # load images metadata
    images_metadata = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Metadata: %s', len(images_metadata))

    mean_m, std_m, mean_sharpened, std_sharpened = load_pickle(IMAGES_MEANS_STDS_FILENAME)
    logging.info('Mean & Std: %s - %s, %s - %s', mean_m.shape, std_m.shape, mean_sharpened.shape, std_sharpened.shape)

    images = sorted(list(images_data_sharpened.keys()))
    nb_images = len(images)
    logging.info('Train images: %s', nb_images)

    images_masks_stacked = {
        img_id: np.stack([images_masks[img_id][target_class] for target_class in classes], axis=-1)
        for img_id in images
        }

    nb_channels_m = images_data_m[images[0]].shape[2]
    nb_channels_sharpened = images_data_sharpened[images[0]].shape[2]
    logging.info('Channels: %s, %s', nb_channels_m, nb_channels_sharpened)

    # skip vehicles and misc manmade structures
    classes_to_skip = {1, 3, 4, 5, 6, 7, 8}  # {2, 9, 10}
    needed_classes = [c for c in range(nb_classes) if c + 1 not in classes_to_skip]
    needed_classes_names = [c for i, c in enumerate(classes_names) if i + 1 not in classes_to_skip]
    logging.info('Skipping classes: %s', classes_to_skip)

    # skip M bands that were pansharpened
    m_bands_to_skip = {4, 2, 1, 6}
    needed_m_bands = [i for i in range(nb_channels_m) if i not in m_bands_to_skip]
    logging.info('Skipping M bands: %s', m_bands_to_skip)

    patch_size = (64, 64,)  # (224, 224,)
    patch_size_sharpened = (patch_size[0], patch_size[1],)
    patch_size_m = (patch_size_sharpened[0] // 4, patch_size_sharpened[1] // 4,)
    logging.info('Patch sizes: %s, %s, %s', patch_size, patch_size_sharpened, patch_size_m)

    val_size = 256
    logging.info('Validation size: %s', val_size)

    sess_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model_params = {
        'nb_classes': nb_classes - len(classes_to_skip),
    }
    model = CombinedModel(**model_params)
    model.set_session(sess)
    model.set_tensorboard_dir(os.path.join(TENSORBOARD_DIR, model_name))

    # TODO: not a fixed size
    model.add_input('X_sharpened', [patch_size_sharpened[0], patch_size_sharpened[1], nb_channels_sharpened])
    model.add_input('X_m', [patch_size_m[0], patch_size_m[1], nb_channels_m - len(m_bands_to_skip)])
    model.add_input('Y', [patch_size[0], patch_size[1], ], dtype=tf.uint8)

    model.build_model()

    # train model

    nb_iterations = 100000
    nb_samples_train = 1000  # 10 1000
    nb_samples_val = 512  # 10 512
    batch_size = 30  # 5 30

    for iteration_number in range(1, nb_iterations + 1):
        try:
            patches = sample_patches(images,
                                     [images_masks_stacked, images_data_sharpened, images_data_m],
                                     [patch_size, patch_size_sharpened, patch_size_m],
                                     nb_samples_train, kind='train', val_size=val_size)

            Y, X_sharpened, X_m = patches[0], patches[1], patches[2]
            Y_softmax = convert_masks_to_softmax(Y, classes_to_skip=classes_to_skip)
            X_m = X_m[:, :, :, needed_m_bands]

            data_dict_train = {'X_sharpened': X_sharpened, 'X_m': X_m, 'Y': Y_softmax}
            model.train_model(data_dict_train, nb_epoch=1, batch_size=batch_size)

            # validate the model
            if iteration_number % 5 == 0:

                # calc jaccard val
                patches_val = sample_patches(images,
                                             [images_masks_stacked, images_data_sharpened, images_data_m],
                                             [patch_size, patch_size_sharpened, patch_size_m],
                                             nb_samples_val, kind='val', val_size=val_size)

                Y_val, X_sharpened_val, X_m_val = patches_val[0], patches_val[1], patches_val[2]
                X_m_val = X_m_val[:, :, :, needed_m_bands]

                data_dict_val = {'X_sharpened': X_sharpened_val, 'X_m': X_m_val, }
                Y_val_pred_probs = model.predict(data_dict_val, batch_size=batch_size)
                Y_val_pred = np.stack([convert_softmax_to_masks(Y_val_pred_probs[i])
                                       for i in range(nb_samples_val)], axis=0)

                Y_val = Y_val[:, :, :, needed_classes]
                jaccard_val = jaccard_coef(Y_val_pred, Y_val, mean=False)

                # calc jaccard train
                patches_train_val = sample_patches(images,
                                                   [images_masks_stacked, images_data_sharpened, images_data_m],
                                                   [patch_size, patch_size_sharpened, patch_size_m],
                                                   nb_samples_val, kind='train', val_size=val_size)

                Y_train_val, X_sharpened_train_val, X_m_train_val = \
                    patches_train_val[0], patches_train_val[1], patches_train_val[2]
                X_m_train_val = X_m_train_val[:, :, :, needed_m_bands]

                data_dict_val = {'X_sharpened': X_sharpened_train_val, 'X_m': X_m_train_val, }
                Y_train_val_pred_probs = model.predict(data_dict_val, batch_size=batch_size)
                Y_train_val_pred = np.stack([convert_softmax_to_masks(Y_train_val_pred_probs[i])
                                             for i in range(nb_samples_val)], axis=0)

                Y_train_val = Y_train_val[:, :, :, needed_classes]
                jaccard_train_val = jaccard_coef(Y_train_val_pred, Y_train_val, mean=False)

                logging.info('Iteration %s, jaccard val: %.5f, jaccard train: %.5f',
                             iteration_number, np.mean(jaccard_val), np.mean(jaccard_train_val))

                for i, cls in enumerate(needed_classes_names):
                    model.write_scalar_summary('jaccard_val/{}'.format(cls), jaccard_val[i])
                    model.write_scalar_summary('jaccard_train/{}'.format(cls), jaccard_train_val[i])

                model.write_scalar_summary('jaccard_mean/val', np.mean(jaccard_val))
                model.write_scalar_summary('jaccard_mean/train', np.mean(jaccard_train_val))

            # save the model
            if iteration_number % 15 == 0:
                model_filename = os.path.join(MODELS_DIR, model_name)
                saved_filename = model.save_model(model_filename)
                logging.info('Model saved: %s', saved_filename)

        except KeyboardInterrupt:
            break


def predict(kind, model_name, global_step):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    logging.info('Prediction mode')

    nb_channels_m = 8
    nb_channels_sharpened = 4
    nb_classes = 10

    # skip vehicles and misc manmade structures
    classes_to_skip = {2, 9, 10}
    logging.info('Skipping classes: %s', classes_to_skip)

    # skip M bands that were pansharpened
    m_bands_to_skip = {4, 2, 1, 6}
    needed_m_bands = [i for i in range(nb_channels_m) if i not in m_bands_to_skip]
    logging.info('Skipping M bands: %s', m_bands_to_skip)

    patch_size = (64, 64,)  # (224, 224,)
    patch_size_sharpened = (patch_size[0], patch_size[1],)
    patch_size_m = (patch_size_sharpened[0] // 4, patch_size_sharpened[1] // 4,)
    logging.info('Patch sizes: %s, %s, %s', patch_size, patch_size_sharpened, patch_size_m)

    images_metadata = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Metadata: %s', len(images_metadata))

    images_metadata_polygons = load_pickle(IMAGES_METADATA_POLYGONS_FILENAME)
    logging.info('Polygons metadata: %s', len(images_metadata_polygons))

    images_all = list(images_metadata.keys())
    images_train = list(images_metadata_polygons.keys())
    images_test = sorted(set(images_all) - set(images_train))

    if kind == 'test':
        target_images = images_test
    elif kind == 'train':
        target_images = images_train
    else:
        raise ValueError('Unknown kind: {}'.format(kind))

    nb_target_images = len(target_images)
    logging.info('Target images: %s - %s', kind, nb_target_images)

    batch_size = 25

    # create and load model
    sess_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model_params = {
        'nb_classes': nb_classes - len(classes_to_skip),
    }
    model = CombinedModel(**model_params)
    model.set_session(sess)
    # model.set_tensorboard_dir(os.path.join(TENSORBOARD_DIR, 'simple_model'))

    # TODO: not a fixed size
    model.add_input('X_sharpened', [patch_size_sharpened[0], patch_size_sharpened[1], nb_channels_sharpened])
    model.add_input('X_m', [patch_size_m[0], patch_size_m[1], nb_channels_m - len(m_bands_to_skip)])
    model.add_input('Y', [patch_size[0], patch_size[1], ], dtype=tf.uint8)

    model.build_model()

    model_to_restore = '{}-{}'.format(model_name, global_step)
    model_filename = os.path.join(MODELS_DIR, model_to_restore)
    model.restore_model(model_filename)
    logging.info('Model restored: %s', os.path.basename(model_filename))

    for img_number, img_id in enumerate(target_images):
        img_filename = os.path.join(IMAGES_NORMALIZED_DATA_DIR, img_id + '.pkl')
        img_normalized_sharpened, img_normalized_m = load_pickle(img_filename)

        patches, patches_coord = split_image_to_patches(
            [img_normalized_sharpened, img_normalized_m],
            [patch_size_sharpened, patch_size_m],
            overlap=0.5)

        X_sharpened = np.array(patches[0])
        X_m = np.array(patches[1])
        X_m = X_m[:, :, :, needed_m_bands]

        data_dict = {'X_sharpened': X_sharpened, 'X_m': X_m, }
        classes_probs_patches = model.predict(data_dict, batch_size=batch_size)

        classes_probs = join_mask_patches(
            classes_probs_patches, patches_coord[0],
            images_metadata[img_id]['height_rgb'], images_metadata[img_id]['width_rgb'],
            softmax=True, normalization=False)

        masks_without_excluded = convert_softmax_to_masks(classes_probs)

        # join masks and put zeros insted of excluded classes
        zeros_filler = np.zeros_like(masks_without_excluded[:, :, 0])
        masks_all = []
        j = 0
        for i in range(nb_classes):
            if i + 1 not in classes_to_skip:
                masks_all.append(masks_without_excluded[:, :, j])
                j += 1
            else:
                masks_all.append(zeros_filler)

        masks = np.stack(masks_all, axis=-1)

        mask_filename = os.path.join(IMAGES_PREDICTION_MASK_DIR, '{0}_{1}.npy'.format(img_id, model_name))
        np.save(mask_filename, masks)

        logging.info('Predicted: %s/%s [%.2f]',
                     img_number + 1, nb_target_images, 100 * (img_number + 1) / nb_target_images)


if __name__ == '__main__':
    task = 'main'
    model_name = 'combined_model_jaccard_softmax_only_small'  # 'resnet' 'simple_model_jaccard_sigmoid'
    global_step = 41310
    kind = 'test'

    if len(sys.argv) > 1:
        task = sys.argv[1]

    if task == 'predict':
        if len(sys.argv) > 2:
            kind = sys.argv[2]

        if len(sys.argv) > 3:
            global_step = sys.argv[3]

        predict(kind, model_name, global_step)

    if task == 'main':
        main(model_name)
