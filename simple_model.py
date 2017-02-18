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

from utils.data import load_pickle, convert_mask_to_one_hot, convert_softmax_to_masks
from utils.matplotlib import matplotlib_setup
from config import IMAGES_NORMALIZED_FILENAME, IMAGES_MASKS_FILENAME, TENSORBOARD_DIR, MODELS_DIR, \
    IMAGES_METADATA_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME, \
    IMAGES_NORMALIZED_DATA_DIR, IMAGES_PREDICTION_MASK_DIR
from utils.polygon import split_image_to_patches, join_mask_patches, sample_patches
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


class ResNetModel(BaseModel):
    def __init__(self, **kwargs):
        super(ResNetModel, self).__init__()
        logging.info('Using model: %s', type(self).__name__)

        self.nb_classes = kwargs.get('nb_classes')

        self.op_jaccard = None

    def build_model(self):
        input = self.input_dict['X']
        input_shape = tf.shape(input)

        targets = self.input_dict['Y']

        net = input

        with slim.arg_scope(nets.resnet_v2.resnet_arg_scope(self.is_train)):
            net, end_points = nets.resnet_v2.resnet_v2_50(net, num_classes=None, global_pool=False, output_stride=16)

        batch_norm_params = {'is_training': self.is_train, 'decay': 0.999, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05)):
            # first, upsample x2 and add scored pool4
            net = slim.conv2d_transpose(net, 256, [4, 4], stride=2, scope='upsample_1', padding='SAME')
            block1 = end_points['resnet_v2_50/block2/unit_3/bottleneck_v2']
            block1_scored = slim.conv2d(block1, 256, [1, 1], scope='block1_scored', activation_fn=None)
            net = net + block1_scored

            # second, upsample x2 and add scored pool3
            net = slim.conv2d_transpose(net, 128, [4, 4], stride=2, scope='upsample_2')
            block2 = end_points['resnet_v2_50/block1/unit_2/bottleneck_v2']
            block2_scored = slim.conv2d(block2, 128, [1, 1], scope='pool3_scored', activation_fn=None)
            net = net + block2_scored

            # finally, upsample x8
            net = slim.conv2d_transpose(net, 64, [8, 8], stride=4, scope='upsample_3')


        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            weights_regularizer=slim.l2_regularizer(0.005)
                            ):
            # net = slim.conv2d(net, 64, [3, 3], scope='conv_final')
            net = slim.conv2d(net, self.nb_classes, [1, 1], scope='conv_final_classes', activation_fn=None)

        ########
        # Logits

        with tf.name_scope('prediction'):
            classes_probs = tf.nn.sigmoid(net)

            self.op_predict = classes_probs
            self.op_jaccard = tf_utils.jaccard_coef(targets, classes_probs, mean=False)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(net, targets)
            cross_entropy_classes = tf.reduce_sum(cross_entropy, axis=[1, 2])
            cross_entropy_batch = tf.reduce_sum(cross_entropy_classes, axis=-1)
            loss_ce = tf.reduce_mean(cross_entropy_batch)

            loss_jaccard = tf_utils.jaccard_coef(targets, classes_probs)

            loss_weighted = loss_ce - 1000 * tf.log(loss_jaccard)

            slim.losses.add_loss(loss_weighted)
            self.op_loss = slim.losses.get_total_loss(add_regularization_losses=True)



def main(model_name):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    classes_names = ['Buildings', 'Misc. Manmade structures', 'Road', 'Track', 'Trees', 'Crops', 'Waterway',
                     'Standing water', 'Vehicle Large', 'Vehicle Small', ]

    classes_names = [c.strip().lower().replace(' ', '_').replace('.', '') for c in classes_names]

    # load images
    images_data = load_pickle(IMAGES_NORMALIZED_FILENAME)
    images_masks = load_pickle(IMAGES_MASKS_FILENAME)
    logging.info('Images: %s, masks: %s', len(images_data), len(images_masks))

    # load images metadata
    images_metadata, channels_mean, channels_std = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Images metadata: %s, mean: %s, std: %s',
                 len(images_metadata), channels_mean.shape, channels_std.shape)

    images = sorted(list(images_data.keys()))
    nb_images = len(images)
    logging.info('Train images: %s', nb_images)

    nb_classes = len(images_masks[images[0]])
    classes = np.arange(1, nb_classes + 1)
    images_masks_stacked = {
        img_id: np.stack([images_masks[img_id][target_class] for target_class in classes], axis=-1)
        for img_id in images
        }
    nb_channels = images_data[images[0]].shape[2]
    logging.info('Classes: %s, channels: %s', nb_classes, nb_channels)

    patch_size = (224, 224,)
    val_size = 256
    logging.info('Patch size: %s, validation size: %s', patch_size, val_size)

    sess_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model_params = {
        'nb_classes': nb_classes,
    }
    model = ResNetModel(**model_params)
    model.set_session(sess)
    model.set_tensorboard_dir(os.path.join(TENSORBOARD_DIR, model_name))

    # TODO: not a fixed size
    model.add_input('X', [patch_size[0], patch_size[1], nb_channels])
    model.add_input('Y', [patch_size[0], patch_size[1], nb_classes])  # , dtype=tf.uint8

    model.build_model()

    # train model

    nb_iterations = 100000
    nb_samples_train = 10  # 10 1000
    nb_samples_val = 10  # 10 512
    batch_size = 30  # 5 30

    for iteration_number in range(1, nb_iterations + 1):
        try:
            X, Y = sample_patches(images, images_data, images_masks_stacked, patch_size, nb_samples_train,
                                  kind='train', val_size=val_size)

            data_dict_train = {'X': X, 'Y': Y}
            model.train_model(data_dict_train, nb_epoch=1, batch_size=batch_size)

            # validate the model
            if iteration_number % 5 == 0:
                X_val, Y_val = sample_patches(images, images_data, images_masks_stacked, patch_size, nb_samples_val,
                                              kind='val', val_size=val_size)

                data_dict_val = {'X': X_val, 'Y': Y_val}
                jaccard_val = model.predict(data_dict_val, target_op=model.op_jaccard, batch_size=batch_size)

                X_train_val, Y_train_val = sample_patches(images, images_data, images_masks_stacked, patch_size,
                                                          nb_samples_val, kind='train', val_size=val_size)

                data_dict_train_val = {'X': X_train_val, 'Y': Y_train_val}
                jaccard_train_val = model.predict(data_dict_train_val, target_op=model.op_jaccard, batch_size=batch_size)

                logging.info('Iteration %s, jaccard val: %.5f, jaccard train: %.5f',
                             iteration_number, np.mean(jaccard_val), np.mean(jaccard_train_val))

                for i, cls in zip(range(nb_classes), classes_names):
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

    # load images metadata
    images_metadata, channels_mean, channels_std = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Images metadata: %s, mean: %s, std: %s', len(images_metadata), channels_mean.shape,
                 channels_std.shape)

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

    patch_size = (224, 224,)
    nb_channels = 3
    nb_classes = 10

    batch_size = 30

    # create and load model
    sess_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model_params = {
        'nb_classes': nb_classes,
    }
    model = SimpleModel(**model_params)
    model.set_session(sess)
    # model.set_tensorboard_dir(os.path.join(TENSORBOARD_DIR, 'simple_model'))

    # TODO: not a fixed size
    model.add_input('X', [patch_size[0], patch_size[0], nb_channels])
    model.add_input('Y', [patch_size[0], patch_size[1], nb_classes])  # , dtype=tf.uint8

    model.build_model()

    model_to_restore = '{}-{}'.format(model_name, global_step)
    model_filename = os.path.join(MODELS_DIR, model_to_restore)
    model.restore_model(model_filename)
    logging.info('Model restored: %s', os.path.basename(model_filename))

    for img_number, img_id in enumerate(target_images):
        img_filename = os.path.join(IMAGES_NORMALIZED_DATA_DIR, img_id + '.npy')
        img_data = np.load(img_filename)

        patches, patches_coord = split_image_to_patches(img_data, patch_size, leftovers=True, add_random=200)

        X = np.array(patches)
        data_dict = {'X': X}
        classes_prob = model.predict(data_dict, batch_size=batch_size)
        masks = np.round(np.array(classes_prob))
        masks_joined = join_mask_patches(masks, patches_coord, img_data.shape[0], img_data.shape[1],
                                                softmax=False, normalization=True)

        masks_joined = np.round(masks_joined)

        mask_filename = os.path.join(IMAGES_PREDICTION_MASK_DIR, img_id + '.npy')
        np.save(mask_filename, masks_joined)

        logging.info('Predicted: %s/%s [%.2f]',
                     img_number + 1, nb_target_images, 100 * (img_number + 1) / nb_target_images)


if __name__ == '__main__':
    task = 'main'
    model_name = 'resnet_jaccard1000'  # 'resnet' 'simple_model_jaccard_sigmoid'

    if len(sys.argv) > 1:
        task = sys.argv[1]

    if task == 'predict':
        kind = 'test'
        if len(sys.argv) > 2:
            kind = sys.argv[2]

        global_step = 555
        if len(sys.argv) > 3:
            global_step = sys.argv[3]

        predict(kind, model_name, global_step)

    if task == 'main':
        main(model_name)
