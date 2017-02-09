import logging
from collections import Counter
import sys
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from random import shuffle

from tensorflow_helpers.models.base_model import BaseModel
from tensorflow_helpers.utils.data import batch_generator

from utils.data import load_grid_sizes, load_polygons, load_images, load_pickle, convert_mask_to_one_hot
from utils.matplotlib import matplotlib_setup, plot_mask, plot_image, plot_test_predictions
from config import IMAGES_NORMALIZED_FILENAME, IMAGES_MASKS_FILENAME, FIGURES_DIR, TENSORBOARD_DIR, MODELS_DIR, \
    IMAGES_METADATA_FILENAME, TRAIN_PATCHES_COORDINATES_FILENAME, IMAGES_METADATA_POLYGONS_FILENAME, \
    IMAGES_TEST_NORMALIZED_DATA_DIR, IMAGES_TEST_PREDICTION_MASK_DIR

# https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation
# https://github.com/shekkizh/FCN.tensorflow
# https://github.com/warmspringwinds/tf-image-segmentation
from utils.polygon import split_image_to_patches, join_patches_to_image, sample_patches_batch


class SimpleModel(BaseModel):
    def __init__(self, **kwargs):
        super(SimpleModel, self).__init__()

        self.nb_classes = kwargs.get('nb_classes')

    def build_model(self):
        input = self.input_dict['X']
        input_shape = tf.shape(input)

        targets = self.input_dict['Y']
        targets_one_hot = tf.one_hot(targets, self.nb_classes + 1)

        net = input

        # VGG-16
        batch_norm_params = {'is_training': self.is_train, 'decay': 0.999, 'updates_collections': None}
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
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

            net = slim.conv2d(net, 4096, [1, 1], scope='conv6')
            net = slim.conv2d(net, 1000, [1, 1], scope='conv7')

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
            net = slim.conv2d_transpose(net, self.nb_classes + 1, [16, 16], stride=8, scope='upsample_3',
                                        activation_fn=None)

        ########
        # Logits

        with tf.name_scope('prediction'):
            classes_probs = tf.nn.softmax(net)

            self.op_predict = classes_probs

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net, targets_one_hot)
            loss = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=[1, 2]))

            self.op_loss = loss

    def write_image_summary(self, name, image, nb_test_images=3):
        if self.log_writer is not None:
            image_data = tf.constant(image, dtype=tf.float32)

            summary_op = tf.summary.image(name, image_data, max_outputs=nb_test_images)
            summary_res = self.sess.run(summary_op)
            self.log_writer.add_summary(summary_res, self._global_step)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

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

    patch_size = (256, 256,)
    logging.info('Patch size: %s', patch_size)

    sess_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model_params = {
        'nb_classes': nb_classes
    }
    model = SimpleModel(**model_params)
    model.set_session(sess)
    model.set_tensorboard_dir(os.path.join(TENSORBOARD_DIR, 'simple_model'))

    # TODO: not a fixed size
    model.add_input('X', [patch_size[0], patch_size[1], nb_channels])
    model.add_input('Y', [patch_size[0], patch_size[1], ], dtype=tf.uint8)

    model.build_model()

    # train model

    nb_iterations = 100000
    sample_size = 1000
    batch_size = 40
    nb_test_images = 3

    for iteration_number in range(1, nb_iterations + 1):
        try:
            X, Y = sample_patches_batch(images, images_data, images_masks_stacked, patch_size, sample_size)
            Y_one_hot = convert_mask_to_one_hot(Y)

            data_dict_train = {'X': X, 'Y': Y_one_hot}
            model.train_model(data_dict_train, nb_epoch=1, batch_size=batch_size)

            if iteration_number % 15 == 0:
                model_name = os.path.join(MODELS_DIR, 'simple_model')
                saved_filename = model.save_model(model_name, global_step=iteration_number)
                logging.info('Model saved: %s', saved_filename)

        except KeyboardInterrupt:
            break


def predict():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    logging.info('Prediction')

    # load images metadata
    images_metadata, channels_mean, channels_std = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Images metadata: %s, mean: %s, std: %s', len(images_metadata), channels_mean.shape,
                 channels_std.shape)

    images_metadata_polygons = load_pickle(IMAGES_METADATA_POLYGONS_FILENAME)
    logging.info('Polygons metadata: %s', len(images_metadata_polygons))

    images_all = list(images_metadata.keys())
    images_train = list(images_metadata_polygons.keys())
    images_test = sorted(set(images_all) - set(images_train))
    # images_test = images_test[:10]
    nb_test_images = len(images_test)
    logging.info('Test images: %s', nb_test_images)

    patch_size = (256, 256)
    nb_channels = 3
    nb_classes = 10

    # create and load model
    sess_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model_params = {

    }
    model = SimpleModel(**model_params)
    model.set_session(sess)
    model.set_tensorboard_dir(os.path.join(TENSORBOARD_DIR, 'simple_model'))

    # TODO: not a fixed size
    model.add_input('X', [patch_size[0], patch_size[0], nb_channels])
    model.add_input('Y', [patch_size[0], patch_size[0], nb_classes])

    model.build_model()

    model_filename = os.path.join(MODELS_DIR, 'simple_model-91')
    model.restore_model(model_filename)
    logging.info('Model restored: %s', os.path.basename(model_filename))

    for i, img_id in enumerate(images_test):
        img_filename = os.path.join(IMAGES_TEST_NORMALIZED_DATA_DIR, img_id + '.npy')
        img_data = np.load(img_filename)

        patches, patches_coord = split_image_to_patches(img_data, patch_size)

        X = np.array(patches)
        data_dict = {'X': X}
        classes_prob = model.predict(data_dict)
        mask_patches = np.round(np.array(classes_prob)).astype(np.uint8)

        mask_joined = join_patches_to_image(mask_patches, patches_coord, img_data.shape[0], img_data.shape[1])

        mask_filename = os.path.join(IMAGES_TEST_PREDICTION_MASK_DIR, img_id + '.npy')
        np.save(mask_filename, mask_joined)

        if (i + 1) % 10 == 0:
            logging.info('Predicted: %s/%s [%.2f]', i, nb_test_images, 100 * i / nb_test_images)


if __name__ == '__main__':
    task = 'main'

    if len(sys.argv) > 1:
        task = sys.argv[1]

    if task == 'predict':
        predict()

    if task == 'main':
        main()
