import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow_helpers.models.base_model import BaseModel

from utils.data import load_grid_sizes, load_polygons, load_images, load_pickle
from utils.matplotlib import matplotlib_setup, plot_mask, plot_image, plot_test_predictions
from config import IMAGES_NORMALIZED_FILENAME, IMAGES_MASKS_FILENAME, FIGURES_DIR, TENSORBOARD_DIR, MODELS_DIR, \
    IMAGES_METADATA_FILENAME, TRAIN_PATCHES_FILENAME


# https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation
# https://github.com/shekkizh/FCN.tensorflow
# https://github.com/warmspringwinds/tf-image-segmentation


class SimpleModel(BaseModel):
    def __init__(self, **kwargs):
        super(SimpleModel, self).__init__()

    def build_model(self):
        input = self.input_dict['X']
        input_shape = tf.shape(input)

        targets = self.input_dict['Y']
        targets_shape = tf.shape(targets)
        nb_classes = int(targets.get_shape()[3])

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
            net = slim.conv2d_transpose(net, nb_classes, [16, 16], stride=8, scope='upsample_3', activation_fn=None)

        ########
        # Logits

        with tf.name_scope('prediction'):
            classes_probs = tf.nn.sigmoid(net)

            self.op_predict = classes_probs

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(net, targets)
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

    # load sampled patches
    with np.load(TRAIN_PATCHES_FILENAME) as data:
        X = data['train_patches']
        Y = data['train_mask']

    logging.info('X: %s, Y: %s', X.shape, Y.shape)

    # load images metadata
    images_metadata, channels_mean, channels_std = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Images metadata: %s, mean: %s, std: %s',
                 len(images_metadata), channels_mean.shape, channels_std.shape)

    sess_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model_params = {

    }
    model = SimpleModel(**model_params)
    model.set_session(sess)
    model.set_tensorboard_dir(os.path.join(TENSORBOARD_DIR, 'simple_model'))

    patch_size = (X.shape[1],X.shape[2],)
    nb_channels = X.shape[3]
    nb_classes = Y.shape[3]

    # TODO: not a fixed size
    model.add_input('X', [patch_size[0], patch_size[0], nb_channels])
    model.add_input('Y', [patch_size[0], patch_size[0], nb_classes])

    model.build_model()

    # train model

    nb_epoch = 10
    batch_size = 40
    nb_test_images = 3
    data_dict_train = {'X': X, 'Y': Y}
    for i in range(nb_epoch):
        try:
            model.train_model(data_dict_train, nb_epoch=1, batch_size=batch_size)

            if i % 2 == 0:
                X_test = X[:nb_test_images]
                Y_test = Y[:nb_test_images]
                data_dict_test = {'X': X_test}
                Y_pred_probs = model.predict(data_dict_test)
                Y_pred_probs = np.array(Y_pred_probs)
                Y_pred = np.round(Y_pred_probs)

                classes_to_plot = [1, 6] # Building and crops

                for target_class in classes_to_plot:
                    prefix = 'epoch_{}_class_{}/'.format(i, target_class)
                    model.write_image_summary(prefix + 'image', X_test * channels_std + channels_mean)

                    Y_true_class = np.expand_dims(Y_test[:,:,:,target_class-1], 3)
                    Y_pred_class = np.expand_dims(Y_pred[:,:,:,target_class-1], 3)
                    model.write_image_summary(prefix + 'true', Y_true_class)
                    model.write_image_summary(prefix + 'pred', Y_pred_class)

                model_name = os.path.join(MODELS_DIR, 'simple_model')
                saved_filaname = model.save_model(model_name, global_step=i)
                logging.info('Model saved: %s', saved_filaname)

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
