import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


from tensorflow_helpers.models.base_model import BaseModel

from utils.data import load_grid_sizes, load_polygons, load_images, load_pickle
from utils.matplotlib import matplotlib_setup, plot_mask, plot_image, plot_test_predictions
from config import DATA_DIR, GRID_SIZES_FILENAME, POLYGONS_FILENAME, IMAGES_DIR, IMAGES_METADATA_MASKS_FILENAME, \
    DEBUG_IMAGE


# https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation
# https://github.com/shekkizh/FCN.tensorflow
# https://github.com/warmspringwinds/tf-image-segmentation


class SimpleModel(BaseModel):
    def __init__(self, **kwargs):
        super(SimpleModel, self).__init__()

    def build_model(self):
        input = self.input_dict['X']
        input_shape = tf.shape(input)

        net = input

        # VGG-16
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
        net = slim.conv2d(net, 4096, [1, 1], scope='conv7')

        # upsampling

        ################
        # FCN-32s
        # net = slim.convolution2d_transpose(net, 1, [64, 64], stride=32 ,scope='upsamle_1') # 2 * factor - factor % 2

        ################
        # FCN-16s
        #
        # # first, upsample x2 and add scored pool4
        # net = slim.convolution2d_transpose(net, 1, [4, 4], stride=2, scope='upsamle_1')
        # pool4_scored = slim.conv2d(pool4, 1, [1, 1], scope='pool4_scored', activation_fn=None)
        # net = net + pool4_scored
        #
        # # second, umsample x16
        # net = slim.convolution2d_transpose(net, 1, [32, 32], stride=16 ,scope='upsamle_2')


        ################
        # FCN-8s

        # first, upsample x2 and add scored pool4
        net = slim.convolution2d_transpose(net, 1, [4, 4], stride=2, scope='upsamle_1')
        pool4_scored = slim.conv2d(pool4, 1, [1, 1], scope='pool4_scored', activation_fn=None)
        net = net + pool4_scored

        # second, upsample x2 and add scored pool3
        net = slim.convolution2d_transpose(net, 1, [4, 4], stride=2, scope='upsamle_2')
        pool3_scored = slim.conv2d(pool3, 1, [1, 1], scope='pool3_scored', activation_fn=None)
        net = net + pool3_scored

        # finally, upsample x8
        net = slim.convolution2d_transpose(net, 1, [32, 32], stride=8, scope='upsamle_3')

        # input_shape = input.get_shape()
        # logits = tf.reshape(net, (-1, int(input_shape[1]), int(input_shape[2]),))
        logits = tf.reshape(net, (-1, input_shape[1], input_shape[2],))

        with tf.name_scope('prediction'):
            predict = tf.nn.sigmoid(logits)

            self.op_predict = predict

        with tf.name_scope('loss'):
            targets = self.input_dict['Y']

            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)

            loss = tf.reduce_mean(cross_entropy)

            self.op_loss = loss

def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : %(levelname)s : %(module)s : %(message)s", datefmt="%d-%m-%Y %H:%M:%S"
    )

    matplotlib_setup()

    import matplotlib.pyplot as plt

    images_data, images_metadata, images_masks, channels_mean = load_pickle(IMAGES_METADATA_MASKS_FILENAME)
    logging.info('Images: %s, metadata: %s, masks: %s', len(images_data), len(images_metadata), len(images_masks))

    sess_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess = tf.Session(config=sess_config)

    model_params = {

    }
    model = SimpleModel(**model_params)
    model.set_session(sess)

    # model.add_input('X', [None, None, 3])
    # model.add_input('Y', [None, None])

    patch_size = (256, 256,)
    nb_channels = 3
    target_class = 1

    model.add_input('X', [patch_size[0], patch_size[0], nb_channels])
    model.add_input('Y', [patch_size[0], patch_size[0], ])

    model.build_model()

    # train model

    nb_epoch = 1
    batch_size = 32
    images = np.array([img_id for img_id in images_data.keys() if images_masks[img_id][target_class].sum() > 0])
    X = None
    Y = None
    for i in range(nb_epoch):
        # sample random patches from images
        train_patches = []
        train_mask = []

        while len(train_patches) <= batch_size:
            img_id = np.random.choice(images)

            img_data = images_data[img_id]
            img_mask_data = images_masks[img_id][target_class]

            img_height = img_data.shape[0]
            img_width = img_data.shape[1]

            img_random_c1 = (np.random.randint(0, img_height - patch_size[0]), np.random.randint(0, img_width - patch_size[1]))
            img_random_c2 = (img_random_c1[0] + patch_size[0], img_random_c1[1] + patch_size[1])

            img_patch = img_data[img_random_c1[0]:img_random_c2[0],img_random_c1[1]:img_random_c2[1],:]
            img_mask = img_mask_data[img_random_c1[0]:img_random_c2[0],img_random_c1[1]:img_random_c2[1]]

            # add only samples with some amount of target class
            mask_fraction = img_mask.sum() / (patch_size[0] * patch_size[1])
            if mask_fraction >= 0.1:
                train_patches.append(img_patch)
                train_mask.append(img_mask)

        X = np.array(train_patches)
        Y = np.array(train_mask)

        data_dict_train = {'X': X, 'Y': Y}
        model.train_model(data_dict_train, nb_epoch=1, batch_size=batch_size)


        # plot sample predictions
        if i % 10 == 0:
            nb_test_images = 3

            X_test = X[:nb_test_images]
            Y_true = Y[:nb_test_images]
            data_dict_test = {'X': X_test}
            Y_pred = model.predict(data_dict_test)
            Y_pred = np.array(Y_pred)
            Y_pred[Y_pred < 0.5] = 0
            Y_pred[Y_pred >= 0.5] = 1

            title = 'Epoch {}'.format(i)
            plot_test_predictions(X_test, Y_true, Y_pred, channels_mean, title=title)


if __name__ == '__main__':
    main()
