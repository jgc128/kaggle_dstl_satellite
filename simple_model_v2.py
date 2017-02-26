import os
import logging

from sacred import Experiment

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow_helpers.models.base_model import BaseModel

from config import CLASSES_NAMES, IMAGES_NORMALIZED_SHARPENED_FILENAME, IMAGES_MASKS_FILENAME, IMAGES_METADATA_FILENAME, \
    IMAGES_MEANS_STDS_FILENAME, TENSORBOARD_DIR, MODELS_DIR, IMAGES_NORMALIZED_DATA_DIR, IMAGES_PREDICTION_MASK_DIR
from utils.data import load_pickle, convert_masks_to_softmax, convert_softmax_to_masks, get_train_test_images_ids
from utils.matplotlib import matplotlib_setup
from utils.polygon import stack_masks, sample_patches, jaccard_coef, split_image_to_patches, join_mask_patches


class SimpleModel(BaseModel):
    def __init__(self, **kwargs):
        super(SimpleModel, self).__init__()
        logging.info('Creating model: %s', type(self).__name__)

        self.nb_classes = kwargs.get('nb_classes')

        self.regularization = kwargs.get('regularization', 0.0005)

    def build_model(self):
        batch_norm_params = {'is_training': self.is_train, 'decay': 0.999, 'updates_collections': None}

        with tf.name_scope('input'):
            input = self.input_dict['X']

            targets = self.input_dict['Y_softmax']
            targets_one_hot = tf.one_hot(targets, self.nb_classes + 1)

        # VGG-16
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(self.regularization)
                            ):
            net = slim.conv2d(input, 64, [3, 3], scope='conv1_1')
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

            # add a few conv layers as the output
            net = slim.conv2d(net, 32, [3, 3], scope='conv8_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv8_2')

        ########
        # Logits
        with slim.arg_scope([slim.conv2d, ],
                            normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(self.regularization)
                            ):
            logits = slim.conv2d(net, self.nb_classes + 1, [1, 1], scope='conv_final_classes', activation_fn=None)

        with tf.name_scope('prediction'):
            classes_probs = tf.nn.softmax(logits)

            self.op_predict = classes_probs

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets_one_hot, logits=logits)
            loss_ce = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=[1, 2]))

            tf.losses.add_loss(loss_ce)
            self.op_loss = tf.losses.get_total_loss(add_regularization_losses=True)


ex_name = 'simple_model_v2'
ex = Experiment(ex_name)


@ex.config
def config():
    nb_epoch = 20
    batch_size = 64

    model_name = 'softmax_without_small_pansharpen'

    classes_to_skip = [9, 10]
    patch_size = [224, 224]
    val_size = 256

    nb_iterations = 100000
    nb_samples_train = 1000
    nb_samples_val = 512
    batch_size = 30

    regularization = 0.0005

    model_load_step = -1
    debug = False


@ex.named_config
def big_objects():
    model_name = 'softmax_without_small_pansharpen_big_objects'
    classes_to_skip = [2, 5, 9, 10]
    patch_size = [224, 224]


@ex.named_config
def small_objects():
    model_name = 'softmax_without_small_pansharpen_small_objects'
    classes_to_skip = [1, 3, 4, 6, 7, 8]
    patch_size = [64, 64]


@ex.named_config
def debug_run():
    debug = True

    nb_iterations = 100000
    nb_samples_train = 10
    nb_samples_val = 5
    batch_size = 5


@ex.named_config
def prediction():
    model_load_step = 20700
    model_name = 'softmax_without_small_pansharpen_big_objects_small_patch'
    patch_size = [64, 64]
    batch_size = 350


@ex.capture
def evaluate_model_jaccard(model, images, images_masks_stacked, images_data, needed_classes, kind, batch_size):
    data_dict = sample_data_dict(images, images_masks_stacked, images_data, kind)
    nb_samples = len(data_dict['X'])

    Y_pred_probs = model.predict(data_dict, batch_size=batch_size)
    Y_pred = np.stack([convert_softmax_to_masks(Y_pred_probs[i]) for i in range(nb_samples)], axis=0)

    Y = data_dict['Y']
    Y = Y[:, :, :, needed_classes]
    jaccard = jaccard_coef(Y_pred, Y, mean=False)

    return jaccard


@ex.capture
def sample_data_dict(images, images_masks_stacked, images_data, kind,
                     patch_size, nb_samples_train, nb_samples_val, val_size, classes_to_skip):
    if kind == 'val':
        nb_samples = nb_samples_val
    elif kind == 'train':
        nb_samples = nb_samples_train
    else:
        raise AttributeError('Kind {} is not supported'.format(kind))

    patches = sample_patches(images, [images_masks_stacked, images_data], [patch_size, patch_size],
                             nb_samples, kind=kind, val_size=val_size)

    Y, X = patches[0], patches[1]
    Y_softmax = convert_masks_to_softmax(Y, classes_to_skip=classes_to_skip)

    data_dict = {'X': X, 'Y': Y, 'Y_softmax': Y_softmax}

    return data_dict


@ex.main
def main(model_name, classes_to_skip, patch_size, nb_iterations, batch_size, debug, regularization, model_load_step):
    # set-up matplotlib
    matplotlib_setup()

    classes_names = [c.strip().lower().replace(' ', '_').replace('.', '') for c in CLASSES_NAMES]
    nb_classes = len(classes_names)
    classes = np.arange(1, nb_classes + 1)
    logging.info('Classes: %s', nb_classes)

    images_all, images_train, images_test = get_train_test_images_ids()
    logging.info('Train: %s, test: %s, all: %s', len(images_train), len(images_test), len(images_all))

    # load images data
    images_data = load_pickle(IMAGES_NORMALIZED_SHARPENED_FILENAME)
    logging.info('Images: %s', len(images_data))

    # load masks
    images_masks = load_pickle(IMAGES_MASKS_FILENAME)
    logging.info('Masks: %s', len(images_masks))

    # load images metadata
    images_metadata = load_pickle(IMAGES_METADATA_FILENAME)
    logging.info('Metadata: %s', len(images_metadata))

    mean_sharpened, std_sharpened = load_pickle(IMAGES_MEANS_STDS_FILENAME)
    logging.info('Mean: %s, Std: %s', mean_sharpened.shape, std_sharpened.shape)

    images = sorted(list(images_data.keys()))
    nb_images = len(images)
    logging.info('Train images: %s', nb_images)

    images_masks_stacked = stack_masks(images, images_masks, classes)
    logging.info('Masks stacked: %s', len(images_masks_stacked))

    nb_channels = images_data[images[0]].shape[2]
    logging.info('Channels: %s', nb_channels)

    # skip vehicles and misc manmade structures
    needed_classes = [c for c in range(nb_classes) if c + 1 not in classes_to_skip]
    needed_classes_names = [c for i, c in enumerate(classes_names) if i + 1 not in classes_to_skip]
    nb_needed_classes = len(needed_classes)
    logging.info('Skipping classes: %s, needed classes: %s', classes_to_skip, nb_needed_classes)

    # create model

    sess_config = tf.ConfigProto(inter_op_parallelism_threads=4, intra_op_parallelism_threads=4)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model_params = {
        'nb_classes': nb_needed_classes,
        'regularization': regularization,
    }
    model = SimpleModel(**model_params)
    model.set_session(sess)
    if not debug:
        model.set_tensorboard_dir(os.path.join(TENSORBOARD_DIR, model_name))

    # TODO: not a fixed size
    model.add_input('X', [patch_size[0], patch_size[1], nb_channels])
    model.add_input('Y_softmax', [patch_size[0], patch_size[1], ], dtype=tf.uint8)

    model.build_model()

    # train model
    if model_load_step == -1:

        iteration_number = 0
        jaccard_val_mean = 0
        jaccard_train_mean = 0
        for iteration_number in range(1, nb_iterations + 1):
            try:
                data_dict_train = sample_data_dict(images, images_masks_stacked, images_data, 'train')
                model.train_model(data_dict_train, nb_epoch=1, batch_size=batch_size)

                # validate the model
                if iteration_number % 5 == 0:
                    jaccard_val = evaluate_model_jaccard(model, images, images_masks_stacked, images_data,
                                                         needed_classes, kind='val')
                    jaccard_train = evaluate_model_jaccard(model, images, images_masks_stacked, images_data,
                                                           needed_classes, kind='train')
                    jaccard_val_mean = np.mean(jaccard_val)
                    jaccard_train_mean = np.mean(jaccard_train)

                    logging.info('Iteration %s, jaccard val: %.5f, jaccard train: %.5f',
                                 iteration_number, jaccard_val_mean, jaccard_train_mean)

                    for i, cls in enumerate(needed_classes_names):
                        model.write_scalar_summary('jaccard_val/{}'.format(cls), jaccard_val[i])
                        model.write_scalar_summary('jaccard_train/{}'.format(cls), jaccard_train[i])

                    model.write_scalar_summary('jaccard_mean/val', jaccard_val_mean)
                    model.write_scalar_summary('jaccard_mean/train', jaccard_train_mean)

                # save the model
                if iteration_number % 15 == 0:
                    model_filename = os.path.join(MODELS_DIR, model_name)
                    saved_filename = model.save_model(model_filename)
                    logging.info('Model saved: %s', saved_filename)

            except KeyboardInterrupt:
                break

        result = {
            'iteration': iteration_number,
            'jaccard_val': jaccard_val_mean,
            'jaccard_train': jaccard_train_mean,
        }
        return result

    # predict
    else:
        model_to_restore = '{}-{}'.format(model_name, model_load_step)
        model_filename = os.path.join(MODELS_DIR, model_to_restore)
        model.restore_model(model_filename)
        logging.info('Model restored: %s', os.path.basename(model_filename))

        target_images = images_test
        for img_number, img_id in enumerate(target_images):
            img_filename = os.path.join(IMAGES_NORMALIZED_DATA_DIR, img_id + '.pkl')
            img_normalized = load_pickle(img_filename)

            patches, patches_coord = split_image_to_patches([img_normalized, ], [patch_size, ], overlap=0.5)

            X = patches[0]

            data_dict = {'X': X}
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
                         img_number + 1, len(target_images), 100 * (img_number + 1) / len(target_images))

        result = {

        }
        return result


if __name__ == '__main__':
    ex.run_commandline()
