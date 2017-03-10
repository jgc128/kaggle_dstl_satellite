import os
import logging

from sacred import Experiment

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
# inception = tf.contrib.slim.nets.inception
import tensorflow.contrib.slim.nets
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils
# import tensorflow.contrib.slim.nets.resnet_v2 as resnet_v2
from tensorflow_helpers.models.base_model import BaseModel

from config import CLASSES_NAMES, IMAGES_NORMALIZED_SHARPENED_FILENAME, IMAGES_MASKS_FILENAME, IMAGES_METADATA_FILENAME, \
    IMAGES_MEANS_STDS_FILENAME, TENSORBOARD_DIR, MODELS_DIR, IMAGES_NORMALIZED_DATA_DIR, IMAGES_PREDICTION_MASK_DIR
from utils.data import load_pickle, convert_masks_to_softmax, convert_softmax_to_masks, get_train_test_images_ids, \
    save_prediction_mask
from utils.matplotlib import matplotlib_setup
from utils.polygon import stack_masks, sample_patches, jaccard_coef, split_image_to_patches, join_mask_patches


class SimpleModel(BaseModel):
    def __init__(self, **kwargs):
        super(SimpleModel, self).__init__()
        logging.info('Creating model: %s', type(self).__name__)

        self.nb_classes = kwargs.get('nb_classes')

        self.regularization = kwargs.get('regularization', 0.005)

    def build_model(self):
        batch_norm_params = {'is_training': self.is_train, 'decay': 0.999, }

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

            total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                with tf.control_dependencies([updates]):
                    total_loss = tf.identity(total_loss)

            self.op_loss = total_loss


class ResNetModel(BaseModel):
    def __init__(self, **kwargs):
        super(ResNetModel, self).__init__()
        logging.info('Creating model: %s', type(self).__name__)

        self.nb_classes = kwargs.get('nb_classes')

        self.regularization = kwargs.get('regularization', 0.0005)

    def build_model(self):
        with tf.name_scope('input'):
            input = self.input_dict['X']

            targets = self.input_dict['Y_softmax']
            targets_one_hot = tf.one_hot(targets, self.nb_classes + 1)

        # with slim.arg_scope(inception.inception_v3_arg_scope()):
        #     net, endpoints = inception.inception_v3(input, is_training=self.is_train)
        #     zzz = 0
        with slim.arg_scope(resnet_utils.resnet_arg_scope(is_training=self.is_train)):
            net, end_points = resnet_v2.resnet_v2_101(input,
                                                      num_classes=None,
                                                      global_pool=False,
                                                      output_stride=16)
            zzz = 0

        with tf.name_scope('upsamplig'):
            # upsampling

            # first, upsample x2
            net = slim.conv2d_transpose(net, 256, [4, 4], stride=2, scope='upsample_1')
            block1_scored = slim.conv2d(end_points['resnet_v2_101/block1'], 256, [1, 1], scope='upsample_1_scored',
                                        activation_fn=None)
            net = net + block1_scored

            # second, upsample x2
            net = slim.conv2d_transpose(net, 128, [4, 4], stride=2, scope='upsample_2')
            block2_scored = slim.conv2d(end_points['resnet_v2_101/block1/unit_1/bottleneck_v2'], 128, [1, 1],
                                        scope='upsample_2_scored', activation_fn=None)
            net = net + block2_scored

            # finally, upsample x4
            net = slim.conv2d_transpose(net, 64, [16, 16], stride=4, scope='upsample_3')

            # add a few conv layers as the output
            net = slim.conv2d(net, 32, [3, 3], scope='conv8_1')
            net = slim.conv2d(net, 32, [3, 3], scope='conv8_2')

        ########
        # Logits
        with slim.arg_scope([slim.conv2d, ],
                            normalizer_fn=slim.batch_norm,
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

            total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                updates = tf.group(*update_ops)
                with tf.control_dependencies([updates]):
                    total_loss = tf.identity(total_loss)

            self.op_loss = total_loss


class TiramisuModel(BaseModel):
    def __init__(self, **kwargs):
        super(TiramisuModel, self).__init__()
        logging.info('Creating model: %s', type(self).__name__)

        self.nb_classes = kwargs.get('nb_classes')

        self.regularization = kwargs.get('regularization', 0.005)

        self.batch_norm_params = {'is_training': self.is_train, 'decay': 0.999, }

    def _layer(self, inpt, nb_filters, scope=None):
        l = slim.conv2d(inpt, nb_filters, [3, 3], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm, normalizer_params=self.batch_norm_params,
                        scope=scope)
        l = slim.dropout(l, keep_prob=0.8, is_training=self.is_train, scope=scope)
        return l

    def _transition_down(self, inpt, nb_filters, scope=None):
        l = slim.conv2d(inpt, nb_filters, [1, 1], activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm, normalizer_params=self.batch_norm_params,
                        scope=scope)
        l = slim.dropout(l, keep_prob=0.8, is_training=self.is_train, scope=scope)
        l = slim.max_pool2d(l, [2, 2], scope=scope)
        return l

    def _transition_up(self, inpt, nb_filters, scope=None):
        l = slim.conv2d_transpose(inpt, nb_filters, [3, 3], stride=2,
                                  normalizer_fn=slim.batch_norm, normalizer_params=self.batch_norm_params,
                                  scope=scope)
        return l

    def _dense_block(self, inpt, nb_filters, nb_layers, scope=None):
        outputs = []
        inpt_i = inpt
        for i in range(nb_layers):
            l = self._layer(inpt_i, nb_filters=nb_filters, scope=str(scope) + '/l' + str(i + 1))

            outputs.append(l)
            inpt_i = tf.concat([inpt_i, l], axis=-1)

        oupt = tf.concat(outputs, axis=-1)
        return oupt

    def build_model(self):

        with tf.variable_scope('input'):
            input = self.input_dict['X']

            targets = self.input_dict['Y_softmax']
            targets_one_hot = tf.one_hot(targets, self.nb_classes + 1)

        with tf.variable_scope('tiramisu'):
            nb_layers_per_block = [4, 5, 7, 10, ]  # 12,
            nb_layers_bottleneck = 12  # 15
            nb_filters_initial = 48
            growth_rate = 16

            ##############
            # Initial conv
            initial = slim.conv2d(input, nb_filters_initial, [3, 3],
                                  normalizer_fn=slim.batch_norm, normalizer_params=self.batch_norm_params,
                                  weights_regularizer=slim.l2_regularizer(self.regularization),
                                  scope='initial')

            #############
            # Sample down
            with tf.variable_scope('down'):
                dense_blocks = []
                inpt_i = initial
                for i, nb_layers in enumerate(nb_layers_per_block):
                    db = self._dense_block(inpt_i, growth_rate, nb_layers, scope='DB_' + str(nb_layers) + '_layers')
                    db_concat = tf.concat([db, inpt_i], axis=-1)

                    nb_filters = db_concat.get_shape()[3]
                    td = self._transition_down(db_concat, nb_filters, scope='TD_' + str(nb_layers) + '_layers')

                    dense_blocks.append(db_concat)
                    inpt_i = td

            ############
            # Bottleneck
            with tf.variable_scope('bottleneck'):
                db = self._dense_block(inpt_i, growth_rate, nb_layers_bottleneck,
                                       scope='DB_' + str(nb_layers_bottleneck) + '_layers')
                db_concat = tf.concat([db, inpt_i], axis=-1)

            ###########
            # Sample up
            nb_layers_last = nb_layers_bottleneck
            with tf.variable_scope('up'):
                inpt_i = db_concat
                for i, nb_layers in reversed(list(enumerate(nb_layers_per_block))):
                    tu = self._transition_up(inpt_i, nb_layers_last * growth_rate,
                                             scope='TU_' + str(nb_layers) + '_layers')
                    db = self._dense_block(tu, growth_rate, nb_layers, scope='DB_' + str(nb_layers) + '_layers')
                    db_concat = tf.concat([tu, db, dense_blocks[i]], axis=-1)

                    inpt_i = db_concat
                    nb_layers_last = nb_layers

            ########
            # Logits
            with tf.name_scope('logits'):
                logits = slim.conv2d(db_concat, self.nb_classes + 1, [1, 1], activation_fn=None,
                                     normalizer_fn=slim.batch_norm, normalizer_params=self.batch_norm_params,
                                     weights_regularizer=slim.l2_regularizer(self.regularization),
                                     scope='conv_final_classes'
                                     )

            with tf.name_scope('prediction'):
                classes_probs = tf.nn.softmax(logits)

                self.op_predict = classes_probs

            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=targets_one_hot, logits=logits)
                loss_ce = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=[1, 2]))

                tf.losses.add_loss(loss_ce)

                total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                if update_ops:
                    updates = tf.group(*update_ops)
                    with tf.control_dependencies([updates]):
                        total_loss = tf.identity(total_loss)

                self.op_loss = total_loss


ex_name = 'simple_model_v2'
ex = Experiment(ex_name)


@ex.config
def config():
    nb_epoch = 20
    batch_size = 64

    model_name_prefix = 'softmax_pansharpen'
    model_name_suffix = 'no_vehicles'
    model_name = '{}_{}'.format(model_name_prefix, model_name_suffix)

    classes_to_skip = [9, 10]
    patch_size = [224, 224]
    val_size = 256

    nb_iterations = 1000000
    nb_samples_train = 1000
    nb_samples_val = 512
    batch_size = 30

    regularization = 0.0005

    model_load_step = -1
    prediction_images = 'train'
    debug = False


@ex.named_config
def big_objects():
    classes_to_skip = [2, 5, 9, 10]
    model_name_suffix = 'big_objects'
    # model_load_step = 10875


@ex.named_config
def small_objects():
    classes_to_skip = [1, 3, 4, 6, 7, 8]
    model_name_suffix = 'small_objects'
    # model_load_step = 11225

@ex.named_config
def model_vgg():
    model_name_prefix = 'softmax_pansharpen_vgg'
    model_class = 'SimpleModel'
    patch_size = [128, 128]
    batch_size = 50


@ex.named_config
def model_tiramisu():
    model_name_prefix = 'softmax_pansharpen_tiramisu'
    model_class = 'TiramisuModel'
    patch_size = [80, 80]
    batch_size = 40


@ex.named_config
def debug_run():
    debug = True
    model_name = 'debug'

    nb_iterations = 3
    nb_samples_train = 10
    nb_samples_val = 5
    batch_size = 5


@ex.named_config
def prediction():
    # model_load_step = 20000  # big
    # model_load_step = 17778 # small
    # batch_size = 100
    # patch_size = [128, 128]

    patch_size = [960, 960]
    batch_size = 1


@ex.capture
def create_model(model_params, model_class):
    logging.info('Creating model: %s', model_class)
    return globals()[model_class](**model_params)


@ex.capture
def evaluate_model_jaccard(model, images, images_masks_stacked, images_data, needed_classes, kind, batch_size):
    data_dict = sample_data_dict(images, images_masks_stacked, images_data, kind, needed_classes)
    nb_samples = len(data_dict['X'])

    Y_pred_probs = model.predict(data_dict, batch_size=batch_size)
    Y_pred = np.stack([convert_softmax_to_masks(Y_pred_probs[i]) for i in range(nb_samples)], axis=0)

    Y = data_dict['Y']
    Y = Y[:, :, :, needed_classes]
    jaccard = jaccard_coef(Y_pred, Y, mean=False)

    return jaccard


@ex.capture
def sample_data_dict(images, images_masks_stacked, images_data, kind, needed_classes,
                     patch_size, nb_samples_train, nb_samples_val, val_size):
    if kind == 'val':
        nb_samples = nb_samples_val
    elif kind == 'train':
        nb_samples = nb_samples_train
    else:
        raise AttributeError('Kind {} is not supported'.format(kind))

    patches = sample_patches(images, [images_masks_stacked, images_data], [patch_size, patch_size],
                             nb_samples, kind=kind, val_size=val_size, needed_classes=None)  # needed_classes

    Y, X = patches[0], patches[1]
    Y_softmax = convert_masks_to_softmax(Y, needed_classes=needed_classes)

    data_dict = {'X': X, 'Y': Y, 'Y_softmax': Y_softmax}

    return data_dict


@ex.main
def main(model_name, classes_to_skip, patch_size, nb_iterations, batch_size, debug, regularization,
         model_load_step, prediction_images):
    # set-up matplotlib
    matplotlib_setup()

    logging.info('Model name: %s', model_name)

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
    model = create_model(model_params)  # SimpleModel(**model_params)
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
                data_dict_train = sample_data_dict(images, images_masks_stacked, images_data, 'train', needed_classes)
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
                if iteration_number % 500 == 0:
                    model_filename = os.path.join(MODELS_DIR, model_name)
                    saved_filename = model.save_model(model_filename)
                    logging.info('Model saved: %s', saved_filename)

            except KeyboardInterrupt:
                break

        model_filename = os.path.join(MODELS_DIR, model_name)
        saved_filename = model.save_model(model_filename)
        logging.info('Model saved: %s', saved_filename)

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

        if prediction_images == 'train':
            target_images = images_train
        elif prediction_images == 'test':
            target_images = images_test
        else:
            raise ValueError('Prediction images `{}` unknown'.format(prediction_images))

        for img_number, img_id in enumerate(target_images):
            if img_id != '6060_2_3':
                continue

            img_filename = os.path.join(IMAGES_NORMALIZED_DATA_DIR, img_id + '.pkl')
            img_normalized = load_pickle(img_filename)

            patches, patches_coord = split_image_to_patches([img_normalized, ], [patch_size, ], overlap=0.8)
            logging.info('Patches: %s', len(patches[0]))

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

            save_prediction_mask(IMAGES_PREDICTION_MASK_DIR, masks, img_id, model_name)

            logging.info('Predicted: %s/%s [%.2f]',
                         img_number + 1, len(target_images), 100 * (img_number + 1) / len(target_images))

        result = {

        }
        return result


if __name__ == '__main__':
    ex.run_commandline()
