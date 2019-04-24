import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.losses as losses
import tensorflow.train as train
import sys
sys.path.insert(0, "../")
import eager_STFT as eagerSTFT
import scipy.signal as signal
from IPython.core.debugger import Tracer
debug_here = Tracer()


def compute_parameter_total(trainable_variables):
    total_parameters = 0
    for variable in trainable_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print('var_name', variable.name, 'shape', shape, 'dim', len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        print('parameters', variable_parameters)
        total_parameters += variable_parameters
    print('total:', total_parameters)
    return total_parameters


def mask_nan_to_num(input_array, dropout_rate):
    if dropout_rate > 0:
        random_array = np.random.uniform(0, 1, input_array.shape)
        input_array = np.where(random_array > dropout_rate, input_array, np.NaN)
    return np.concatenate([np.nan_to_num(input_array),
                           np.isnan(input_array).astype(np.float32)],
                          -1)


def mean_relative_error(labels, predictions, mask=None):
    if mask is not None:
        mask_labels = np.where(mask == 1, labels,
                               np.zeros(shape=predictions.shape))
        mask_predictions = np.where(mask == 1, predictions,
                                    np.zeros(shape=predictions.shape))
        return np.sum(np.abs(mask_labels - mask_predictions))/np.sum(np.abs(mask_labels))
    else:
        return np.sum(np.abs(predictions - labels))/np.sum(np.abs(labels))


def mean_absolute_error(labels, predictions, mask=None):
    if mask is not None:
        mask_labels = np.where(mask == 1, labels,
                               np.zeros(shape=predictions.shape))
        mask_predictions = np.where(mask == 1, predictions,
                                    np.zeros(shape=predictions.shape))
        return np.sum(np.abs(mask_predictions - mask_labels))/np.prod(mask_labels.shape)
    else:
        return np.sum(np.abs(predictions - labels))/np.prod(labels.shape)


class CNNImputationGraph(object):
    def __init__(self,
                 learning_rate=0.001,
                 dropout_rate=0.3,
                 noise_std=.5,
                 data_format='channels_last',
                 activation=layers.LeakyReLU(),
                 sequence_length=168,
                 features=36,
                 channels=1):

        layer_lst = []
        layer_lst.append(layers.Dropout(0.6))
        layer_lst.append(layers.Conv2D(5, [12, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.Conv2D(5, [12, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Conv2D(5, [12, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Conv2D(5, [12, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.Conv2D(1, [12, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.input_values = tf.placeholder(
                tf.float32, [None, sequence_length, features, channels])
            if noise_std > 0:
                self.input_values += tf.random.normal(tf.shape(self.input_values),
                                                      stddev=noise_std)
            self.targets = tf.placeholder(
                tf.float32, [None, sequence_length, features, 1])
            hidden_and_out = [self.input_values]
            with tf.variable_scope('classification_CNN'):
                for layer in layer_lst:
                    print(hidden_and_out[-1])
                    hidden_and_out.append(layer(hidden_and_out[-1]))
            print(hidden_and_out[-1])

            def residual_out_fun(input_values, last_hidden):
                return input_values + last_hidden

            def identity_out_fun(input_values, last_hidden):
                return last_hidden

            out_fun = residual_out_fun
            self.out = out_fun(self.input_values, hidden_and_out[-1])
            self.loss = tf.losses.mean_squared_error(self.targets, self.out)
            # opt = train.RMSPropOptimizer(learning_rate)
            opt = train.AdamOptimizer(learning_rate)
            self.weight_update = opt.minimize(self.loss)

            test_hidden_and_out = [self.input_values]
            with tf.variable_scope('classification_CNN', reuse=True):
                for layer in layer_lst:
                    if type(layer) is not layers.Dropout:
                        print(test_hidden_and_out[-1])
                        test_hidden_and_out.append(layer(test_hidden_and_out[-1]))
            print(test_hidden_and_out[-1])
            self.test_out = out_fun(self.input_values, test_hidden_and_out[-1])
            self.test_loss = tf.losses.mean_squared_error(self.targets, self.test_out)
            self.input_loss = tf.losses.mean_squared_error(self.targets,
                                                           self.input_values)

            self.init_global = tf.initializers.global_variables()
            self.init_local = tf.initializers.local_variables()
            self.parameter_total = compute_parameter_total(tf.trainable_variables())


class DfnImputation(object):

    def __init__(self,
                 learning_rate=0.001,
                 data_format='channels_last',
                 activation=layers.LeakyReLU(),
                 decay_rate=None,
                 decay_steps=None,
                 sequence_length=36,
                 features=36):

        self._filter_size = [sequence_length, features]

        layer_lst = []
        layer_lst.append(layers.Conv2D(16, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Conv2D(32, [3, 3], [2, 2], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Conv2D(64, [3, 3], [2, 2], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Conv2D(86, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.UpSampling2D([2, 2],
                                             data_format=data_format))
        layer_lst.append(layers.Conv2D(64, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.UpSampling2D([2, 2],
                                             data_format=data_format))
        layer_lst.append(layers.Conv2D(32, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))

        df_layer = layers.Conv2D(
            filters=self._filter_size[0]*self._filter_size[1],
            kernel_size=[1, 1], padding="SAME")
        db_layer = layers.Conv2D(filters=1, kernel_size=[1, 1], padding="SAME")

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.input_values = tf.placeholder(
                tf.float32, [None, sequence_length, features, 2])
            self.targets = tf.placeholder(
                tf.float32, [None, sequence_length, features, 1])
            hidden_and_out = [self.input_values]

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.variable_scope('imputation_CNN'):
                for layer in layer_lst:
                    print(hidden_and_out[-1])
                    hidden_and_out.append(layer(hidden_and_out[-1]))
                print(hidden_and_out[-1])

                def compute_dfn(input_array, out):
                    filters = df_layer(out)
                    bias = db_layer(out)
                    input_frame_transformed = tf.extract_image_patches(
                        tf.expand_dims(input_array[:, :, :, 0], -1),
                        [1, self._filter_size[0], self._filter_size[1], 1],
                        strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
                    out = tf.reduce_sum(filters * input_frame_transformed,
                                        -1, keepdims=True)
                    with tf.variable_scope('residual_connection'):
                        out = out + tf.expand_dims(input_array[:, :, :, 0], -1) + bias
                    return out, filters, bias

                self.out, _, _ = \
                    compute_dfn(self.input_values, hidden_and_out[-1])

            mask_loss = False
            if mask_loss:
                with tf.variable_scope('mask_loss'):
                    mask = tf.expand_dims(self.input_values[:, :, :, 1], -1)
                    mask = tf.cast(mask, tf.bool)
                    mask_targets = tf.where(mask, self.targets,
                                            tf.zeros_like(self.targets))
                    mask_out = tf.where(mask, self.out, tf.zeros_like(self.out))
                    self.loss = tf.losses.mean_squared_error(mask_targets, mask_out)
            else:
                self.loss = tf.losses.mean_squared_error(self.targets, self.out)
            # opt = train.RMSPropOptimizer(learning_rate)

            if decay_rate and decay_steps:
                learning_rate = tf.train.exponential_decay(learning_rate,
                                                           self.global_step,
                                                           decay_steps, decay_rate,
                                                           staircase=True)
            self.learning_rate_summary = tf.summary.scalar('learning_rate',
                                                           learning_rate)

            opt = train.AdamOptimizer(learning_rate)
            self.weight_update = opt.minimize(self.loss, global_step=self.global_step)

            test_hidden_and_out = [self.input_values]
            with tf.variable_scope('classification_CNN', reuse=True):
                for layer in layer_lst:
                    if type(layer) is not layers.Dropout:
                        print(test_hidden_and_out[-1])
                        test_hidden_and_out.append(layer(test_hidden_and_out[-1]))
                print(test_hidden_and_out[-1])
                self.test_out, self.dynamic_filters, self.dynamic_bias = \
                    compute_dfn(self.input_values, test_hidden_and_out[-1])
            self.test_loss = tf.losses.mean_squared_error(self.targets, self.test_out)

            self.input_loss = tf.losses.mean_squared_error(
                self.targets, tf.expand_dims(self.input_values[:, :, :, 0], -1))

            self.init_global = tf.initializers.global_variables()
            self.init_local = tf.initializers.local_variables()
            self.parameter_total = compute_parameter_total(tf.trainable_variables())

    def train(self, sess, input_lst, target_lst, tensorboard_logger, input_dropout):
        assert len(input_lst) == len(target_lst)
        for i in range(len(input_lst)):
            input_array = np.expand_dims(input_lst[i], -1)
            target_array = np.expand_dims(target_lst[i], -1)
            input_array = mask_nan_to_num(input_array, dropout_rate=input_dropout)
            target_array = mask_nan_to_num(target_array, dropout_rate=0)
            # debug_here()
            removed_mask = input_array[:, :, :, 1] - target_array[:, :, :, 1]
            input_array[:, :, :, 1] = removed_mask
            feed_dict = {self.input_values: input_array,
                         self.targets: np.expand_dims(target_array[:, :, :, 0], -1)}
            loss_np, input_loss_np, _, np_step, lr_summary = \
                sess.run([self.loss, self.input_loss, self.weight_update,
                          self.global_step, self.learning_rate_summary],
                         feed_dict=feed_dict)
            tensorboard_logger.add_np_scalar(loss_np, np_step, tag='train_loss')
            tensorboard_logger.add_tf_summary(lr_summary, np_step)
            return loss_np, input_loss_np, np_step

    def val(self, sess, input_lst_val, target_lst_val, tensorboard_logger,
            air_handler, sequence_length):
        norm_data_val_lst = []
        norm_data_gt_val_lst = []
        test_out_np_val_lst = []
        # print('train', e, i, loss_np, input_loss_np, loss_np/input_loss_np*100)
        # do a validation pass over the data.
        for j in range(len(input_lst_val)):
            norm_data_val = np.expand_dims(input_lst_val[j], -1)
            norm_data_gt_val = np.expand_dims(target_lst_val[j], -1)
            norm_data_val = mask_nan_to_num(norm_data_val, dropout_rate=0)
            norm_data_gt_val = mask_nan_to_num(norm_data_gt_val, dropout_rate=0)
            feed_dict = {
                self.input_values: norm_data_val,
                self.targets: np.expand_dims(norm_data_gt_val[:, :, :, 0], -1)}
            loss_np_val, input_loss_np_val, test_out_np_val, np_step = \
                sess.run([self.loss, self.input_loss,
                          self.test_out,
                          self.global_step], feed_dict=feed_dict)
            sel = int(sequence_length/2)
            norm_data_val_lst.append(norm_data_val[:, sel, :, :])
            norm_data_gt_val_lst.append(norm_data_gt_val[:, sel, :, :])
            test_out_np_val_lst.append(test_out_np_val[:, sel, :, :])

        norm_data_val = np.concatenate(norm_data_val_lst, 0)
        norm_data_gt_val = np.concatenate(norm_data_gt_val_lst, 0)
        test_out_np_val = np.concatenate(test_out_np_val_lst, 0)

        assert np.linalg.norm(norm_data_val[:, :, 0]
                              - np.nan_to_num(air_handler.norm_data_val)) == 0

        removed_mask = norm_data_val[:, :, 1] - norm_data_gt_val[:, :, 1]
        mean = air_handler.mean
        std = air_handler.std
        mre_idt = mean_relative_error(norm_data_gt_val[:, :, 0]*std + mean,
                                      norm_data_val[:, :, 0]*std + mean,
                                      mask=removed_mask)
        # mae_val = mean_absolute_error(norm_data_gt_val[:, :, :, 0]*std + mean,
        #                               test_out_np_val[:, :, :, 0]*std + mean,
        #                               mask=removed_mask)
        mre_val = mean_relative_error(norm_data_gt_val[:, :, 0]*std + mean,
                                      test_out_np_val[:, :, 0]*std + mean,
                                      mask=removed_mask)

        tensorboard_logger.add_np_scalar(mre_val, np_step, tag='mre_val')
        tensorboard_logger.add_np_image(norm_data_val[72:108, :, 0],
                                        np_step, tag='norm_data_val')
        tensorboard_logger.add_np_image(test_out_np_val[72:108, :, 0],
                                        np_step, tag='norm_data_val')
        tensorboard_logger.add_np_image(norm_data_gt_val[72:108, :, 0],
                                        np_step, tag='norm_data_val')
        to_plot = np.abs(test_out_np_val[72:108, :, 0] - norm_data_val[72:108, :, 0])
        tensorboard_logger.add_np_image(to_plot, np_step, tag='out_in_diff')
        to_plot = np.abs(test_out_np_val[72:108, :, 0] -
                         norm_data_gt_val[72:108, :, 0])
        tensorboard_logger.add_np_image(to_plot, np_step, tag='out_gt_diff')
        to_plot = np.abs(norm_data_gt_val[72:108, :, 0] - norm_data_val[72:108, :, 0])
        tensorboard_logger.add_np_image(to_plot, np_step, tag='in_gt_diff')
        return loss_np_val, mre_val, mre_idt, test_out_np_val


class AdvDfnImputation(object):

    def __init__(self,
                 learning_rate=0.001,
                 data_format='channels_last',
                 activation=layers.LeakyReLU(),
                 decay_rate=None,
                 decay_steps=None,
                 sequence_length=36,
                 features=36):

        self._filter_size = [sequence_length, features]

        def create_network():
            layer_lst = []
            layer_lst.append(layers.Conv2D(16, [3, 3], [1, 1], 'SAME',
                                           data_format=data_format,
                                           activation=activation))
            # layer_lst.append(layers.Dropout(dropout_rate))
            layer_lst.append(layers.Conv2D(32, [3, 3], [2, 2], 'SAME',
                                           data_format=data_format,
                                           activation=activation))
            # layer_lst.append(layers.Dropout(dropout_rate))
            layer_lst.append(layers.Conv2D(64, [3, 3], [2, 2], 'SAME',
                                           data_format=data_format,
                                           activation=activation))
            # layer_lst.append(layers.Dropout(dropout_rate))
            layer_lst.append(layers.Conv2D(86, [3, 3], [1, 1], 'SAME',
                                           data_format=data_format,
                                           activation=activation))
            # layer_lst.append(layers.Dropout(dropout_rate))
            layer_lst.append(layers.UpSampling2D([2, 2],
                                                 data_format=data_format))
            layer_lst.append(layers.Conv2D(64, [3, 3], [1, 1], 'SAME',
                                           data_format=data_format,
                                           activation=activation))
            # layer_lst.append(layers.Dropout(dropout_rate))
            layer_lst.append(layers.UpSampling2D([2, 2],
                                                 data_format=data_format))
            layer_lst.append(layers.Conv2D(32, [3, 3], [1, 1], 'SAME',
                                           data_format=data_format,
                                           activation=activation))
            return layer_lst

        df_layer = layers.Conv2D(
            filters=self._filter_size[0]*self._filter_size[1],
            kernel_size=[1, 1], padding="SAME")
        db_layer = layers.Conv2D(filters=1, kernel_size=[1, 1], padding="SAME")
        # todo try sigmoid activation.
        to_mask_layer = layers.Conv2D(1, [1, 1], [1, 1], 'SAME',
                                      data_format=data_format)

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.input_values = tf.placeholder(
                tf.float32, [None, sequence_length, features, 2])
            self.targets = tf.placeholder(
                tf.float32, [None, sequence_length, features, 1])
            hidden_and_out = [self.input_values]

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            with tf.variable_scope('imputation_CNN'):
                layer_lst = create_network()
                for layer in layer_lst:
                    print(hidden_and_out[-1])
                    hidden_and_out.append(layer(hidden_and_out[-1]))
                print(hidden_and_out[-1])

                def compute_dfn(input_array, out):
                    filters = df_layer(out)
                    bias = db_layer(out)
                    input_frame_transformed = tf.extract_image_patches(
                        tf.expand_dims(input_array[:, :, :, 0], -1),
                        [1, self._filter_size[0], self._filter_size[1], 1],
                        strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
                    out = tf.reduce_sum(filters * input_frame_transformed,
                                        -1, keepdims=True)
                    with tf.variable_scope('residual_connection'):
                        out = out + tf.expand_dims(input_array[:, :, :, 0], -1) + bias
                    return out, filters, bias

                self.out, _, _ = \
                    compute_dfn(self.input_values, hidden_and_out[-1])

            def run_adv(adv_input_values, layer_lst, reuse=None):
                adv_hidden_and_out = [adv_input_values]
                with tf.variable_scope('adversarial_CNN', reuse=reuse):
                    for layer in layer_lst:
                        print(adv_hidden_and_out[-1])
                        adv_hidden_and_out.append(layer(adv_hidden_and_out[-1]))
                    print(adv_hidden_and_out[-1])
                    out = to_mask_layer(adv_hidden_and_out[-1])
                    return out

            adv_lst = create_network()
            adv_int = tf.expand_dims(self.input_values[:, :, :, 0], -1)
            self.input_adv_out = run_adv(adv_int, adv_lst)
            self.adv_out = run_adv(self.out, adv_lst, reuse=True)

            input_mask = tf.expand_dims(self.input_values[:, :, :, 1], -1)
            self.adv_input_loss = tf.losses.sigmoid_cross_entropy(
                logits=self.input_adv_out, multi_class_labels=input_mask)
            self.adv_dfn_loss = tf.losses.sigmoid_cross_entropy(
                logits=self.adv_out, multi_class_labels=input_mask)
            # TODO: Finish me!
            self.dfn_loss = tf.reduce_mean(self.adv_out)

            if decay_rate and decay_steps:
                learning_rate = tf.train.exponential_decay(learning_rate,
                                                           self.global_step,
                                                           decay_steps, decay_rate,
                                                           staircase=True)
            self.learning_rate_summary = tf.summary.scalar('learning_rate',
                                                           learning_rate)

            opt = train.AdamOptimizer(learning_rate)
            imputation_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='imputation_CNN')
            adv_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                         scope='adversarial_CNN')
            self.weight_update_class = opt.minimize(self.dfn_loss,
                                                    global_step=self.global_step,
                                                    var_list=imputation_vars)
            self.weight_update_adv_in = opt.minimize(self.adv_input_loss,
                                                     var_list=adv_vars)
            self.weight_update_adv_dfn = opt.minimize(self.adv_dfn_loss,
                                                      var_list=adv_vars)
            self.input_loss = tf.losses.mean_squared_error(
                self.targets, tf.expand_dims(self.input_values[:, :, :, 0], -1))
            self.mse_loss = tf.losses.mean_squared_error(
                tf.expand_dims(self.targets[:, :, :, 0], -1),
                tf.expand_dims(self.input_values[:, :, :, 0], -1))
            self.test_out = self.out

            self.init_global = tf.initializers.global_variables()
            self.init_local = tf.initializers.local_variables()
            self.parameter_total = compute_parameter_total(tf.trainable_variables())

    def train_adv(self, sess, input_lst, target_lst, tensorboard_logger, input_dropout):
        assert len(input_lst) == len(target_lst)
        adv_input_loss = None
        adv_dfn_loss = None
        for i in range(len(input_lst)):
            input_array = np.expand_dims(input_lst[i], -1)
            target_array = np.expand_dims(target_lst[i], -1)
            input_array = mask_nan_to_num(input_array, dropout_rate=input_dropout)
            target_array = mask_nan_to_num(target_array, dropout_rate=0)
            # debug_here()
            # removed_mask = input_array[:, :, :, 1] - target_array[:, :, :, 1]
            # input_array[:, :, :, 1] = removed_mask
            feed_dict = {self.input_values: input_array,
                         self.targets: np.expand_dims(target_array[:, :, :, 0], -1)}
            if i % 2 == 0:
                adv_input_loss, _, lr_summary = \
                    sess.run([self.adv_input_loss, self.weight_update_adv_in,
                              self.learning_rate_summary],
                             feed_dict=feed_dict)
            if i % 2 == 1:
                adv_dfn_loss, _, lr_summary = \
                    sess.run([self.adv_dfn_loss, self.weight_update_adv_dfn,
                              self.learning_rate_summary],
                             feed_dict=feed_dict)
        return adv_input_loss, adv_dfn_loss

    def train(self, sess, input_lst, target_lst, tensorboard_logger, input_dropout):
        assert len(input_lst) == len(target_lst)
        for i in range(len(input_lst)):
            input_array = np.expand_dims(input_lst[i], -1)
            target_array = np.expand_dims(target_lst[i], -1)
            input_array = mask_nan_to_num(input_array, dropout_rate=input_dropout)
            target_array = mask_nan_to_num(target_array, dropout_rate=0)
            # debug_here()
            # removed_mask = input_array[:, :, :, 1] - target_array[:, :, :, 1]
            # input_array[:, :, :, 1] = removed_mask
            feed_dict = {self.input_values: input_array,
                         self.targets: np.expand_dims(target_array[:, :, :, 0], -1)}
            if i % 3 == 0:
                loss_np, input_loss_np, _, np_step, lr_summary = \
                    sess.run([self.mse_loss, self.input_loss, self.weight_update_adv_in,
                              self.global_step, self.learning_rate_summary],
                             feed_dict=feed_dict)
            if i % 3 == 1:
                loss_np, input_loss_np, _, np_step, lr_summary = \
                    sess.run([self.mse_loss, self.input_loss, self.weight_update_adv_dfn,
                              self.global_step, self.learning_rate_summary],
                             feed_dict=feed_dict)
            if i % 3 == 2:
                loss_np, input_loss_np, _, np_step, lr_summary = \
                    sess.run([self.mse_loss, self.input_loss, self.weight_update_class,
                              self.global_step, self.learning_rate_summary],
                             feed_dict=feed_dict)

            tensorboard_logger.add_np_scalar(loss_np, np_step, tag='train_loss')
            tensorboard_logger.add_tf_summary(lr_summary, np_step)
            return loss_np, input_loss_np, np_step

    def val(self, sess, input_lst_val, target_lst_val, tensorboard_logger,
            air_handler, sequence_length):
        norm_data_val_lst = []
        norm_data_gt_val_lst = []
        test_out_np_val_lst = []
        # print('train', e, i, loss_np, input_loss_np, loss_np/input_loss_np*100)
        # do a validation pass over the data.
        for j in range(len(input_lst_val)):
            norm_data_val = np.expand_dims(input_lst_val[j], -1)
            norm_data_gt_val = np.expand_dims(target_lst_val[j], -1)
            norm_data_val = mask_nan_to_num(norm_data_val, dropout_rate=0)
            norm_data_gt_val = mask_nan_to_num(norm_data_gt_val, dropout_rate=0)
            feed_dict = {
                self.input_values: norm_data_val,
                self.targets: np.expand_dims(norm_data_gt_val[:, :, :, 0], -1)}
            loss_np_val, input_loss_np_val, test_out_np_val, np_step = \
                sess.run([self.mse_loss, self.input_loss,
                          self.test_out,
                          self.global_step], feed_dict=feed_dict)
            sel = int(sequence_length/2)
            norm_data_val_lst.append(norm_data_val[:, sel, :, :])
            norm_data_gt_val_lst.append(norm_data_gt_val[:, sel, :, :])
            test_out_np_val_lst.append(test_out_np_val[:, sel, :, :])

        norm_data_val = np.concatenate(norm_data_val_lst, 0)
        norm_data_gt_val = np.concatenate(norm_data_gt_val_lst, 0)
        test_out_np_val = np.concatenate(test_out_np_val_lst, 0)

        assert np.linalg.norm(norm_data_val[:, :, 0]
                              - np.nan_to_num(air_handler.norm_data_val)) == 0

        removed_mask = norm_data_val[:, :, 1] - norm_data_gt_val[:, :, 1]
        mean = air_handler.mean
        std = air_handler.std
        mre_idt = mean_relative_error(norm_data_gt_val[:, :, 0]*std + mean,
                                      norm_data_val[:, :, 0]*std + mean,
                                      mask=removed_mask)
        # mae_val = mean_absolute_error(norm_data_gt_val[:, :, :, 0]*std + mean,
        #                               test_out_np_val[:, :, :, 0]*std + mean,
        #                               mask=removed_mask)
        mre_val = mean_relative_error(norm_data_gt_val[:, :, 0]*std + mean,
                                      test_out_np_val[:, :, 0]*std + mean,
                                      mask=removed_mask)

        tensorboard_logger.add_np_scalar(mre_val, np_step, tag='mre_val')
        tensorboard_logger.add_np_image(norm_data_val[72:108, :, 0],
                                        np_step, tag='norm_data_val')
        tensorboard_logger.add_np_image(test_out_np_val[72:108, :, 0],
                                        np_step, tag='norm_data_val')
        tensorboard_logger.add_np_image(norm_data_gt_val[72:108, :, 0],
                                        np_step, tag='norm_data_val')
        to_plot = np.abs(test_out_np_val[72:108, :, 0] - norm_data_val[72:108, :, 0])
        tensorboard_logger.add_np_image(to_plot, np_step, tag='out_in_diff')
        to_plot = np.abs(test_out_np_val[72:108, :, 0] -
                         norm_data_gt_val[72:108, :, 0])
        tensorboard_logger.add_np_image(to_plot, np_step, tag='out_gt_diff')
        to_plot = np.abs(norm_data_gt_val[72:108, :, 0] - norm_data_val[72:108, :, 0])
        tensorboard_logger.add_np_image(to_plot, np_step, tag='in_gt_diff')
        return loss_np_val, mre_val, mre_idt, test_out_np_val
