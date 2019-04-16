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
        # print('parameters', variable_parameters)
        total_parameters += variable_parameters
    print('total:', total_parameters)
    return total_parameters


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
                 dropout_rate=0.3,
                 data_format='channels_last',
                 activation=layers.LeakyReLU(),
                 sequence_length=168,
                 features=36):

        self._filter_size = [16, features]

        layer_lst = []
        # layer_lst.append(layers.Dropout(0.2))
        layer_lst.append(layers.Conv2D(16, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.Conv2D(32, [3, 3], [2, 2], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.Conv2D(64, [3, 3], [2, 2], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.Conv2D(128, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.UpSampling2D([2, 2],
                                             data_format=data_format))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.Conv2D(64, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.UpSampling2D([2, 2],
                                             data_format=data_format))
        layer_lst.append(layers.Conv2D(32, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))

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
            with tf.variable_scope('classification_CNN'):
                for layer in layer_lst:
                    print(hidden_and_out[-1])
                    hidden_and_out.append(layer(hidden_and_out[-1]))
            print(hidden_and_out[-1])

            def compute_dfn(input_array, out):
                filters = df_layer(out)
                bias = db_layer(out)
                input_frame_transformed = tf.extract_image_patches(
                    tf.expand_dims(input_array[:, :, :, 0], -1) + bias,
                    [1, self._filter_size[0], self._filter_size[1], 1],
                    strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
                return tf.reduce_sum(
                    filters * input_frame_transformed,
                    -1, keepdims=True)

            self.out = compute_dfn(self.input_values, hidden_and_out[-1])

            mask_loss = False
            if mask_loss:
                mask = tf.expand_dims(self.input_values[:, :, :, 1], -1)
                mask = tf.cast(mask, tf.bool)
                mask_targets = tf.where(mask, self.targets, tf.zeros_like(self.targets))
                mask_out = tf.where(mask, self.out, tf.zeros_like(self.out))
                self.loss = tf.losses.mean_squared_error(mask_targets, mask_out)
            else:
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
            self.test_out = compute_dfn(self.input_values,
                                        test_hidden_and_out[-1])
            self.test_loss = tf.losses.mean_squared_error(self.targets, self.test_out)

            self.input_loss = tf.losses.mean_squared_error(
                self.targets, tf.expand_dims(self.input_values[:, :, :, 0], -1))

            self.init_global = tf.initializers.global_variables()
            self.init_local = tf.initializers.local_variables()
            self.parameter_total = compute_parameter_total(tf.trainable_variables())


class CNNImputationFFT(object):

    def __init__(self,
                 learning_rate=0.001,
                 dropout_rate=0.3,
                 data_format='channels_last',
                 activation=layers.LeakyReLU(),
                 sequence_length=168,
                 features=36):

        self._filter_size = [3, 3]

        layer_lst = []
        # layer_lst.append(layers.Dropout(0.2))
        layer_lst.append(layers.Conv2D(20, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.Conv2D(30, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.Conv2D(30, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))
        layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(layers.Conv2D(20, [3, 3], [1, 1], 'SAME',
                                       data_format=data_format,
                                       activation=activation))

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.input_values = tf.placeholder(
                tf.float32, [None, sequence_length, features, 2])
            self.targets = tf.placeholder(
                tf.float32, [None, sequence_length, features, 1])

            self.input_values_0 = self.input_values[:, :, :, 0]
            window_size = int(sequence_length/2)
            window = tf.constant(signal.get_window('hann', window_size),
                                 dtype=tf.float32)
            input_values_t = tf.transpose(self.input_values_0, [0, 2, 1])
            input_values_f = eagerSTFT.stft(input_values_t, window,
                                            window_size,
                                            noverlap=int(window_size/2))
            input_values_f = tf.concat([tf.real(input_values_f),
                                        tf.imag(input_values_f)], -1)
            # freq_shape = input_values_f.shape.as_list()
            hidden_and_out = [input_values_f]
            with tf.variable_scope('classification_CNN'):
                for layer in layer_lst:

                    print(hidden_and_out[-1])
                    hidden_and_out.append(layer(hidden_and_out[-1]))
            print(hidden_and_out[-1])
            debug_here()
            out = tf.complex(hidden_and_out[-1][:, :, :, :10],
                             hidden_and_out[-1][:, :, :, 10:])
            self.out = eagerSTFT.istft(out, window,
                                       window_size,
                                       noverlap=int(window_size/2))
            self.out = tf.expand_dims(self.out, -1)
            mask_loss = False
            if mask_loss:
                mask = tf.expand_dims(self.input_values[:, :, :, 1], -1)
                mask = tf.cast(mask, tf.bool)
                mask_targets = tf.where(mask, self.targets, tf.zeros_like(self.targets))
                mask_out = tf.where(mask, self.out, tf.zeros_like(self.out))
                self.loss = tf.losses.mean_squared_error(mask_targets, mask_out)
            else:
                self.loss = tf.losses.mean_squared_error(self.targets, self.out)
            # opt = train.RMSPropOptimizer(learning_rate)
            opt = train.AdamOptimizer(learning_rate)
            self.weight_update = opt.minimize(self.loss)

            test_hidden_and_out = [input_values_f]
            with tf.variable_scope('classification_CNN', reuse=True):
                for layer in layer_lst:
                    if type(layer) is not layers.Dropout:
                        print(test_hidden_and_out[-1])
                        test_hidden_and_out.append(layer(test_hidden_and_out[-1]))
            print(test_hidden_and_out[-1])
            test_out = tf.complex(test_hidden_and_out[-1][:, :, :, :10],
                                  test_hidden_and_out[-1][:, :, :, 10:])
            self.test_out = eagerSTFT.istft(test_out, window, window_size,
                                            noverlap=int(window_size/2),
                                            epsilon=0.1)
            self.test_out = tf.expand_dims(self.test_out, -1)
            self.test_loss = tf.losses.mean_squared_error(self.targets, self.test_out)
            self.input_loss = tf.losses.mean_squared_error(
                self.targets, tf.expand_dims(self.input_values_0, -1))

            self.init_global = tf.initializers.global_variables()
            self.init_local = tf.initializers.local_variables()
            self.parameter_total = compute_parameter_total(tf.trainable_variables())
