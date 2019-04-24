import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.losses as losses
import tensorflow.train as train
import sys
sys.path.insert(0, "../")
import eager_STFT as eagerSTFT
import scipy.signal as signal
import custom_conv as cconv

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


class CNNImputationFFT(object):

    def __init__(self,
                 learning_rate=0.001,
                 data_format='channels_last',
                 activation=cconv.SplitRelu(),
                 decay_rate=None,
                 decay_steps=None,
                 sequence_length=168,
                 features=36):

        layer_lst = []
        layer_lst.append(cconv.ComplexConv2D(
            36, [3, 3], [1, 1], 'SAME', scope='_1', activation=activation))
        # layer_lst.append(cconv.Dropout(dropout_rate))
        layer_lst.append(cconv.ComplexConv2D(
            40, [3, 3], [2, 1], 'SAME', scope='_2', activation=activation))
        # layer_lst.append(cconv.Dropout(dropout_rate))
        layer_lst.append(cconv.ComplexConv2D(
            44, [3, 3], [2, 1], 'SAME', scope='_3', activation=activation))
        # layer_lst.append(cconv.Dropout(dropout_rate))
        layer_lst.append(cconv.ComplexConv2D(
            48, [3, 3], [1, 1], 'SAME', scope='_4', activation=activation))
        layer_lst.append(cconv.ComplexConv2D(
            52, [3, 3], [1, 1], 'SAME', scope='_5', activation=activation))
        # layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(cconv.ComplexUpSampling2D(size=[2, 1]))
        layer_lst.append(cconv.ComplexConv2D(
            44, [3, 3], [1, 1], 'SAME', scope='_6', activation=activation))
        # layer_lst.append(layers.Dropout(dropout_rate))
        layer_lst.append(cconv.ComplexUpSampling2D(size=[2, 1]))
        layer_lst.append(cconv.ComplexConv2D(
            36, [3, 3], [1, 1], 'SAME', scope='_7', activation=activation))

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.input_values = tf.placeholder(
                tf.float32, [None, sequence_length, features, 2])
            self.targets = tf.placeholder(
                tf.float32, [None, sequence_length, features, 1])

            self.input_values_0 = self.input_values[:, :, :, 0]
            window_size = 19
            window = tf.constant(signal.get_window('boxcar', window_size),
                                 dtype=tf.float32)
            # transpose into batch ,features, time
            input_values_t = tf.transpose(self.input_values_0, [0, 2, 1])
            noverlap = int(window_size-1)
            input_values_f = eagerSTFT.stft(input_values_t, window,
                                            window_size,
                                            noverlap=noverlap)
            # transpose into batch, time, freq, features.
            input_values_f = tf.transpose(input_values_f, [0, 2, 3, 1])
            hidden_and_out = [input_values_f]
            with tf.variable_scope('imputation_CNN'):
                for layer in layer_lst:
                    print(hidden_and_out[-1])
                    hidden_and_out.append(layer(hidden_and_out[-1]))
            print(hidden_and_out[-1])
            # out = input_values_f + hidden_and_out[-1]

            def freq_residuals(out, inputs):
                with tf.variable_scope('freq_residuals'):
                    return out + inputs

            def identity(out, inputs):
                return out

            out = freq_residuals(hidden_and_out[-1], input_values_f)
            out = tf.transpose(out, [0, 3, 1, 2])
            epsilon = 0.0001
            self.out = eagerSTFT.istft(out, window,
                                       window_size,
                                       noverlap=noverlap,
                                       nfft=[window_size],
                                       epsilon=epsilon)
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

            if decay_rate and decay_steps:
                learning_rate = tf.train.exponential_decay(learning_rate,
                                                           self.global_step,
                                                           decay_steps, decay_rate,
                                                           staircase=True)
            self.learning_rate_summary = tf.summary.scalar('learning_rate',
                                                           learning_rate)

            opt = train.AdamOptimizer(learning_rate)
            self.weight_update = opt.minimize(self.loss, global_step=self.global_step)

            self.test_out = self.out
            self.test_loss = tf.losses.mean_squared_error(self.targets, self.test_out)
            self.input_loss = tf.losses.mean_squared_error(
                self.targets, tf.expand_dims(self.input_values_0, -1))

            self.init_global = tf.initializers.global_variables()
            self.init_local = tf.initializers.local_variables()
            self.parameter_total = compute_parameter_total(tf.trainable_variables())
