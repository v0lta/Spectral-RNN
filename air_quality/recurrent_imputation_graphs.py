import tensorflow as tf
import numpy as np
import scipy.signal as signal
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as backend
import tensorflow.losses as losses
import tensorflow.train as train
import matplotlib.pyplot as plt
from air_data_handler import AirDataHandler
# from imputation_graphs import CNNImputationGraph
from imputation_graphs import DfnImputation
from imputation_graphs import compute_parameter_total

from IPython.core.debugger import Tracer
debug_here = Tracer()


# implement a dfn RNN-cell with access to past encodings.
class DFNCell(layers.Layer):

    def __init__(self, filter_size, enc_shapes, mid_shapes,
                 dec_shapes, enc_strides):
        self.filter_size = filter_size
        self.enc_shapes = enc_shapes
        self.mid_shapes = mid_shapes
        self.dec_shapes = dec_shapes
        self.enc_strides = enc_strides
        self.activation = layers.LeakyReLU()
        self.padding = 'same'
        assert len(self.enc_shapes) == len(self.enc_strides)
        super().__init__()

    def build(self, input_shape):
        self.enc_kernel_lst = []
        self.enc_bias_lst = []
        self.mid_kernel_lst = []
        self.mid_bias_lst = []
        self.dec_kernel_lst = []
        self.dec_bias_lst = []
        for ks_no, kernel_shape in enumerate(self.enc_shapes):
            self.enc_kernel_lst.append(
                self.add_weight(shape=kernel_shape, initializer='uniform',
                                name='ek_'+str(ks_no)))
            self.enc_bias_lst.append(
                self.add_weight(shape=(kernel_shape[-1],), initializer='uniform',
                                name='eb_'+str(ks_no)))
        for ks_no, kernel_shape in enumerate(self.mid_shapes):
            self.mid_kernel_lst.append(
                self.add_weight(shape=kernel_shape, initializer='uniform',
                                name='mk_'+str(ks_no)))
            self.mid_bias_lst.append(
                self.add_weight(shape=(kernel_shape[-1],), initializer='uniform',
                                name='mb_'+str(ks_no)))
        for ks_no, kernel_shape in enumerate(self.dec_shapes):
            self.dec_kernel_lst.append(
                self.add_weight(shape=kernel_shape, initializer='uniform',
                                name='dk_'+str(ks_no)))
            self.dec_bias_lst.append(
                self.add_weight(shape=(kernel_shape[-1],), initializer='uniform',
                                name='db_'+str(ks_no)))

        self.df_kernel = self.add_weight(
            shape=(1, 1, self.dec_shapes[-1][-1], np.prod(self.filter_size)),
            initializer='uniform', name='dfk')
        self.df_bias = self.add_weight(shape=(np.prod(self.filter_size),),
                                       initializer='uniform', name='dfb')
        self.db_kernel = self.add_weight(shape=(1, 1, self.dec_shapes[-1][-1], 1),
                                         initializer='uniform', name='dbk')
        self.db_bias = self.add_weight(shape=(1,), initializer='uniform', name='dbb')
        self.built = True

    def call(self, inputs, states):
        enc_hidden_lst = [inputs]
        # the encoder
        for kernel_no, encoder_kernel in enumerate(self.enc_kernel_lst):
            conv_out = backend.conv2d(x=enc_hidden_lst[-1],
                                      kernel=encoder_kernel,
                                      padding=self.padding,
                                      strides=self.enc_strides[kernel_no],
                                      data_format='channels_last')
            enc_hidden_lst.append(
                self.activation(conv_out + self.enc_bias_lst[kernel_no]))
            print(enc_hidden_lst[-1])
        # the mid part
        mid_hidden_lst = [states]
        for kernel_no, mid_kernel in enumerate(self.mid_kernel_lst):
            state_conv = backend.conv2d(x=mid_hidden_lst[-1],
                                        kernel=mid_kernel,
                                        padding=self.padding,
                                        data_format='channels_last')
            mid_hidden_lst.append(
                self.activation(state_conv + self.mid_bias_lst[kernel_no]))
            print(mid_hidden_lst[-1])
        mid = mid_hidden_lst[-1] + enc_hidden_lst[-1]
        new_state = mid
        # the decoder_part
        upsample_lst = self.enc_strides[::-1]
        dec_hidden_out_lst = [mid]
        for kernel_no, dec_kernel in enumerate(self.dec_kernel_lst):
            if upsample_lst[kernel_no] != (1, 1):
                up_out = backend.resize_images(
                    x=dec_hidden_out_lst[-1], height_factor=upsample_lst[kernel_no][0],
                    width_factor=upsample_lst[kernel_no][1], data_format='channels_last')
            else:
                up_out = dec_hidden_out_lst[-1]
            conv_out = backend.conv2d(x=up_out,
                                      kernel=dec_kernel,
                                      padding=self.padding,
                                      data_format='channels_last')

            dec_hidden_out_lst.append(
                self.activation(conv_out + self.dec_bias_lst[kernel_no]))
            print(dec_hidden_out_lst[-1])

        filter_conv = backend.conv2d(x=dec_hidden_out_lst[-1],
                                     kernel=self.df_kernel,
                                     padding=self.padding,
                                     data_format='channels_last')
        dyn_filters = self.activation(filter_conv + self.df_bias)
        bias_conv = backend.conv2d(x=dec_hidden_out_lst[-1],
                                   kernel=self.db_kernel,
                                   padding=self.padding,
                                   data_format='channels_last')
        dyn_bias = self.activation(bias_conv + self.db_bias)
        input_frame_transformed = tf.extract_image_patches(
            tf.expand_dims(inputs[:, :, :, 0], -1) + dyn_bias,
            [1, self.filter_size[0], self.filter_size[1], 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding="SAME")
        out = tf.reduce_sum(
            dyn_filters * input_frame_transformed,
            -1, keepdims=True)
        return out, new_state


class RNNImputation(object):
        def __init__(self,
                     learning_rate=0.001,
                     cell_size=128,
                     data_format='channels_last',
                     activation=layers.LeakyReLU(),
                     sequence_length=168,
                     features=36,
                     decay_rate=None,
                     decay_steps=None):

            self.tf_graph = tf.Graph()
            with self.tf_graph.as_default():
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.input_values = tf.placeholder(
                    tf.float32, [None, sequence_length, features, 2])
                self.targets = tf.placeholder(
                    tf.float32, [None, sequence_length, features, 1])

                flat_input = tf.concat([self.input_values[:, :, :, 0],
                                        self.input_values[:, :, :, 1]], -1)

                cell = layers.GRUCell(cell_size)
                rnn = layers.RNN(cell, return_sequences=True)
                blstm_layer = layers.Bidirectional(rnn)
                out_map = layers.Dense(features)

                rnn_out = blstm_layer(flat_input)
                self.out = out_map(rnn_out)
                self.out = tf.expand_dims(self.out, -1)
                debug_here()

                mask_loss = False
                if mask_loss:
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

                self.test_out = self.out
                self.test_loss = self.loss
                self.input_loss = tf.losses.mean_squared_error(
                    self.targets, tf.expand_dims(self.input_values[:, :, :, 0], -1))

                self.init_global = tf.initializers.global_variables()
                self.init_local = tf.initializers.local_variables()
                self.parameter_total = compute_parameter_total(tf.trainable_variables())
