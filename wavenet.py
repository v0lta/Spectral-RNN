# TODO: finish me!

import tensorflow as tf
import eager_STFT as eagerSTFT


class WaveNet(object):
    def __init__(self, dilation, depth, kernel_size):
        # the lists must have the same length (one entry per layer).
        assert len(dilation) == len(depth)
        assert len(depth) == len(kernel_size)
        self._dilation = dilation
        self._depth = depth
        self._kernel_size = kernel_size
        self._padding = "VALID"
        self._conv = tf.layers.Conv1D

    def residual_block(self, input, pos):
        """
        Definition of a wave-net residual block.
        """
        with tf.variable_scope("wave_residual"):
            conv_out = self._conv(filters=self._depth[pos]*2,
                                  kernel_size=self._kernel_size[pos],
                                  dilation_rate=self._dilation[pos])
            conv_out_tanh, conv_out_sigmoid = tf.split(conv_out, 2, axis=-1)
            z = tf.nn.tanh(conv_out_tanh)*tf.nn.sigmoid(conv_out_sigmoid)




    def train_step(input):
        """
        Returns a single WaveNet prediciton.
        """
        return None

    def evaluation_step(input):
        """
        Returns multiple recurrent prediction steps.
        """
        return None
