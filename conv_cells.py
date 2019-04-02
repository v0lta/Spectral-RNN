import numpy as np
import collections
import tensorflow as tf

from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.rnn import RNNCell


class ConvLSTM(RNNCell):
    """Efficient reimplementation of a convolutional LSTM with peepholes.
        Reference:  Convolutional LSTM network: A machine learning approach
                    for precipitation nowcasting."""

    def __init__(self, kernel_size, depth, input_dims, output_depth=None,
                 strides=None, peepholes=True, transpose=False, output_strides=False,
                 reuse=None, normalize=True, is_training=True, couple=True,
                 trainable=True, constrain=False):
        """ input_dims: [height, width, channels]
            kernel_size: The size of the kernels, which are slided over the image
                         [size_in_X, size_in_Y].
            strides: The step sizes with which the sliding is done (will lead to
                     downsampling if >1) [size_in_X, size_in_Y].
            depth: The kernel depth.
            output_depth: The depth of the output convolution.
            peepholes: If true peephole weights will be used.
        """
        super().__init__(_reuse=reuse)
        self.kernel_size = kernel_size
        self.peepholes = peepholes
        self.depth = depth
        self.input_dims = [int(dim) for dim in input_dims]
        if strides is None:
            self.strides = [1, 1]
        else:
            self.strides = strides
        self.output_depth = output_depth
        self.transpose = transpose
        self.output_strides = output_strides
        self.normalize = normalize
        self.is_training = is_training
        self.trainable = trainable
        self.couple = couple
        if constrain is True:
            assert normalize is True
        self.constrained = constrain

    @property
    def recurrent_size(self):
        """ Shape of the tensors flowing along the recurrent connections.
        """
        if self.transpose is False and self.output_strides is False:
            return tf.TensorShape([np.ceil(self.input_dims[0] / self.strides[0]),
                                   np.ceil(self.input_dims[1] / self.strides[1]),
                                   self.depth])
        else:
            return tf.TensorShape([self.input_dims[0],
                                   self.input_dims[1],
                                   self.depth])

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell.
        """
        if self.output_strides is True:
            return tf.TensorShape([np.ceil(self.input_dims[0] / self.strides[0]),
                                   np.ceil(self.input_dims[1] / self.strides[1]),
                                   self.depth])
        elif self.output_depth is None:
            return self.recurrent_size
        else:
            if self.transpose is False:
                return tf.TensorShape([int(self.recurrent_size[0]),
                                       int(self.recurrent_size[1]),
                                       self.output_depth])
            else:
                return tf.TensorShape([np.ceil(self.input_dims[0] * self.strides[0]),
                                       np.ceil(self.input_dims[1] * self.strides[1]),
                                       self.output_depth])

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer,
        a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        return LSTMStateTuple(self.recurrent_size, self.recurrent_size)

    def __call__(self, x, state, scope=None):
        """
          Args:
            x: Input tensor of shape [batch_size, height, width, channels]
          Returns:
            outputs: shape self.output_size
            state: shape self.state_size
        """
        input_shape = tf.Tensor.get_shape(x)
        num_channels = int(input_shape[-1])
        batch_size = x.shape[0].value

        c, h = state
        with tf.variable_scope(scope, default_name=str(type(self).__name__)):
            # print(tf.contrib.framework.get_name_scope())
            # print(tf.get_variable_scope().reuse)
            # print('couple?', self.couple)
            # print('normalize?', self.normalize)

            if self.couple:
                depth_f = 3
            else:
                depth_f = 4

            if self.normalize:
                # normalization according to:
                # Weight Normalization: A Simple Reparameterization to Accelerate
                # Training of Deep Neural Networks by Salimans, Tim
                # Kingma, Diederik P.
                vi = tf.get_variable('input_weights', [int(np.prod(self.kernel_size))
                                                       * num_channels
                                                       * self.depth*depth_f],
                                     trainable=self.trainable)
                gi = tf.get_variable('input_length', [],
                                     trainable=self.trainable)
                vr = tf.get_variable('recurrent_weights', (int(np.prod(self.kernel_size))
                                                           * self.depth
                                                           * self.depth*depth_f),
                                     trainable=self.trainable)
                gr = tf.get_variable('recurrent_length', [],
                                     trainable=self.trainable)
                # gr = tf.constant(0.0001)
                vb = tf.get_variable('bias', [self.depth*depth_f],
                                     trainable=self.trainable)
                gb = tf.get_variable('bias_length', [],
                                     trainable=self.trainable)
                if self.peepholes is True:
                    vpi = tf.get_variable('input_peep', self.recurrent_size,
                                          trainable=self.trainable)
                    gpi = tf.get_variable('input_peep_length', [],
                                          trainable=self.trainable)
                    vpo = tf.get_variable('out_peep', self.recurrent_size,
                                          trainable=self.trainable)
                    gpo = tf.get_variable('output_peep_length', [],
                                          trainable=self.trainable)

                    if self.constrained:
                        gpi = tf.nn.sigmoid(gpi)
                        gpo = tf.nn.sigmoid(gpo)

                    pi = gpi*tf.norm(vpi)*vpi
                    po = gpo*tf.norm(vpo)*vpo
                    if not self.couple:
                        vpf = tf.get_variable('forget_peep', self.recurrent_size,
                                              trainable=self.trainable)
                        gpf = tf.get_variable('forget_peep_length', [],
                                              trainable=self.trainable)

                        if self.constrained:
                            gpf = tf.nn.sigmoid(gpf)

                        pf = gpf*tf.norm(vpf)*vpf

                if self.constrained:
                    gi = tf.nn.sigmoid(gi)
                    gr = tf.nn.sigmoid(gr)
                    gb = tf.nn.sigmoid(gb)

                Wi = gi*tf.norm(vi)*tf.reshape(vi, (self.kernel_size
                                                    + [num_channels]
                                                    + [self.depth*depth_f]))
                Wr = gr*tf.norm(vr)*tf.reshape(vr, (self.kernel_size
                                                    + [self.depth]
                                                    + [self.depth*depth_f]))
                b = gb*tf.norm(vb)*vb
            else:
                Wi = tf.get_variable('input_weights', (self.kernel_size
                                                       + [num_channels]
                                                       + [self.depth*depth_f]),
                                     trainable=self.trainable)
                Wr = tf.get_variable('recurrent_weights', (self.kernel_size
                                                           + [self.depth]
                                                           + [self.depth*depth_f]),
                                     trainable=self.trainable)
                b = tf.get_variable('bias', [self.depth*depth_f],
                                    trainable=self.trainable)
                if self.peepholes is True:
                    pi = tf.get_variable('input_peep', self.recurrent_size,
                                         trainable=self.trainable)
                    po = tf.get_variable('out_peep', self.recurrent_size,
                                         trainable=self.trainable)
                    if not self.couple:
                        pf = tf.get_variable('forget_peep', self.recurrent_size,
                                             trainable=self.trainable)

            # Seperate convolutions for inputs and recurrecies.
            # Slower but allows input downsampling.
            if self.transpose is False and self.output_strides is False:
                input_conv = tf.nn.convolution(x, Wi, 'SAME', strides=self.strides)
            else:
                input_conv = tf.nn.convolution(x, Wi, 'SAME')
            rec_conv = tf.nn.convolution(h, Wr, 'SAME')
            res = input_conv + rec_conv + b
            if self.couple:
                z, i, o = tf.split(res, depth_f, axis=self.recurrent_size.ndims)
            else:
                z, i, f, o = tf.split(res, depth_f, axis=self.recurrent_size.ndims)

            if self.peepholes is True:
                i += c*pi
                if not self.couple:
                    f += c*pf
            z = tf.nn.tanh(z, name='z_tanh')
            i = tf.nn.sigmoid(i, name='i_sig')
            if self.couple:
                f = (1 - i)
            else:
                f = tf.nn.sigmoid(f, name='f_sig')
            c += z*i + f*c
            if self.peepholes is True:
                o += c*po
            o = tf.nn.sigmoid(o, name='o_sig')
            h = o*tf.nn.tanh(c, name='c_tanh')

            # c = tf.clip_by_value(c, -1e10, 1e10)

            state = LSTMStateTuple(c, h)
            if self.output_depth is not None:
                with tf.variable_scope('output_proj'):
                    if self.transpose:
                        Wdeconv = tf.get_variable('input_weights', (self.kernel_size
                                                                    + [self.output_depth]
                                                                    + [self.depth]),
                                                  trainable=self.trainable)
                        shape = [batch_size, int(self.output_size[0]),
                                 int(self.output_size[1]), int(self.output_size[2])]
                        out = tf.nn.tanh(tf.nn.conv2d_transpose(
                            h, Wdeconv, strides=[1] + self.strides + [1],
                            output_shape=shape))
                    else:
                        Wproj = tf.get_variable('projection_weights',
                                                (self.kernel_size
                                                 + [self.depth]
                                                 + [self.output_depth]),
                                                trainable=self.trainable)
                        out = tf.nn.convolution(h, Wproj, padding='SAME')

                        shape = [batch_size, int(self.output_size[0]),
                                 int(self.output_size[1]), int(self.output_size[2])]
                        # out = tf.reshape(out, [batch_size] + self.input_dims[:2] + [-1])
                        out = tf.reshape(out, shape)
                        tf.Tensor.set_shape(out, shape)
            else:
                if self.output_strides is False or self.strides == [1, 1]:
                    out = h
                else:
                    Wproj = tf.get_variable('projection_weights', (self.kernel_size
                                                                   + [self.depth]
                                                                   + [self.depth]),
                                            trainable=self.trainable)
                    # out = tf.nn.convolution(h, Wproj, padding='SAME',
                    #                         strides=self.strides)
                    out = tf.nn.tanh(tf.nn.convolution(h, Wproj, padding='SAME',
                                                       strides=self.strides))
        return out, state


class ConvGRU(RNNCell):
    """Efficient reimplementation of a convolutional LSTM with peepholes.
        Reference:  Convolutional LSTM network: A machine learning approach
                    for precipitation nowcasting."""

    def __init__(self, kernel_size, depth, input_dims, output_depth=None,
                 strides=None, peepholes=True, transpose=False, output_strides=False,
                 reuse=None, normalize=True, is_training=True, trainable=True,
                 constrain=False):
        """ input_dims: [height, width, channels]
            kernel_size: The size of the kernels, which are slided over the image
                         [size_in_X, size_in_Y].
            strides: The step sizes with which the sliding is done (will lead to
                     downsampling if >1) [size_in_X, size_in_Y].
            depth: The kernel depth.
            output_depth: The depth of the output convolution.
            peepholes: If true peephole weights will be used.
        """
        super().__init__(_reuse=reuse)
        self.kernel_size = kernel_size
        self.peepholes = peepholes
        self.depth = depth
        self.input_dims = [int(dim) for dim in input_dims]
        if strides is None:
            self.strides = [1, 1]
        else:
            self.strides = strides
        self.output_depth = output_depth
        self.transpose = transpose
        self.output_strides = output_strides
        self.normalize = normalize
        self.is_training = is_training
        self.trainable = trainable
        if constrain is True:
            assert normalize is True
        self.constrained = constrain

    @property
    def recurrent_size(self):
        """ Shape of the tensors flowing along the recurrent connections.
        """
        if self.transpose is False and self.output_strides is False:
            return tf.TensorShape([np.ceil(self.input_dims[0] / self.strides[0]),
                                   np.ceil(self.input_dims[1] / self.strides[1]),
                                   self.depth])
        else:
            return tf.TensorShape([self.input_dims[0],
                                   self.input_dims[1],
                                   self.depth])

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell.
        """
        if self.output_strides is True:
            return tf.TensorShape([np.ceil(self.input_dims[0] / self.strides[0]),
                                   np.ceil(self.input_dims[1] / self.strides[1]),
                                   self.depth])
        elif self.output_depth is None:
            return self.recurrent_size
        else:
            if self.transpose is False:
                return tf.TensorShape([int(self.recurrent_size[0]),
                                       int(self.recurrent_size[1]),
                                       self.output_depth])
            else:
                return tf.TensorShape([np.ceil(self.input_dims[0] * self.strides[0]),
                                       np.ceil(self.input_dims[1] * self.strides[1]),
                                       self.output_depth])

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        It can be represented by an Integer,
        a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        return LSTMStateTuple(self.recurrent_size, self.recurrent_size)

    def __call__(self, x, state, scope=None):
        """
          Args:
            x: Input tensor of shape [batch_size, height, width, channels]
          Returns:
            outputs: shape self.output_size
            state: shape self.state_size
        """
        input_shape = tf.Tensor.get_shape(x)
        num_channels = int(input_shape[-1])
        batch_size = x.shape[0].value

        c, h = state
        with tf.variable_scope(scope, default_name=str(type(self).__name__)):
            # print(tf.contrib.framework.get_name_scope())
            # print(tf.get_variable_scope().reuse)
            # print('couple?', self.couple)
            # print('normalize?', self.normalize)

            if self.couple:
                depth_f = 3

            if self.normalize:
                # normalization according to:
                # Weight Normalization: A Simple Reparameterization to Accelerate
                # Training of Deep Neural Networks by Salimans, Tim
                # Kingma, Diederik P.
                vr = tf.get_variable('reset_weights', [int(np.prod(self.kernel_size))
                                                       * num_channels
                                                       * self.depth*depth_f],
                                     trainable=self.trainable)
                gr = tf.get_variable('reset_length', [],
                                     initializer=tf.random_normal_initializer(.0, 0.0001),
                                     trainable=self.trainable)
                vu = tf.get_variable('update_weights', (int(np.prod(self.kernel_size))
                                                        * self.depth
                                                        * self.depth*depth_f),
                                     trainable=self.trainable)
                gu = tf.get_variable('update_length', [],
                                     initializer=tf.random_normal_initializer(.0, 0.0001),
                                     trainable=self.trainable)
                # gr = tf.constant(0.0001)
                vb = tf.get_variable('bias', [self.depth*depth_f],
                                     trainable=self.trainable)
                gb = tf.get_variable('bias_length', [],
                                     initializer=tf.zeros_initializer(),
                                     trainable=self.trainable)

                if self.constrained:
                    gr = tf.nn.sigmoid(gr)
                    gu = tf.nn.sigmoid(gu)

                Wr = gr*tf.norm(vr)*tf.reshape(vr, (self.kernel_size
                                                    + [num_channels]
                                                    + [self.depth*depth_f]))
                Wu = gu*tf.norm(vu)*tf.reshape(vu, (self.kernel_size
                                                    + [self.depth]
                                                    + [self.depth*depth_f]))
                b = gb*tf.norm(vb)*vb
            else:
                Wr = tf.get_variable('reset_weights', (self.kernel_size
                                                       + [num_channels]
                                                       + [self.depth*depth_f]),
                                     trainable=self.trainable)
                Wu = tf.get_variable('update_weights', (self.kernel_size
                                                        + [self.depth]
                                                        + [self.depth*depth_f]),
                                     trainable=self.trainable)
                b = tf.get_variable('bias', [self.depth*depth_f],
                                    trainable=self.trainable)

            # Seperate convolutions for inputs and recurrecies.
            # Slower but allows input downsampling.
            if self.transpose is False and self.output_strides is False:
                input_conv = tf.nn.convolution(x, Wr, 'SAME', strides=self.strides)
            else:
                input_conv = tf.nn.convolution(x, Wr, 'SAME')
            rec_conv = tf.nn.convolution(h, Wu, 'SAME')
            res = input_conv + rec_conv + b
            z, r, u = tf.split(res, depth_f, axis=self.recurrent_size.ndims)

            z = tf.nn.tanh(z, name='z_tanh')
            r = tf.nn.sigmoid(r, name='i_sig')
            h += z*u + (1 - u)*h

            if self.output_depth is not None:
                with tf.variable_scope('output_proj'):
                    if self.transpose:
                        Wdeconv = tf.get_variable('input_weights', (self.kernel_size
                                                                    + [self.output_depth]
                                                                    + [self.depth]),
                                                  trainable=self.trainable)
                        shape = [batch_size, int(self.output_size[0]),
                                 int(self.output_size[1]), int(self.output_size[2])]
                        out = tf.nn.tanh(tf.nn.conv2d_transpose(
                            h, Wdeconv, strides=[1] + self.strides + [1],
                            output_shape=shape))
                    else:
                        Wproj = tf.get_variable('projection_weights',
                                                (self.kernel_size
                                                 + [self.depth]
                                                 + [self.output_depth]),
                                                trainable=self.trainable)
                        out = tf.nn.convolution(h, Wproj, padding='SAME')

                        shape = [batch_size, int(self.output_size[0]),
                                 int(self.output_size[1]), int(self.output_size[2])]
                        # out = tf.reshape(out, [batch_size] + self.input_dims[:2] + [-1])
                        out = tf.reshape(out, shape)
                        tf.Tensor.set_shape(out, shape)
            else:
                if self.output_strides is False or self.strides == [1, 1]:
                    out = h
                else:
                    Wproj = tf.get_variable('projection_weights', (self.kernel_size
                                                                   + [self.depth]
                                                                   + [self.depth]),
                                            trainable=self.trainable)
                    # out = tf.nn.convolution(h, Wproj, padding='SAME',
                    #                         strides=self.strides)
                    out = tf.nn.tanh(tf.nn.convolution(h, Wproj, padding='SAME',
                                                       strides=self.strides))
        return out, h
