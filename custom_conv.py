import tensorflow as tf
from IPython.core.debugger import Tracer
debug_here = Tracer()


def complex_conv1D(h, filter_width, depth, stride, padding, scope='', reuse=None):
    """
    Implement W*h by using the distributive property of the convolution.
    """
    in_channels = h.get_shape().as_list()[2]

    with tf.variable_scope('complex_conv1D' + scope, reuse=reuse):
        if 0:
            print("REAL!")
            Wstack = tf.get_variable('cmplx_conv_weights',
                                     [filter_width, in_channels, depth],
                                     initializer=tf.glorot_normal_initializer())
            return tf.nn.conv1d(tf.abs(h), Wstack, stride=stride, padding=padding)
        else:
            Wstack = tf.get_variable('cmplx_conv_weights',
                                     [filter_width, in_channels, depth, 2],
                                     initializer=tf.glorot_normal_initializer())
            f_real = Wstack[:, :, :, 0]
            f_imag = Wstack[:, :, :, 1]
            x = tf.real(h)
            y = tf.imag(h)

            cat_x = tf.concat([x, y], axis=-1)
            cat_kernel_4_real = tf.concat([f_real, -f_imag], axis=-2)
            cat_kernel_4_imag = tf.concat([f_imag, f_real], axis=-2)
            cat_kernels_4_complex = tf.concat([cat_kernel_4_real,
                                               cat_kernel_4_imag],
                                              axis=-1)
            conv = tf.nn.conv1d(value=cat_x, filters=cat_kernels_4_complex,
                                stride=stride, padding=padding)
            conv_2 = tf.split(conv, axis=-1, num_or_size_splits=2)
            return tf.complex(conv_2[0], conv_2[1])


class ComplexConv2D(object):
    def __init__(self, depth, filters, strides, padding, scope='',
                 activation=None):
        self.depth = depth
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.scope = scope
        self.activation = activation

    def __call__(self, h, reuse=None):
        """
        Implement W*h by using the distributive property of the convolution.
        """
        in_channels = h.get_shape().as_list()[-1]

        with tf.variable_scope('complex_conv2D' + self.scope, reuse=reuse):
            if 0:
                print("REAL!")
                Wstack = tf.get_variable('cmplx_conv_weights',
                                         [self.filters[0], self.filters[1],
                                          in_channels, self.depth],
                                         initializer=tf.glorot_normal_initializer())
                return tf.nn.conv2d(tf.abs(h), Wstack,
                                    stride=[1, self.strides[0], self.strides[1], 1],
                                    padding=self.padding)
            else:
                Wstack = tf.get_variable('cmplx_conv_weights',
                                         [self.filters[0],
                                          self.filters[1],
                                          in_channels, self.depth, 2],
                                         initializer=tf.glorot_normal_initializer())
                f_real = Wstack[:, :, :, :, 0]
                f_imag = Wstack[:, :, :, :, 1]
                x = tf.real(h)
                y = tf.imag(h)

                cat_x = tf.concat([x, y], axis=-1)
                cat_kernel_4_real = tf.concat([f_real, -f_imag], axis=-2)
                cat_kernel_4_imag = tf.concat([f_imag, f_real], axis=-2)
                cat_kernels_4_complex = tf.concat([cat_kernel_4_real,
                                                   cat_kernel_4_imag],
                                                  axis=-1)
                conv = tf.nn.conv2d(input=cat_x, filter=cat_kernels_4_complex,
                                    strides=[1, self.strides[0], self.strides[1], 1],
                                    padding=self.padding)
                conv_2 = tf.split(conv, axis=-1, num_or_size_splits=2)
                if self.activation:
                    return self.activation(tf.complex(conv_2[0], conv_2[1]))
                else:
                    return tf.complex(conv_2[0], conv_2[1])


class ComplexUpSampling1D(tf.keras.layers.UpSampling1D):
    '''
    A complex valued 1D-upsampling layer.
    '''

    def __init__(self, size=2, data_format=None, **kwargs):
        super().__init__(size=size, data_format=None, **kwargs)

    def __call__(self, z):
        x = tf.real(z)
        y = tf.imag(z)
        cat_x = tf.concat([x, y], axis=-1)
        up = super().__call__(cat_x)
        up_2 = tf.split(up, axis=-1, num_or_size_splits=2)
        up_z = tf.complex(up_2[0], up_2[1])
        return up_z


class ComplexUpSampling2D(tf.keras.layers.UpSampling2D):
    '''
    A complex valued 2D-upsampling layer.
    '''

    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super().__init__(size=size, data_format=None, **kwargs)

    def __call__(self, z):
        x = tf.real(z)
        y = tf.imag(z)
        # ok, because bilinear interpolation involves the sum
        # of scaled entries at the nodes.
        cat_x = tf.concat([x, y], axis=-1)
        up = super().__call__(cat_x)
        up_2 = tf.split(up, axis=-1, num_or_size_splits=2)
        up_z = tf.complex(up_2[0], up_2[1])
        return up_z


class SplitRelu(object):

    def __init__(self, scope=''):
        self._scope = scope

    def __call__(self, z, reuse=None):
        with tf.variable_scope('split_relu' + self._scope):
            x = tf.real(z)
            y = tf.imag(z)
            return tf.complex(tf.nn.relu(x), tf.nn.relu(y))


def complex_max_pool1d(h, ksize, strides, padding, scope=None):
    """
    Complex pooling.
    """
    with tf.variable_scope('complex_max_pool1d' + scope):
        real_pool = tf.nn.max_pool(tf.expand_dims(tf.real(h), 1), ksize, strides, padding)
        imag_pool = tf.nn.max_pool(tf.expand_dims(tf.imag(h), 1), ksize, strides, padding)
    return tf.complex(real_pool, imag_pool)
