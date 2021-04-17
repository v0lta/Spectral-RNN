import tensorflow as tf
from tensorflow.python.ops.nn_ops import _get_noise_shape


def complex_dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    '''
    Implementation of complex dropout based on tf.nn.dropout.
    '''
    with tf.name_scope(name, "complex_dropout", [x]) as name:
        # Early return if nothing needs to be dropped.
        if isinstance(keep_prob, float) and keep_prob == 1:
            return x

        noise_shape = _get_noise_shape(x, noise_shape)
        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(
            noise_shape, seed=seed, dtype=tf.float32)
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = tf.floor(random_tensor)
        ret = tf.complex(tf.div(tf.real(x), keep_prob) * binary_tensor,
                         tf.div(tf.imag(x), keep_prob) * binary_tensor)
    return ret
