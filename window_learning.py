import math
import eager_STFT as eagerSTFT
import tensorflow as tf
import ipdb
import matplotlib.pyplot as plt


graph = tf.Graph()
learning_rate = 0.001
batch_size = 64
window_size = 128
iterations = 10
overlap = int(window_size*0.75)


def gaussian_window(window_size, epsilon):
    '''
    Implementation of a gaussian window function with
    parameter sigma.

    Returns:
        window tensor of shape [window_size]
    '''
    with tf.variable_scope("gaussian_window"):
        init = tf.constant(0.7)
        sigma = tf.get_variable('sigma', initializer=init,
                                trainable=True)
        # positive
        sigma = sigma*sigma
        # positive in [0.01, 0.501] this prevents degneration
        # into rectangular window.
        # sigma = tf.nn.sigmoid(sigma)/2.0 + 0.01
        # positive
        N = window_size
        n = tf.linspace(float(0), float(window_size),
                        window_size)
        w = (n - (N - 1.0)/2.0)/(sigma * (N - 1.0)/2.0)
        w = -0.5*w*w
        w = tf.math.exp(w)
        return w + epsilon


def kaiser_window(alpha, window_size):
    # numerical issues nans in half of the window....
    # do not use for now.
    with tf.variable_scope('kaiser_window'):
        N = window_size
        n = tf.linspace(float(0), float(window_size), window_size)
        top = 2.0*n / (N-1.0) - 1.0
        top = math.pi * alpha * tf.sqrt(1.0 - top*top)
        top = tf.math.bessel_i0(top)
        bottom = tf.math.bessel_i0(math.pi * alpha)
        return top/bottom


# def plank_taper(epsilon, window_size):
#     '''
#     plank taper window
#     '''
#     N = window_size
#     n = tf.linspace(float(0), float(window_size), window_size)
#     ap = 1.0 / (1.0 + 2*n / (N - 1) - 1)
#     bp = 1.0 / (1.0 - 2*epsilon + (2*n / (N - 1) - 1))
#     Zp = 2*epsilon*(ap + bp)
#     rising_edge = 1 / (tf.math.exp(Zp) + 1)
#     const_samples = tf.floor((1.0 - epsilon)*(N - 1) - epsilon*(N-1), tf.int32)
#     ones = tf.ones([const_samples], tf.float32)
#     am = 1.0 / (1.0 - 2*n / (N - 1) - 1)
#     bm = 1.0 / (1.0 - 2*epsilon - (2*n / (N - 1) - 1))
#     Zm = 2*epsilon*(am + bm)
#     falling_edge = 1 / (tf.math.exp(Zp) + 1)
    # todo. gather and concatenate....


if __name__ == "__main__":
    with graph.as_default():
        spikes, states = eagerSTFT.generate_data(batch_size=batch_size, delta_t=0.01,
                                                 tmax=10.24, rnd=True)
        # epsilon = tf.get_variable('epsilon', shape=[1], dtype=tf.float32)
        # epsilon = tf.nn.sigmoid(epsilon)
        epsilon = tf.constant(0.001)
        window = gaussian_window(window_size, epsilon)
        last_spikes = tf.transpose(spikes, [0, 2, 1])
        result_tf = eagerSTFT.stft(last_spikes, window, window_size, overlap)
        rec_tf = eagerSTFT.istft(result_tf,
                                 window,
                                 nperseg=window_size,
                                 noverlap=overlap,
                                 epsilon=None)

        loss = tf.losses.mean_squared_error(last_spikes, rec_tf[:, :, :1025])
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        opt_op = optimizer.minimize(loss)
        init_op = tf.global_variables_initializer()
        ipdb.set_trace()

    with tf.Session(graph=graph) as sess:
        for i in range(0, iterations):
            init_op.run()
            out_loss, out_window, out_epsilon, _ = sess.run([loss, window, epsilon,
                                                             opt_op])
            print(i, out_loss, out_epsilon)
