import math
import numpy as np
import eager_STFT as eagerSTFT
import tensorflow as tf
# import ipdb
import matplotlib.pyplot as plt


def gaussian_window(window_size, sigma=None):
    '''
    Implementation of a gaussian window function with
    parameter sigma.

    Returns:
        window tensor of shape [window_size]
    '''
    with tf.variable_scope("gaussian_window"):
        init = tf.constant(0.7)
        if sigma is None:
            sigma = tf.get_variable('sigma', initializer=init,
                                    trainable=True)
            # sigma must be > 0!
            sigma = sigma*sigma
        tf.summary.scalar('window_sigma', sigma)
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
        return w

# def kaiser_window(alpha, window_size):
#     # numerical issues nans in half of the window....
#     # do not use for now.
#     # TODO: debug.
#     with tf.variable_scope('kaiser_window'):
#         N = window_size
#         n = tf.linspace(float(0), float(window_size), window_size)
#         top = 2.0*n / (N-1.0) - 1.0
#         top = math.pi * alpha * tf.sqrt(1.0 - top*top)
#         top = tf.math.bessel_i0(top)
#         bottom = tf.math.bessel_i0(math.pi * alpha)
#         return top/bottom


def plank_taper(window_size):
    '''
    Plank taper window implementation similar to:
    https://arxiv.org/pdf/1003.2939.pdf
    The window center is moved from zero into the center
    of the window.
    '''
    with tf.variable_scope('plank_window'):
        init = tf.constant(0.0)
        epsilon = tf.get_variable('epsilon', initializer=init,
                                  trainable=True)
        epsilon = tf.nn.sigmoid(epsilon)/2.0
        # positive
        tf.summary.scalar('window_epsilon', epsilon)

        T = window_size
        t = tf.linspace(float(0), float(window_size) + 1,
                        window_size + 1)
        t1 = 0
        t2 = T/2.0*(1 - 2.0*epsilon)
        t3 = T - T/2.0*(1 - 2.0*epsilon)
        t4 = T
        print(t1, t2, t3, t4)

        Zr = (t2 - t1)/(t - t1) + (t2 - t1)/(t - t2)
        rising_edge = 1 / (tf.math.exp(Zr) + 1)
        Zf = (t3 - t4)/(t - t3) + (t3 - t4)/(t - t4)
        falling_edge = 1 / (tf.math.exp(Zf) + 1)

        t1 = tf.cast(tf.round(t1), tf.int32)
        t2 = tf.cast(tf.round(t2), tf.int32)
        t3 = tf.cast(tf.round(t3), tf.int32)
        t4 = tf.cast(tf.round(t4), tf.int32)

        rising_elements = tf.gather(rising_edge, tf.range(t1, t2))
        ones = tf.ones(t3 - t2, tf.float32)
        falling_elements = tf.gather(falling_edge, tf.range(t3, t4))
        plank_taper_window = tf.concat([rising_elements,
                                        ones,
                                        falling_elements],
                                       axis=0)
    return plank_taper_window


def tukey_window(window_size):
    '''
    Tukey window implementation.
    '''
    with tf.variable_scope('tukey_window'):
        init = tf.constant(0.0)
        alpha = tf.get_variable('alpha', initializer=init,
                                trainable=True)
        alpha = tf.nn.sigmoid(alpha)
        # positive
        tf.summary.scalar('window_alpha', alpha)

        pi = tf.constant(np.pi, tf.float32)
        N = window_size
        n = tf.linspace(float(0), float(window_size) + 1,
                        window_size + 1)
        n1 = alpha*N/2.0
        n2 = N * (1.0 - alpha/2.0)
        rising_edge = 0.5*(1.0 + tf.cos(pi * (2.0*n/(alpha*N) - 1)))
        falling_edge = 0.5*(1.0 + tf.cos(pi * (2*n/(alpha*N)-2.0/alpha + 1)))

        n1 = tf.cast(tf.round(n1), tf.int32)
        n2 = tf.cast(tf.round(n2), tf.int32)

        rising_elements = tf.gather(rising_edge, tf.range(0, n1))
        ones = tf.ones(n2 - n1, tf.float32)
        falling_elements = tf.gather(falling_edge, tf.range(n2, N))
        window = tf.concat([rising_elements,
                            ones,
                            falling_elements],
                           axis=0)
    return window


def gauss_plank_window(window_size):
    '''
    A combined gauss plank fft-window function.
    '''
    with tf.variable_scope('gauss_plank_window'):
        gauss = gaussian_window(window_size)
        plank = plank_taper(window_size)
    return gauss * plank


if __name__ == "__main__":
    import mackey_glass_generator
    graph = tf.Graph()
    learning_rate = 0.001
    batch_size = 2
    window_size = 128
    overlap = int(window_size*0.75)

    with graph.as_default():
        mackey = mackey_glass_generator.generate_mackey(
            batch_size=batch_size, delta_t=0.1, tmax=102.4, rnd=False)
        mackey = tf.expand_dims(mackey, -1)
        # epsilon = tf.get_variable('epsilon', shape=[1], dtype=tf.float32)
        # epsilon = tf.nn.sigmoid(epsilon)
        epsilon = tf.constant(0.001)
        sigma = tf.placeholder(shape=[], dtype=tf.float32)
        spikes = tf.placeholder(shape=[batch_size, 1024, 1], dtype=tf.float32)
        window = gaussian_window(window_size, sigma=sigma)
        last_spikes = tf.transpose(spikes, [0, 2, 1])
        result_tf = eagerSTFT.stft(last_spikes, window, window_size, overlap)
        rec_tf = eagerSTFT.istft(result_tf,
                                 window,
                                 nperseg=window_size,
                                 noverlap=overlap,
                                 epsilon=epsilon)
        init_op = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess:
        init_op.run()
        spikes_np = sess.run([mackey])[0]

        sigma_np = 0.5
        feed_dict = {sigma: sigma_np,
                     spikes: spikes_np}
        time, freq, window_np = sess.run([rec_tf, result_tf, window],
                                         feed_dict=feed_dict)
        plt.plot(time[0, 0, :])
        plt.plot(spikes_np[0, :, 0])
        plt.show()
        plt.plot(window_np)
        plt.show()
        plt.title('sigma_' + str(sigma_np))
        plt.imshow(np.abs(freq[0, 0, :]))
        plt.show()

        sigma_np = 0.32
        feed_dict = {sigma: sigma_np,
                     spikes: spikes_np}
        time, freq, window_np = sess.run([rec_tf, result_tf, window],
                                         feed_dict=feed_dict)
        plt.plot(time[0, 0, :])
        plt.plot(spikes_np[0, :, 0])
        plt.show()
        plt.plot(window_np)
        plt.show()
        plt.imshow(np.abs(freq[0, 0, :]))
        plt.title('sigma_' + str(sigma_np))
        plt.show()

        sigma_np = 1.2
        feed_dict = {sigma: sigma_np,
                     spikes: spikes_np}
        time, freq, window_np = sess.run([rec_tf, result_tf, window],
                                         feed_dict=feed_dict)
        plt.plot(time[0, 0, :])
        plt.plot(spikes_np[0, :, 0])
        plt.show()
        plt.plot(window_np)
        plt.show()
        plt.imshow(np.abs(freq[0, 0, :]))
        plt.title('sigma_' + str(sigma_np))
        plt.show()
