import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib2tikz as tikz
import sys
sys.path.insert(0, "../")

tf.enable_eager_execution()


def gaussian_window(sigma, window_size):
    '''
    Implementation of a gaussian window function with
    parameter sigma.

    Returns:
        window tensor of shape [window_size]
    '''
    with tf.variable_scope("gaussian_window"):
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


def plank_taper(epsilon, window_size):
    '''
    Plank taper window implementation similar to:
    https://arxiv.org/pdf/1003.2939.pdf
    The window center is moved from zero into the center
    of the window.
    '''
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


def tukey_window(alpha, window_size):
    '''
    Tukey window implementation.
    '''
    pi = tf.constant(np.pi, tf.float32)
    N = window_size - 1
    n = tf.linspace(float(0), float(window_size), window_size)
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


def gauss_plank_window(epsilon, sigma, window_size):
    gauss = gaussian_window(sigma, window_size)
    plank = plank_taper(epsilon, window_size)
    return gauss * plank


window_size = 100

# gaussian plot
std_vector = np.linspace(0.0, 1.0, 10)
for std in std_vector:
    win = gaussian_window(std, window_size)
    plt.plot(win.numpy(), label='std ' + str(std))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
# tikz.save('gaussian_window.tex')
plt.close()
plt.clf()

# plank plot
epsilon_vector = np.linspace(0.0, 0.5, 10)
for eps in epsilon_vector:
    win = plank_taper(eps, window_size)
    plt.plot(win.numpy(), label='eps ' + str(eps))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
# tikz.save('plank_window.tex')
plt.close()
plt.clf()

# tukey plot
alpha_vector = np.linspace(0.0, 1.0, 10)
for alpha in alpha_vector:
    win = tukey_window(alpha, window_size)
    plt.plot(win.numpy(), label='alpha' + str(alpha))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# tikz.save('tukey_window.tex')
plt.show()

# gauss - plank window.
std_vector = np.linspace(0.25, 1.0, 3)
epsilon_vector = np.linspace(0.0, 0.5, 3)
for eps in epsilon_vector:
    for std in std_vector:
        win = gauss_plank_window(eps, std, window_size).numpy()
        plt.plot(win, label='std' + str(std) + 'eps' + str(eps))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
tikz.save('gauss_plank_window.tex')
# plt.show()
