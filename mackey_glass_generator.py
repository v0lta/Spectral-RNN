import tensorflow as tf
from IPython.core.debugger import Tracer
debug_here = Tracer()


def generate_mackey(batch_size=100, tmax=200, delta_t=1, rnd=True):
    """
    Generate synthetic training data using the Lorenz system
    of equations (http://www.scholarpedia.org/article/Mackey-Glass_equation):
    dx/dt = beta*(x'/(1+x'))
    The system is simulated using a forward euler scheme
    (https://en.wikipedia.org/wiki/Euler_method).

    Returns:
        spikes: A Tensor of shape [batch_size, time, 1],
    """
    with tf.variable_scope('mackey_generator'):
        steps = int(tmax/delta_t) + 100

        # multi-dimensional data.
        def mackey(x, tau, gamma=0.1, beta=0.2, n=10):
            return beta*x[:, -tau]/(1 + tf.pow(x[:, -tau], n)) - gamma*x[:, -1]

        tau = 17
        x0 = tf.ones([tau])
        x0 = tf.stack(batch_size*[x0], axis=0)
        if rnd:
            print('Mackey initial state is random.')
            x0 += tf.random_uniform(x0.shape, -0.1, 0.1)

        x = x0
        with tf.variable_scope("forward_euler"):
            for _ in range(steps):
                res = tf.expand_dims(x[:, -1] + delta_t*mackey(x, tau), -1)
                x = tf.concat([x, res], -1)
    discard = 100 + tau
    return x[:, discard:]


if __name__ == "__main__":
    tf.enable_eager_execution()
    import matplotlib.pyplot as plt
    mackey = generate_mackey(tmax=1200, delta_t=1)
    print(mackey.shape)
    plt.plot(mackey[0, :].numpy())
    plt.show()


class MackeyGenerator(object):
    '''
    Generates lorenz attractor data in 1 or 3d on the GPU.
    '''

    def __init__(self, batch_size, tmax, delta_t, restore_and_plot=False):
        self.batch_size = batch_size
        self.tmax = tmax
        self.delta_t = delta_t
        self.restore_and_plot = restore_and_plot

    def __call__(self):
        data_nd = generate_mackey(tmax=self.tmax, delta_t=self.delta_t,
                                  batch_size=self.batch_size,
                                  rnd=not self.restore_and_plot)
        data_nd = tf.expand_dims(data_nd, -1)
        print('data_nd_shape', data_nd.shape)
        return data_nd
