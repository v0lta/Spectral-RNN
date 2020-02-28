import tensorflow as tf


def generate_data(tmax=20, delta_t=0.01, sigma=10.0,
                  beta=8.0/3.0, rho=28.0, batch_size=100,
                  rnd=True):
    """
    Generate synthetic training data using the Lorenz system
    of equations (https://en.wikipedia.org/wiki/Lorenz_system):
    dxdt = sigma*(y - x)
    dydt = x * (rho - z) - y
    dzdt = x*y - beta*z
    The system is simulated using a forward euler scheme
    (https://en.wikipedia.org/wiki/Euler_method).

    Params:
        tmax: The simulation time.
        delta_t: The step size.
        sigma: The first Lorenz parameter.
        beta: The second Lorenz parameter.
        rho: The thirs Lorenz parameter.
        batch_size: The first batch dimension.
        rnd: If true the lorenz seed is random.
    Returns:
        spikes: A Tensor of shape [batch_size, time, 1],
        states: A Tensor of shape [batch_size, time, 3].
    """
    with tf.variable_scope('lorenz_generator'):
        # multi-dimensional data.
        def lorenz(x, t):
            return tf.stack([sigma*(x[:, 1] - x[:, 0]),
                             x[:, 0]*(rho - x[:, 2]) - x[:, 1],
                             x[:, 0]*x[:, 1] - beta*x[:, 2]],
                            axis=1)

        state0 = tf.constant([8.0, 6.0, 30.0])
        state0 = tf.stack(batch_size*[state0], axis=0)
        if rnd:
            print('Lorenz initial state is random.')
            state0 += tf.random_uniform([batch_size, 3], -4, 4)
        else:
            add_lst = []
            for i in range(batch_size):
                add_lst.append([0, float(i)*(1.0/batch_size), 0])
            add_tensor = tf.stack(add_lst, axis=0)
            state0 += add_tensor
        states = [state0]
        with tf.variable_scope("forward_euler"):
            for _ in range(int(tmax/delta_t)):
                states.append(states[-1] + delta_t*lorenz(states[-1], None))
        states = tf.stack(states, axis=1)

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot(states[:, 0], states[:, 1], states[:, 2], label='lorenz curve')
        # plt.show()

        # single dimensional data.
        spikes = tf.expand_dims(tf.square(states[:, :, 0]), -1)
        # normalize
        states = states/tf.reduce_max(tf.abs(states))
        spikes = spikes/tf.reduce_max(spikes)
    return spikes, states


class LorenzGenerator(object):
    '''
    Generates lorenz attractor data in 1 or 3d on the GPU.
    '''

    def __init__(self, spikes_instead_of_states, batch_size,
                 tmax, delta_t, restore_and_plot=False):
        self.spikes_instead_of_states = spikes_instead_of_states
        self.batch_size = batch_size
        self.tmax = tmax
        self.delta_t = delta_t
        self.restore_and_plot = restore_and_plot

    def __call__(self):
        if self.spikes_instead_of_states:
            data_nd = generate_data(self.tmax, self.delta_t,
                                    batch_size=self.batch_size,
                                    rnd=not self.restore_and_plot)[0]
        else:
            data_nd = generate_data(self.tmax, self.delta_t,
                                    batch_size=self.batch_size,
                                    rnd=not self.restore_and_plot)[1]
        print('data_nd_shape', data_nd.shape)
        return data_nd


if __name__ == "__main__":
    pd = {}
    pd['spikes_instead_of_states'] = True
    pd['tmax'] = 10.24
    pd['delta_t'] = 0.01
    pd['batch_size'] = 100
    pd['input_samples'] = int(pd['tmax']/pd['delta_t'])+1
    generator = LorenzGenerator(
        pd['spikes_instead_of_states'], pd['batch_size'],
        pd['tmax'], pd['delta_t'], restore_and_plot=False)

    graph = tf.Graph()
    with graph.as_default():
        data_nd = generator()

    with tf.Session(graph=graph) as sess:
        data_nd_np = sess.run([data_nd])

    print('done')
