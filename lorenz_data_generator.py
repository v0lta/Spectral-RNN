import tensorflow as tf
import eager_STFT as eagerSTFT


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
            data_nd = eagerSTFT.generate_data(self.tmax, self.delta_t,
                                              batch_size=self.batch_size,
                                              rnd=not self.restore_and_plot)[0]
        else:
            data_nd = eagerSTFT.generate_data(self.tmax, self.delta_t,
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
