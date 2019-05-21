import numpy as np
import tensorflow as tf
import tensorflow.contrib.signal as tfsignal
import scipy.signal as scisig
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from IPython.core.debugger import Tracer
debug_here = Tracer()


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


def zero_ext(x, n, axis=-1):
    """
    Following:
    https://github.com/scipy/scipy/blob/master/scipy/signal/_arraytools.py
    Zero padding at the boundaries of an array
    Generate a new ndarray that is a zero padded extension of `x` along
    an axis.
    """
    # debug_here()
    # if n < 1:
    #     return x
    # zeros_shape = list(x.shape)
    # zeros_shape[axis] = n
    # zeros = tf.zeros(zeros_shape, dtype=x.dtype)
    # ext = tf.concat((zeros, x, zeros), axis=axis)
    # debug_here()
    dim = len(x.shape)
    paddings = np.zeros([dim, 2], dtype=np.int32)
    paddings[axis] = n
    ext2 = tf.pad(x, tf.constant(paddings, dtype=tf.int32))
    return ext2


def stft(data, window, nperseg, noverlap, nfft=None, sides=None, padded=True,
         scaling='spectrum', boundary='zeros', debug=False):
    # Following:
    # https://github.com/scipy/scipy/blob/v1.1.0/scipy/signal/spectral.py#L847-L991
    # Args:
    #   data: The time domain data to be transformed [expects batch, dim, time].
    #   window: Tensorflow array of size [window_size]
    #   nperseg: The number of samples per window segment.
    #   noverlap: The number of samples overlapping.
    with tf.variable_scope("stft"):
        boundary_funcs = {'zeros': zero_ext,
                          None: None}

        if boundary not in boundary_funcs:
            raise ValueError("Unknown boundary option '{0}', must be one of: {1}"
                             .format(boundary, list(boundary_funcs.keys())))

        if boundary is not None:
            ext_func = boundary_funcs[boundary]
            data = ext_func(data, nperseg//2, axis=-1)

        nstep = nperseg - noverlap
        # do what scipy's spectral_helper does.
        if padded:
            # Pad to integer number of windowed segments
            # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
            dim = len(data.shape)
            nadd = (-(data.shape[-1].value-nperseg) % nstep) % nperseg

            if debug:
                zeros_shape = list(data.shape[:-1]) + [nadd]
            # zeros_shape = list(data.shape[:-1]) + [nadd]
            # data = tf.concat([data, tf.zeros(zeros_shape)], axis=-1)
            zeros = np.zeros([dim, 2], dtype=np.int32)
            zeros[-1, 1] = nadd
            data = tf.pad(data, tf.constant(zeros, dtype=tf.int32))

        # do what numpy's _fft_helper does.
        if nperseg == 1 and noverlap == 0:
            result = tf.expand_dims(data, -1)
        else:
            data_shape = data.shape.as_list()
            step = nperseg - noverlap
            # do the framing
            result = tfsignal.frame(data, nperseg, step)
            # numpy framing.
            shape = data_shape[:-1] + [(data_shape[-1] - noverlap) // step, nperseg]

        # Apply window by multiplication
        assert result.shape.as_list() == shape
        result = window * result
        result = tf.spectral.rfft(result)

        if scaling == 'spectrum':
            scale = 1.0 / tf.reduce_sum(window)**2
        else:
            raise ValueError('Unknown scaling: %r' % scaling)
        scale = tf.sqrt(scale)
        result *= tf.complex(scale, tf.zeros_like(scale))
        # debug_here()
    if debug:
        zeros_shape = list(data.shape[:-1]) + [nadd]
        data_np = np.concatenate((data.numpy(), np.zeros(zeros_shape)), axis=-1)
        strides = data_np.strides[:-1] + (step*data_np.strides[-1], data_np.strides[-1])
        result_np = np.lib.stride_tricks.as_strided(data_np, shape=shape,
                                                    strides=strides)
        result_np = window.numpy() * result_np
        result_np = np.fft.rfft(result_np)
        result_np *= scale.numpy()
        return result, result_np.astype(np.complex64)
    else:
        return result


def istft(Zxx, window, nperseg=None, noverlap=None, nfft=None,
          input_onesided=True, boundary=True, epsilon=None,
          debug=False):
    '''
        Perform the inverse Short Time Fourier transform (iSTFT),
        assuming a time_axis at -2 and a freq_axis at -1.
    Params:
        Zxx: Frequency domain data [batch_size, dim, time, freq].
        window: Window generated by scipy.get_window()
    '''
    with tf.variable_scope("istft"):
        freq_axis = -1
        time_axis = -2

        # debug_here()
        # window = tf.constant(scisig.get_window(window_str, nperseg),
        #                      dtype=tf.float32)

        # if Zxx.ndim < 2:
        #     raise ValueError('Input stft must be at least 2d!')

        # nseg = Zxx.shape[time_axis]
        nseg = Zxx.shape[time_axis].value

        # Assume a onesided input.
        # Assume even segment length
        n_default = 2*(Zxx.shape[freq_axis].value - 1)

        # Check windowing parameters
        if nperseg is None:
            nperseg = n_default
        else:
            nperseg = int(nperseg)
            if nperseg < 1:
                raise ValueError('nperseg must be a positive integer')

        if noverlap is None:
            noverlap = nperseg//2
        else:
            noverlap = int(noverlap)
        if noverlap >= nperseg:
            raise ValueError('noverlap must be less than nperseg.')
        nstep = nperseg - noverlap

        # if not scisig.check_COLA(window_str, nperseg, noverlap):
        #     raise ValueError('Window, STFT shape and noverlap do not satisfy the '
        #                      'COLA constraint.')

        xsubs = tf.spectral.irfft(Zxx)[..., :nperseg]
        # This takes care of the 'spectrum' scaling.
        xsubs_scaled = xsubs * tf.reduce_sum(window)
        unscaled = tfsignal.overlap_and_add(xsubs_scaled*window, nstep)
        norm = tfsignal.overlap_and_add(tf.stack([tf.square(window)]*nseg, 0), nstep)
        if epsilon is None:
            scaled = tf.where(tf.ones_like(unscaled)*norm > 1e-6,
                              unscaled/norm, unscaled)
        else:
            # The epsilon SIGNIFICANTLY increases the reconstruciton error,
            # but the where statement kills the gradient so thats that....
            scaled = unscaled/(norm + epsilon)

        # Remove extension points
        if boundary:
            scaled = scaled[..., nperseg//2:-(nperseg//2)]

    return scaled


if __name__ == "__main__":
    try:
        tf.enable_eager_execution()
    except ValueError:
        print("tensorflow is already in eager mode.")

    # Do some testing!
    # params
    batch_size = 64
    window_size = 32
    overlap = int(window_size*0.75)
    window = tf.constant(scisig.get_window('hann', window_size),
                         dtype=tf.float32)
    # code
    spikes, states = generate_data(batch_size=batch_size, delta_t=0.01, tmax=10.24)
    # plt.plot(spikes.numpy()[0, :, :])
    # plt.savefig('spikes.pdf')
    # plt.show()
    if 1:
        tmp_last_spikes = tf.transpose(spikes, [0, 2, 1])
        result_tf, result_np = stft(tmp_last_spikes, window, window_size, overlap,
                                    debug=True)

        tmp_f, tmp_t, sci_res = scisig.stft(tmp_last_spikes.numpy(),
                                            window=window,
                                            nperseg=window_size,
                                            noverlap=overlap,
                                            axis=-1)

        # debug_here()
        plt.imshow(np.log((np.abs(result_tf.numpy()[0, 0, :, :].T))))
        plt.colorbar()
        plt.show()
        debug_here()

        error = np.linalg.norm((np.transpose(result_tf.numpy(), [0, 1, 3, 2])
                                - sci_res).flatten())

        print('machine precision:', np.finfo(result_tf.numpy().dtype).eps)
        print('error_tf_scipy', error)
        error2 = np.linalg.norm((np.transpose(result_np, [0, 1, 3, 2])
                                 - sci_res).flatten())
        print('error_np_scipy', error2)
        error3 = np.linalg.norm((result_tf.numpy() - result_np).flatten())
        print('error_np_tf', error3)
        # TODO: why is the error still 9*eps?

        # test the istft.
        # scaled = istft(tf.transpose(tf.constant(sci_res), [0, 1, 3, 2]),
        #                window=window,
        #                nperseg=window_size,
        #                noverlap=overlap,
        #                debug=True)
        # debug_here()
        scaled = istft(result_tf,
                       window,
                       nperseg=window_size,
                       noverlap=overlap,
                       debug=True)
        debug_here()

        _, scisig_np = scisig.istft(sci_res, window='hann',
                                    nperseg=window_size,
                                    noverlap=overlap)
        error4 = np.linalg.norm((scaled.numpy() - scisig_np).flatten())
        print('istft error tf', error4)

        plt.plot(scaled.numpy()[0, 0, :])
        plt.plot(scisig_np[0, 0, :])
        plt.show()

    if 0:
        # test the 3d version.
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot(states.numpy()[0, :, 0],
        #         states.numpy()[0, :, 1],
        #         states.numpy()[0, :, 2])
        # plt.show()

        tmp_last_states = tf.transpose(states, [0, 2, 1])
        result_tf, result_np = stft(tmp_last_states, window, window_size, overlap,
                                    debug=True)
        tmp_f, tmp_t, sci_res = scisig.stft(tmp_last_states.numpy(),
                                            window,
                                            nperseg=window_size,
                                            noverlap=overlap,
                                            axis=-1)
        error = np.linalg.norm((np.transpose(result_tf.numpy(), [0, 1, 3, 2])
                                - sci_res).flatten())
        print('machine precision:', np.finfo(result_tf.numpy().dtype).eps)
        print('error_tf_scipy', error)
        error2 = np.linalg.norm((np.transpose(result_np, [0, 1, 3, 2])
                                 - sci_res).flatten())
        print('error_np_scipy', error2)
        error3 = np.linalg.norm((result_tf.numpy() - result_np).flatten())
        print('error_np_tf', error3)
        # TODO: why is the error still 9*eps?

        # test the istft.
        # scaled = istft(tf.transpose(tf.constant(sci_res), [0, 1, 3, 2]),
        #                window=window,
        #                nperseg=window_size,
        #                noverlap=overlap,
        #                debug=True)
        # debug_here()
        scaled = istft(result_tf,
                       window,
                       nperseg=window_size,
                       noverlap=overlap,
                       debug=True)  # epsilon=1e-3)

        _, scisig_np = scisig.istft(sci_res, window,
                                    nperseg=window_size,
                                    noverlap=overlap)
        error4 = np.linalg.norm((scaled.numpy() - scisig_np).flatten())
        print('istft error tf', error4)

        ax.plot(scisig_np[0, 0, :],
                scisig_np[0, 1, :],
                scisig_np[0, 2, :])
        ax.plot(scaled.numpy()[0, 0, :],
                scaled.numpy()[0, 1, :],
                scaled.numpy()[0, 2, :])
        plt.show()

    if 0:
        # import matplotlib2tikz as tikz

        # autocorrelation analysis.
        plt.plot(spikes.numpy()[0, :, :])
        plt.show()
        # tikz.save('spikes.tkz')

        l, c, line, b = plt.acorr(spikes.numpy()[0, :, 0], maxlags=256)
        plt.clf()
        plt.close()

        x = np.concatenate([-np.flip(np.arange(int(len(c)/2)), 0), [1],
                            np.arange(int(len(c)/2))], 0)
        plt.plot(x, c)
        plt.xlabel('shift')
        plt.ylabel('autocorrelation')
        plt.show()
        # tikz.save('spikes_autocorr.tex')