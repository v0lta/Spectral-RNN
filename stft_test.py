import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow.contrib.signal as tfsig
import scipy.signal as scisig
from IPython.core.debugger import Tracer
import functools
# tf.enable_eager_execution()


# data parameters
tmax = 100
delta_t = 0.1
steps = tmax/delta_t

# fft parameters
frame_length = 500
frame_step = 100


def generate_data(tmax=20, delta_t=0.01, sigma=10.0,
                  beta=8.0/3.0, rho=28.0, foward_euler=False):
    # multi-dimensional data.
    def lorenz(x, t):
        assert len(x) == 3
        return np.array([sigma*(x[1]-x[0]),
                         x[0]*(rho - x[2]) - x[1],
                         x[0]*x[1] - beta*x[2]])

    state0 = np.array([8.0, 6.0, 30.0])
    state0 += np.random.uniform(-4, 4, [3])
    t = np.arange(0.0, tmax, delta_t)

    if not foward_euler:
        states = odeint(lorenz, state0, t)
    else:
        states = [state0]
        for _ in t:
            states.append(states[-1] + 0.01*lorenz(states[-1], None))
        states = np.array(states)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot(states[:, 0], states[:, 1], states[:, 2], label='lorenz curve')
    # plt.show()

    # single dimensional data.
    spikes = np.square(states[:, 0])
    # normalize
    states = states/np.max(np.abs(states))
    spikes = spikes/np.max(spikes)
    return spikes, states


x_np, x3d_np = generate_data(tmax, delta_t)

plt.plot(x_np)
plt.show()

window = scisig.get_window('hann', frame_length)
f, t, Zxx = scisig.stft(x_np, delta_t, nperseg=frame_length, window=window)
t, x_re = scisig.istft(Zxx, nperseg=frame_length, window=window)
print('numpy reconstruction error:', np.linalg.norm(x_np - x_re))
plt.plot(x_np)
plt.plot(x_re)
plt.show()


plt.plot(np.abs(Zxx))
plt.show()
