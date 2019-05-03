import numpy as np
import tensorflow as tf
import tensorflow.contrib.signal as tfsignal
import scipy.signal as scisig
import matplotlib.pyplot as plt
from lorenz_data_generator import LorenzGenerator
from mackey_glass_generator import MackeyGenerator
from IPython.core.debugger import Pdb
debug_here = Pdb().set_trace


def dmd(self, window):
    pass


if __name__ == "__main__":
    try:
        tf.enable_eager_execution()
    except ValueError:
        print("tensorflow is already in eager mode.")

    if 1:
        # book experiment
        dt = .01
        L = 10
        N = int(L/dt)
        t = np.linspace(0.0, L, N)
        xclean = 14*np.sin(7*2*np.pi*t) + 5*np.sin(13*2*np.pi*t)
        x = xclean + 10*np.random.randn(xclean.shape[0])
        # construct the signal
        plt.plot(x)
        plt.plot(xclean)
        plt.show()

        xhat = np.fft.rfft(x)
        xpower = np.abs(xhat)*2/N
        Fs = 1/dt
        freqs = Fs*np.linspace(0, N/2, int(N/2)+1)/N
        plt.plot(freqs, xpower)
        plt.show()

        # signal shift stacks.
        s = 500
        Xlst = []
        for k in range(0, s):
            Xlst.append(x[k:(-s+k)])
        X = np.stack(Xlst)
        [U, s, Vh] = np.linalg.svd(X[:, 1:-1])

