import numpy as np
import tensorflow as tf
import tensorflow.contrib.signal as tfsignal
import scipy.signal as scisig
import matplotlib.pyplot as plt
from lorenz_data_generator import LorenzGenerator
from mackey_glass_generator import MackeyGenerator
from IPython.core.debugger import Tracer
debug_here = Tracer()


def dmd(self, window):
    pass