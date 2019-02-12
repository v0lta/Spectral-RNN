import numpy as np


def compute_cgRNN_cell_params(state_size, input_size, output_size):
    # two gates + cell equation times two because complex
    print('Wh', state_size * state_size * 2)
    print('Wi', state_size * input_size * 2)
    print('b', state_size * 2)

    recurrent_matrices = state_size * state_size * 3 * 2
    input_matrices = state_size * input_size * 3 * 2
    bias = state_size * 3 * 2
    out = state_size * output_size * 2 + output_size*2
    total = recurrent_matrices + input_matrices + bias + out
    return total


def compute_state_size(parameters, input_size, output_size):
    # 0 = 6s^2 + 6si + 6s + 2so + 2o - p
    # 0 = 6s^2 + s (6i + 2o + 6) + 2o - p

    # a = 6.0
    # b = 6.0*input_size + 2.0*output_size + 6.0
    # c = 2.0*output_size - parameters

    # s1 = (- b + np.sqrt(b*b - 4.0*a*c)) / 2.0*a
    # s2 = (- b - np.sqrt(b*b - 4.0*a*c)) / 2.0*a

    p = (6.0*input_size + 2.0*output_size + 6.0) / 6.0
    q = (2.0*output_size - parameters) / 6.0
    x1 = -p/2.0 + np.sqrt((p/2.0)*(p/2.0) - q)
    x2 = -p/2.0 - np.sqrt((p/2.0)*(p/2.0) - q)

    return x1, x2
