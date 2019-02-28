import pickle
import numpy as np
from lorenz_exp import run_experiment


spikes_instead_of_states = True
base_dir = 'logs/re_test_250/'
if spikes_instead_of_states:
    dimensions = 1
else:
    dimensions = 3
cell_type = 'gru'
num_units = 156
# num_units = 2048
sample_prob = 1.0
pred_samples = 128
num_proj = dimensions
learning_rate = 0.001
iterations = 15000
GPUs = [3]
batch_size = 250
use_residuals = True
decay_rate = 0.95
decay_steps = 5000
stiefel = True

# data parameters
tmax = 10.24
# tmax = 10.23
delta_t = 0.01
steps = int(tmax/delta_t)+1

# fft parameters
fft = True
if fft:
    window = 'hann'
    window_size = 96
    overlap = int(window_size*0.75)
    step_size = window_size - overlap
    fft_pred_samples = pred_samples // step_size + 1
    num_proj = int(window_size//2 + 1)*dimensions  # the frequencies
    freq_loss = 'complex_abs_time'  # complex_square

    if (freq_loss == 'ad_time') or (freq_loss == 'log_mse_time') \
       or (freq_loss == 'log_mse_mse_time') or (freq_loss == 'mse_log_mse_dlambda') \
       or (freq_loss == 'mse_time') or (freq_loss == 'complex_square_time') \
       or (freq_loss == 'complex_abs_time'):
        epsilon = 1e-2
        # epsilon = 1e-3
        print('epsilon', epsilon)
    else:
        epsilon = None
else:
    window = None
    window_size = None
    overlap = None
    step_size = None
    fft_pred_samples = None
    freq_loss = None
    epsilon = None


# fft = False
# num_proj = dimensions
if 1:
    run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                   num_units, sample_prob, pred_samples, num_proj, learning_rate,
                   decay_rate, decay_steps, iterations, GPUs, batch_size, tmax,
                   delta_t, steps, fft, window, window_size, overlap,
                   step_size, fft_pred_samples, freq_loss, use_residuals,
                   epsilon=epsilon, stiefel=stiefel)

if 0:
    for length_factor in [0, 1, 2]:
        tmp_fft = True
        tmp_num_units = 150
        tmp_pred_samples = (length_factor*64) + pred_samples
        tmp_tmax = tmax
        tmp_steps = int(tmp_tmax/delta_t) + 1
        run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                       tmp_num_units, sample_prob, tmp_pred_samples, num_proj,
                       learning_rate, decay_rate, decay_steps,
                       iterations, GPUs, batch_size, tmp_tmax, delta_t,
                       tmp_steps, tmp_fft, window, window_size, overlap,
                       step_size, fft_pred_samples, freq_loss, use_residuals, epsilon,
                       stiefel=stiefel)

if 0:
    for length_factor in [0, 1, 2]:
        tmp_fft = False
        num_proj = dimensions
        if spikes_instead_of_states:
            tmp_num_units = 160
        else:
            tmp_num_units = 180
        tmp_pred_samples = (length_factor*64) + pred_samples
        tmp_tmax = tmax
        tmp_steps = int(tmp_tmax/delta_t) + 1
        run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                       tmp_num_units, sample_prob, tmp_pred_samples, num_proj,
                       learning_rate, decay_rate, decay_steps,
                       iterations, GPUs, batch_size, tmp_tmax, delta_t,
                       tmp_steps, tmp_fft, window, window_size, overlap,
                       step_size, fft_pred_samples, freq_loss, use_residuals, epsilon,
                       stiefel=stiefel)


if 1:
    # TODO: Ajust for extra window weights.

    parameters = 1000000

    def compute_state_size(parameters, input_size, output_size):
        # 0 = 6s^2 + 6si + 6s + 2so + 2o - p
        # 0 = 6s^2 + s (6i + 2o + 6) + 2o - p

        p = (6.0*input_size + 2.0*output_size + 6.0) / 6.0
        q = (2.0*output_size - parameters) / 6.0
        x1 = -p/2.0 + np.sqrt((p/2.0)*(p/2.0) - q)
        x2 = -p/2.0 - np.sqrt((p/2.0)*(p/2.0) - q)

        return x1, x2

    for length_factor in range(1, 7):
        tmp_fft = True
        tmp_pred_samples = pred_samples
        tmp_tmax = tmax
        tmp_window_size = 40*length_factor
        overlap = int(tmp_window_size*0.75)
        tmp_step_size = tmp_window_size - overlap
        tmp_num_proj = int(tmp_window_size//2 + 1)*dimensions
        tmp_steps = int(tmp_tmax/delta_t) + 1

        io = tmp_window_size/2+1
        tmp_num_units = int(compute_state_size(parameters, io, io)[0])

        run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                       tmp_num_units, sample_prob, tmp_pred_samples, tmp_num_proj,
                       learning_rate, decay_rate, decay_steps,
                       iterations, GPUs, batch_size, tmp_tmax, delta_t,
                       tmp_steps, tmp_fft, window, tmp_window_size, overlap,
                       tmp_step_size, fft_pred_samples, freq_loss, use_residuals, epsilon,
                       stiefel=stiefel)
