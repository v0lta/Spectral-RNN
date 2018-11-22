import pickle
from lorenz_exp import run_experiment


spikes_instead_of_states = True
base_dir = 'logs/suppl/'
if spikes_instead_of_states:
    dimensions = 1
else:
    dimensions = 3
cell_type = 'cgRNN'
num_units = 260
if cell_type == 'uRNN':
    circ_h = 4
    conv_h = 5
    input_samples = 17
    num_units = 150
if cell_type == 'orthogonal':
    num_units = 50
sample_prob = 1.0
pred_samples = 256
num_proj = dimensions
learning_rate = 0.001
iterations = 20001
GPUs = [5]
batch_size = 250
use_residuals = False
decay_rate = 0.9
decay_steps = 10000

# data parameters
tmax = 10.24
# tmax = 10.23
delta_t = 0.01
steps = int(tmax/delta_t)+1

# fft parameters
fft = True
if fft:
    window_function = 'hann'
    window_size = 32
    overlap = int(window_size*0.75)
    step_size = window_size - overlap
    fft_pred_samples = pred_samples // step_size + 1
    num_proj = int(window_size//2 + 1)*dimensions  # the frequencies
    freq_loss = 'mse_log_mse_dlambda'  # 'mse', 'mse_time', 'ad', 'ad_time', 'ad_norm', log_ad
    num_units = 250

    if (freq_loss == 'ad_time') or (freq_loss == 'log_mse_time') \
       or (freq_loss == 'log_mse_mse_time') or (freq_loss == 'mse_log_mse_dlambda') \
       or (freq_loss == 'mse_time') or (freq_loss is None):
        # epsilon = 1e-2
        epsilon = 1e-3
        print('epsilon', epsilon)
    else:
        epsilon = None
else:
    window_function = None
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
                   delta_t, steps, fft, window_function, window_size, overlap,
                   step_size, fft_pred_samples, freq_loss, use_residuals,
                   epsilon=epsilon)

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
                       tmp_steps, tmp_fft, window_function, window_size, overlap,
                       step_size, fft_pred_samples, freq_loss, use_residuals, epsilon)

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
                       tmp_steps, tmp_fft, window_function, window_size, overlap,
                       step_size, fft_pred_samples, freq_loss, use_residuals, epsilon)


if 0:
    for length_factor in [1, 2, 3, 4, 5, 6]:
        tmp_fft = True
        tmp_num_units = 250
        tmp_pred_samples = pred_samples
        tmp_tmax = tmax
        if length_factor == 0:
            tmp_window_size = 8
        else:
            tmp_window_size = 16*length_factor
        overlap = int(tmp_window_size*0.75)
        tmp_step_size = tmp_window_size - overlap
        tmp_num_proj = int(tmp_window_size//2 + 1)*dimensions
        tmp_steps = int(tmp_tmax/delta_t) + 1
        run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                       tmp_num_units, sample_prob, tmp_pred_samples, tmp_num_proj,
                       learning_rate, decay_rate, decay_steps,
                       iterations, GPUs, batch_size, tmp_tmax, delta_t,
                       tmp_steps, tmp_fft, window_function, tmp_window_size, overlap,
                       tmp_step_size, fft_pred_samples, freq_loss, use_residuals, epsilon)
