import pickle
from lorenz_exp import run_experiment

#restore_path = '/home/moritz/infcuda/fft_pred_networks/logs/tf_1d_paper/2018-11-14 15:52:44__iw_cGRU__act_mod_relu_units_250_stfl_True_ga_mod_sigmoid_fft_True_bs_250_ps_256_lr_0.001_dr_0.9_ds_10000_sp_1.0_rc_False_pt_410537_wf_hann_ws_32_ol_24_ffts_8_fftp_33_fl_log_mse_time_eps_0.001_1d'
restore_path = '/home/moritz/infcuda/fft_pred_networks/logs/tf_1d_paper/2018-11-14 15:54:31__res_cell__iw_cGRU__act_mod_relu_units_250_stfl_True_ga_mod_sigmoid_fft_True_bs_250_ps_256_lr_0.001_dr_0.9_ds_10000_sp_1.0_rc_True_pt_410537_wf_hann_ws_32_ol_24_ffts_8_fftp_33_fl_log_mse_time_eps_0.001_1d'

spikes_instead_of_states, base_dir, dimensions, cell_type, \
    num_units, sample_prob, pred_samples, num_proj, \
    init_learning_rate, decay_rate, decay_steps, iterations, \
    GPUs, batch_size, tmax, delta_t, steps, fft, \
    window_function, window_size, overlap, \
    step_size, fft_pred_samples, freq_loss, \
    use_residuals, epsilon, _, _, _ = pickle.load(open(restore_path + '/param.pkl', 'rb'))

restore_step = 15001
restore_and_plot = True


if 0:
    run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                   num_units, sample_prob, pred_samples, num_proj,
                   init_learning_rate, decay_rate, decay_steps, iterations,
                   GPUs, batch_size, tmax, delta_t, steps, fft,
                   window_function, window_size, overlap,
                   step_size, fft_pred_samples, freq_loss,
                   use_residuals, epsilon, restore_and_plot,
                   restore_path, restore_step, '15krcNofft.pdf')

# extract a logfile plot.
