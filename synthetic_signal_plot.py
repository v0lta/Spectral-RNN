import pickle
import numpy as np
import scipy.signal as scisig
import matplotlib.pyplot as plt
from tensorboard_plot_helper_module import plot_logs, return_logs
from lorenz_exp import run_experiment


if 1:
    restore_path1 = '/home/moritz/infcuda/fft_pred_networks/logs/tf_1d_paper/2018-11-14 15:52:44__iw_cGRU__act_mod_relu_units_250_stfl_True_ga_mod_sigmoid_fft_True_bs_250_ps_256_lr_0.001_dr_0.9_ds_10000_sp_1.0_rc_False_pt_410537_wf_hann_ws_32_ol_24_ffts_8_fftp_33_fl_log_mse_time_eps_0.001_1d/'
    spikes_instead_of_states, base_dir, dimensions, cell_type, \
        num_units, sample_prob, pred_samples, num_proj, \
        init_learning_rate, decay_rate, decay_steps, iterations, \
        GPUs, batch_size, tmax, delta_t, steps, fft, \
        window_function, window_size, overlap, \
        step_size, fft_pred_samples, freq_loss, \
        use_residuals, epsilon, _, _, _ = pickle.load(open(restore_path1
                                                           + '/param.pkl', 'rb'))

    restore_step = 15001
    restore_and_plot = True

    decoder_out_np1, data_decoder_np1 = \
        run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                       num_units, sample_prob, pred_samples, num_proj,
                       init_learning_rate, decay_rate, decay_steps, iterations,
                       GPUs, batch_size, tmax, delta_t, steps, fft,
                       window_function, window_size, overlap,
                       step_size, fft_pred_samples, freq_loss,
                       use_residuals, epsilon, restore_and_plot,
                       restore_path1, restore_step, '15krcNofft1.pdf',
                       return_data=True)

    restore_path2 = '/home/moritz/infcuda/fft_pred_networks/logs/tf_1d_paper/2018-11-14 15:52:35__iw_cGRU__act_mod_relu_units_260_stfl_True_ga_mod_sigmoid_fft_False_bs_250_ps_256_lr_0.001_dr_0.9_ds_10000_sp_1.0_rc_False_pt_409243_1d/'
    spikes_instead_of_states, base_dir, dimensions, cell_type, \
        num_units, sample_prob, pred_samples, num_proj, \
        init_learning_rate, decay_rate, decay_steps, iterations, \
        GPUs, batch_size, tmax, delta_t, steps, fft, \
        window_function, window_size, overlap, \
        step_size, fft_pred_samples, freq_loss, \
        use_residuals, epsilon, _, _, _ = pickle.load(open(restore_path2
                                                           + '/param.pkl', 'rb'))

    restore_step = 15001
    restore_and_plot = True

    decoder_out_np2, data_decoder_np2 = \
        run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                       num_units, sample_prob, pred_samples, num_proj,
                       init_learning_rate, decay_rate, decay_steps, iterations,
                       GPUs, batch_size, tmax, delta_t, steps, fft,
                       window_function, window_size, overlap,
                       step_size, fft_pred_samples, freq_loss,
                       use_residuals, epsilon, restore_and_plot,
                       restore_path2, restore_step, '15krcNofft2.pdf',
                       return_data=True)

    restore_path3 = '/home/moritz/infcuda/fft_pred_networks/logs/tf_1d_paper/2018-11-16 13:17:58__iw_cGRU__act_mod_relu_units_250_stfl_True_ga_mod_sigmoid_fft_True_bs_250_ps_256_lr_0.001_dr_0.9_ds_10000_sp_1.0_rc_False_pt_410537_wf_hann_ws_32_ol_24_ffts_8_fftp_33_fl_log_mse_time_eps_0.001_1d/'
    spikes_instead_of_states, base_dir, dimensions, cell_type, \
        num_units, sample_prob, pred_samples, num_proj, \
        init_learning_rate, decay_rate, decay_steps, iterations, \
        GPUs, batch_size, tmax, delta_t, steps, fft, \
        window_function, window_size, overlap, \
        step_size, fft_pred_samples, freq_loss, \
        use_residuals, epsilon, _, _, _ = pickle.load(open(restore_path3
                                                           + '/param.pkl', 'rb'))

    restore_step = 15001
    restore_and_plot = True
    GPUs = [0]

    decoder_out_np3, data_decoder_np3 = \
        run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                       num_units, sample_prob, pred_samples, num_proj,
                       init_learning_rate, decay_rate, decay_steps, iterations,
                       GPUs, batch_size, tmax, delta_t, steps, fft,
                       window_function, window_size, overlap,
                       step_size, fft_pred_samples, freq_loss,
                       use_residuals, epsilon, restore_and_plot,
                       restore_path3, restore_step, '15krcNofft3.pdf',
                       return_data=True)

    plt.close()
    plt.figure()
    batch = 89
    plt.plot(decoder_out_np1[batch, :, 0], label='Fourier cgRNN')
    plt.plot(decoder_out_np3[batch, :, 0], label='log Fourier cgRNN')
    plt.plot(decoder_out_np2[batch, :, 0], label='cgRNN')
    plt.plot(data_decoder_np1[batch, :, 0], label='target')
    plt.legend()
    plt.xlabel('sample no.')
    plt.ylabel('sample value')
    plt.savefig('fft_time_preds.pdf')
    plt.close()

if 1:
    window_size = 32
    # extract a logfile plot.
    plot_path = '/home/moritz/infcuda/fft_pred_networks/logs/tf_1d_paper/'
    logs = return_logs(plot_path, window_size, vtag='time_loss')
    plt.plot(logs[1][0][0], logs[1][0][1], label='Fourier cgRNN')
    plt.plot(logs[4][0][0], logs[4][0][1], label='log Fourier cgRNN')
    plt.plot(logs[2][0][0], logs[2][0][1], label='cgRNN')
    plt.ylim([0.0, 0.025])
    plt.ylabel('time-domain mean squared error')
    plt.xlabel('weight-updates')
    plt.legend()
    plt.savefig('freq_time_mse_steps.pdf')
    plt.close()

    plt.plot(logs[1][0][0]/15001.0*133, logs[1][0][1], label='Fourier cgRNN')
    plt.plot(logs[4][0][0]/15001.0*142, logs[4][0][1], label='log Fourier cgRNN')
    plt.plot(logs[2][0][0]/15001.0*691, logs[2][0][1], label='cgRNN')
    plt.ylim([0.0, 0.025])
    plt.ylabel('time-domain mean squared error')
    plt.xlabel('training-time [min]')
    plt.legend()
    plt.savefig('freq_time_mse_training_time.pdf')
    plt.close()

if 0:
    win = scisig.get_window('hann', 256)
    plt.plot(win)
    plt.xlabel('sample no.')
    plt.ylabel('window magnitude')
    plt.savefig('hann_window.pdf')
