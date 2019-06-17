import io
import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mackey_glass_generator import MackeyGenerator
from power_experiments.prediction_graph import FFTpredictionGraph
import matplotlib2tikz as tikz


def plot(path, restore_step, label, gt=False):
    pd = pickle.load(open(path + '/param.pkl', 'rb'))
    mackeygen = MackeyGenerator(pd['batch_size'],
                                pd['tmax'], pd['delta_t'],
                                restore_and_plot=True)
    pgraph = FFTpredictionGraph(pd, mackeygen)

    # plot this.
    gpu_options = tf.GPUOptions(visible_device_list=str(pd['GPUs'])[1:-1])
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=gpu_options)
    with tf.Session(graph=pgraph.graph, config=config) as sess:
      pgraph.saver.restore(sess, save_path=path
                           + '/weights/' + 'cpk' + '-' + str(restore_step))
      if not pd['fft']:
        np_loss, summary_to_file, np_global_step, \
            datenc_np, datdec_np, decout_np, \
            datand_np = \
            sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                      pgraph.data_encoder_time, pgraph.data_decoder_time,
                      pgraph.decoder_out, pgraph.data_nd])
      else:
        np_loss, summary_to_file, np_global_step, \
            datenc_np, datdec_np, decout_np, \
            datand_np, window_np = \
            sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                      pgraph.data_encoder_time, pgraph.data_decoder_time,
                      pgraph.decoder_out, pgraph.data_nd,
                      pgraph.window])

    plt.plot(decout_np[0, :, 0], label=label)
    if gt:
        plt.plot(datdec_np[0, :, 0], label='ground-truth')


restore_step = 20000
path3 = '/home/moritz/infcuda/fft_pred_networks/logs/mackey1k2c8d_v2/\
2019-05-03 18:57:292019-05-03 18:57:29_gru_size_64_fft_False_bs_12_ps\
_2560_dis_0_lr_0.001_dr_0.9_ds_1000_sp_1.0_rc_True_pt_28928_linre'
plot(path3, restore_step, label='reshape-gru')


path4 = '/home/moritz/infcuda/fft_pred_networks/logs/mackey1k2c8d_v2/\
2019-05-06 09:52:252019-05-06 09:52:25_gru_size_64_fft_True\
_bs_12_ps_2560_dis_0_lr_0.001_dr_0.9_ds_1000_sp_1.0_rc_True_pt_45891\
_wf_learned_gaussian_ws_128_ol_64_ffts_64_fftp_41_fl_None_eps_0.001_fftcr_None'
plot(path4, restore_step, label='fft-gru')

path2 = '/home/moritz/infcuda/fft_pred_networks/logs/mackey1k2c8d_v2/\
2019-05-04 14:30:192019-05-04 14:30:19_gru_size_64_fft_True_bs_12\
_ps_2560_dis_0_lr_0.001_dr_0.9_ds_1000_sp_1.0_rc_True_pt_13508_wf\
_hann_ws_128_ol_64_ffts_64_fftp_41_fl_None_eps_0.001_fftcr_32'

# path2 = '/home/moritz/infcuda/fft_pred_networks/logs/mackey1k2c8d_v2/\
# 2019-05-04 11:21:232019-05-04 11:21:23_gru_size_64_\
# fft_True_bs_12_ps_2560_dis_0_lr_0.001_dr_0.9_ds_1000\
# _sp_1.0_rc_True_pt_14536_wf_hann_ws_128_ol_64_ffts_64_fftp_41\
# _fl_None_eps_0.001_fftcr_16'
plot(path2, restore_step, label='fft-gru-lowpass')

path = '/home/moritz/infcuda/fft_pred_networks/logs/mackey1k2c8d_v2/\
2019-05-04 13:17:172019-05-04 13:17:17_gru_size_64_fft_False_bs_12_\
ps_2560_dis_0_lr_0.001_dr_0.9_ds_1000_sp_1.0_rc_True_pt_12737'
plot(path, restore_step, label='time-gru', gt=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
tikz.save('mackey_fit_full.tex')
plt.show()
