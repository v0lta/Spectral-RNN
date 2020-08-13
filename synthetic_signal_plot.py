import io
import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mackey_glass_generator import MackeyGenerator
from power_experiments.prediction_graph import FFTpredictionGraph
import tikzplotlib as tikz


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


restore_step = 30000

# prediction quality plot.
pred_qual = True
if pred_qual:
    print('mse plot')
    path3 = '/home/moritz/uni/fourier-prediction/log/cvpr_workshop_synthetic_5/\
2020-03-12 09:59:31_gru_size_64_fft_True_bs_12_ps_2560_dis_0_lr_0.001_dr_0.9\
_ds_1000_sp_1.0_rc_True_pt_13509_wf_learned_gaussian_ws_128_ol_64_ffts_64_fftp\
_41_fl_None_eps_0.001_fftcr_32'
    plot(path3, restore_step, label='gru-64-fft-lowpass')

    path1 = '/home/moritz/uni/fourier-prediction/log/cvpr_workshop_synthetic_5/\
2020-03-12 10:57:54_gru_size_64_fft_False_bs_12_ps_2560_dis_0_lr_0.001_dr_0.9\
_ds_1000_sp_1.0_rc_True_pt_12994_downs_32_linre'
    plot(path1, restore_step, label='gru-64-window-downsampled')

    path = '/home/moritz/uni/fourier-prediction/log/cvpr_workshop_synthetic_5/\
2020-03-11 17:05:11_gru_size_64_fft_False_bs_12_ps_2560_dis_0\
_lr_0.001_dr_0.9_ds_1000_sp_1.0_rc_True_pt_12737'
    plot(path, restore_step, label='time-gru', gt=True)
    plt.legend()
    # plt.show()
    tikz.save('mackey_fit_full.tex', standalone=True)
    # plt.savefig('mackey_fit.pdf')
    print('done')

