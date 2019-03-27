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

    # train this.
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
            datenc_np, encout_np, datdec_np, decout_np, \
            datand_np = \
            sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                      pgraph.data_encoder_gt, pgraph.encoder_out,
                      pgraph.data_decoder_gt, pgraph.decoder_out, pgraph.data_nd])
      else:
        np_loss, summary_to_file, np_global_step, \
            datenc_np, encout_np, datdec_np, decout_np, \
            datand_np, window_np = \
            sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                      pgraph.data_encoder_gt, pgraph.encoder_out,
                      pgraph.data_decoder_gt, pgraph.decoder_out, pgraph.data_nd,
                      pgraph.window])

    plt.plot(decout_np[0, :, 0], label=label)
    if gt:
        plt.plot(datdec_np[0, :, 0], label='ground-truth')
    
restore_step = 8000
path2 = '/home/moritz/infcuda/fft_pred_networks/logs/\
mackey2/2019-03-11 13:45:41_gru_size_156_fft\
_True_bs_100_ps_512_dis_0_lr_0.004_dr_0.95_\
ds_390_sp_1.0_rc_True_pt_154727_wf_learned_\
plank_ws_128_ol_64_ffts_64_fftp_9_fl_None_eps_0.01'
plot(path2, restore_step, label='plank-gru')

path3 = '/home/moritz/infcuda/fft_pred_networks/logs/mackey2/\
2019-03-11 13:16:45_gru_size_188_fft_False_bs_100_\
ps_512_dis_0_lr_0.004_dr_0.95_ds_390_sp_1.0_rc_True\
_pt_154788_linre'
plot(path3, restore_step, label='reshape-gru')

path = '/home/moritz/infcuda/fft_pred_networks/logs/mackey2\
/2019-03-11 13:22:06_gru_size_226_fft_False_bs_100_ps_512\
_dis_0_lr_0.004_dr_0.95_ds_390_sp_1.0_rc_True_pt_154811'

plot(path, restore_step, label='time-gru', gt=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
# tikz.save('mackey_fit.tex')
