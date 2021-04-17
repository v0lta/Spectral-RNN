import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib2tikz as tikz
from prediction_graph import FFTpredictionGraph


def organize_into_batches(batches):
    batch_total = len(batches)
    split_into = int(batch_total/pd['batch_size'])
    batch_lst = np.array_split(np.stack(batches),
                               split_into)
    return batch_lst


# day-ahead.
if 1:
    path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/\
power_pred_1d_15_min/2019-03-09 14:14:52_cgRNN_size_128_fft_True_fm_\
True_bs_100_ps_144_dis_48_lr_0.004_dr_0.95_ds_390_sp_1.0_rc_True_pt_\
149350_wf_learned_tukey_ws_96_ol_72_ffts_24_fftp_7_fl_None_eps_0.01'
    pd = pickle.load(open(path + '/param.pkl', "rb"))
    pgraph = FFTpredictionGraph(pd)
    restore_step = 8880
    power_handler = pd['power_handler']

    mse_lst_net = []
    mse_lst_off = []
    gpu_options = tf.GPUOptions(visible_device_list=str(0))
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=gpu_options)
    with tf.Session(graph=pgraph.graph, config=config) as sess:
        pgraph.saver.restore(sess, save_path=path
                             + '/weights/' + 'cpk' + '-' + str(restore_step))
        test_data = power_handler.get_test_set()

        test_batch_lst = organize_into_batches(test_data)
        for no, test_batch in enumerate(test_batch_lst):
            gt_np = np.reshape(test_batch[:pd['batch_size'], :, :, 1],
                               [pd['batch_size'],
                                pd['context_days']*pd['samples_per_day'],
                                1])
            official_pred_np = np.reshape(test_batch[:pd['batch_size'], :, :, 0],
                                          [pd['batch_size'],
                                           pd['context_days']*pd['samples_per_day'], 1])
            feed_dict = {pgraph.data_nd: gt_np}
            if pd['fft']:
                np_loss, np_global_step, \
                    datenc_np, encout_np, datdec_np, decout_np, \
                    datand_np, window_np = \
                    sess.run([pgraph.loss, pgraph.global_step, pgraph.data_encoder_gt,
                              pgraph.encoder_out, pgraph.data_decoder_gt,
                              pgraph.decoder_out, pgraph.data_nd, pgraph.window],
                             feed_dict=feed_dict)
            else:
                np_loss, np_global_step, \
                    datenc_np, encout_np, datdec_np, decout_np, \
                    datand_np = \
                    sess.run([pgraph.loss, pgraph.global_step, pgraph.data_encoder_gt,
                              pgraph.encoder_out, pgraph.data_decoder_gt,
                              pgraph.decoder_out, pgraph.data_nd],
                             feed_dict=feed_dict)
            net_pred = decout_np[0, :, 0]*power_handler.std + power_handler.mean
            official_pred = official_pred_np[0, -pd['pred_samples']:, 0]
            gt = gt_np[0, -pd['pred_samples']:, 0]
            mse_lst_net.append(
                np.mean((gt[pd['discarded_samples']:]
                         - net_pred[pd['discarded_samples']:pd['pred_samples']])**2))
            mse_lst_off.append(
                np.mean((gt[pd['discarded_samples']:]
                         - official_pred[pd['discarded_samples']:pd['pred_samples']])**2))
            print(str(no) + ' ', end='')

        plt.plot(gt[pd['discarded_samples']:], label='ground truth')
        plt.plot(net_pred[pd['discarded_samples']:], label='fft_cgRNN')
        plt.plot(official_pred[pd['discarded_samples']:], label='entsoe.eu')
        plt.legend()
        plt.show()

i = 19
print(i)
net_pred = decout_np[i, :, 0]*power_handler.std + power_handler.mean
official_pred = official_pred_np[i, -pd['pred_samples']:, 0]
gt = gt_np[i, -pd['pred_samples']:, 0]
plt.plot(gt[pd['discarded_samples']:], label='ground truth')
plt.plot(net_pred[pd['discarded_samples']:], label='fft_cgRNN')
plt.plot(official_pred[pd['discarded_samples']:], label='entsoe.eu')
plt.legend()
tikz.save('gt_fft_entsoe.tex')
plt.show()

i = 20
print(i)
net_pred = decout_np[i, :, 0]*power_handler.std + power_handler.mean
official_pred = official_pred_np[i, -pd['pred_samples']:, 0]
gt = gt_np[i, -pd['pred_samples']:, 0]
plt.plot(gt[pd['discarded_samples']:], label='ground truth')
plt.plot(net_pred[pd['discarded_samples']:], label='fft_cgRNN')
plt.plot(official_pred[pd['discarded_samples']:], label='entsoe.eu')
plt.legend()
tikz.save('gt_fft_entsoe2.tex')
plt.show()
