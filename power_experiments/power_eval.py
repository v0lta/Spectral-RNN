import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from prediction_graph import FFTpredictionGraph


def organize_into_batches(batches):
    batch_total = len(batches)
    split_into = int(batch_total/pd['batch_size'])
    batch_lst = np.array_split(np.stack(batches),
                               split_into)
    return batch_lst


path = '/home/wolter/fft_pred_networks/power_experiments/log/test/\
2019-03-08 15:10:34_gru_size_222_fft_False_fm_False_bs_100\
_ps_720_dis_0_lr_0.004_dr_0.95_ds_455_sp_1.0_rc_True_pt_149407/'
pd = pickle.load(open(path + 'param.pkl', "rb"))
pgraph = FFTpredictionGraph(pd)
restore_step = 416
power_handler = pd['power_handler']

mse_lst_net = []
mse_lst_off = []
gpu_options = tf.GPUOptions(visible_device_list=str(7))
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)
with tf.Session(graph=pgraph.graph, config=config) as sess:
    pgraph.saver.restore(sess, save_path=path
                         + '/weights/' + 'cpk' + '-' + str(restore_step))
    test_data = power_handler.get_test_set()

    test_batch_lst = organize_into_batches(test_data)
    for test_batch in test_batch_lst:
        gt = np.reshape(test_batch[:pd['batch_size'], :, :, 1],
                        [pd['batch_size'], pd['context_days']*pd['samples_per_day'],
                         1])
        official_pred = np.reshape(test_batch[:pd['batch_size'], :, :, 0],
                                   [pd['batch_size'],
                                    pd['context_days']*pd['samples_per_day'], 1])
        feed_dict = {pgraph.data_nd: gt}
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
        official_pred = official_pred[0, -pd['pred_samples']:, 0]
        gt = gt[0, -pd['pred_samples']:, 0]
        mse_lst_net.append(
            np.mean((gt[pd['discarded_samples']:]
                     - net_pred[pd['discarded_samples']:pd['pred_samples']])**2))
        mse_lst_off.append(
            np.mean((gt[pd['discarded_samples']:]
                     - official_pred[pd['discarded_samples']:pd['pred_samples']])**2))
        print('.', end='')

plt.plot(net_pred)
plt.show()
