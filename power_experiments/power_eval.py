import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib2tikz as tikz
from prediction_graph import FFTpredictionGraph


def get_pred(path, restore_step):
    pd = pickle.load(open(path + '/param.pkl', "rb"))
    pd['fully_fft_comp'] = None

    def organize_into_batches(batches):
        batch_total = len(batches)
        split_into = int(batch_total/pd['batch_size'])
        batch_lst = np.array_split(np.stack(batches),
                                   split_into)
        return batch_lst
    pgraph = FFTpredictionGraph(pd)
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
                np_loss, np_global_step, decout_np, \
                    datand_np, window_np = \
                    sess.run([pgraph.loss, pgraph.global_step,
                              pgraph.decoder_out, pgraph.data_nd, pgraph.window],
                             feed_dict=feed_dict)
            else:
                np_loss, np_global_step, decout_np, \
                    datand_np = \
                    sess.run([pgraph.loss, pgraph.global_step,
                              pgraph.decoder_out, pgraph.data_nd],
                             feed_dict=feed_dict)
            net_pred = decout_np[:, :, 0]*power_handler.std + power_handler.mean
            official_pred = official_pred_np[:, -pd['pred_samples']:, 0]
            gt = gt_np[:, -pd['pred_samples']:, 0]
            mse_lst_net.append(
                np.mean((gt[:, pd['discarded_samples']:]
                         - net_pred[:, pd['discarded_samples']:pd['pred_samples']])**2))
            mse_lst_off.append(
                np.mean((gt[:, pd['discarded_samples']:]
                         - official_pred[:, pd['discarded_samples']:pd['pred_samples']]
                         )**2))
            print(str(no) + ' ', end='')

    return decout_np, official_pred_np, gt_np, pd


# 60 day predictions
if 0:
    restore_step = 26720
    i = 90  # 90
    print(i)

    path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/\
    power_pred_60d_1h/conv_bin_exp/2019-05-06 14:22:25_gru_size_64_fft_\
    False_fm_False_bs_100_ps_1440_dis_0_lr_0.004_dr_0.95_ds_455_sp_1.0\
    _rc_True_pt_27900_linre'
    decout_np, official_pred_np, gt_np, pd = get_pred(path, restore_step)
    net_pred = decout_np[i, :, 0]*pd['power_handler'].std + pd['power_handler'].mean
    official_pred = official_pred_np[i, - pd['pred_samples']:, 0]
    gt = gt_np[i, -pd['pred_samples']:, 0]
    plt.plot(net_pred[pd['discarded_samples']:], label='time-window')

    path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/\
    power_pred_60d_1h/conv_bin_exp/2019-05-07 16:48:41_gru_size_64_fft_False\
    _fm_False_bs_100_ps_1440_dis_0_lr_0.004_dr_0.95_ds_455_sp_1.0_rc_True_pt_12737'
    decout_np, official_pred_np, gt_np, pd = get_pred(path, restore_step)
    net_pred = decout_np[i, :, 0]*pd['power_handler'].std + pd['power_handler'].mean
    official_pred = official_pred_np[i, - pd['pred_samples']:, 0]
    gt = gt_np[i, -pd['pred_samples']:, 0]
    # plt.plot(net_pred[pd['discarded_samples']:], label='time')

    path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/\
    power_pred_60d_1h/conv_bin_exp/\
    2019-05-06 16:58:46_gru_size_64_fft_True_fm_False_bs_\
    100_ps_1440_dis_0_lr_0.004_dr_0.95_ds_455_sp_1.0_rc_True_pt_\
    43835_wf_learned_gaussian_ws_120_ol_60_ffts_60_fftp_25_fl_None_eps_0.01_fftcr_None'
    decout_np, official_pred_np, gt_np, pd = get_pred(path, restore_step)
    net_pred = decout_np[i, :, 0]*pd['power_handler'].std + pd['power_handler'].mean
    official_pred = official_pred_np[i, - pd['pred_samples']:, 0]
    gt = gt_np[i, -pd['pred_samples']:, 0]
    plt.plot(net_pred[pd['discarded_samples']:], label='fft')

    path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/\
    power_pred_60d_1h/conv_bin_exp/\
    2019-05-06 17:25:54_gru_size_64_fft_True_fm_False_bs_100_ps_1440\
    _dis_0_lr_0.004_dr_0.95_ds_455_sp_1.0_rc_True_pt_20191_wf_learned_gaussian\
    _ws_120_ol_60_ffts_60_fftp_25_fl_None_eps_0.01_fftcr_4'
    decout_np, official_pred_np, gt_np, pd = get_pred(path, restore_step)
    net_pred = decout_np[i, :, 0]*pd['power_handler'].std + pd['power_handler'].mean
    official_pred = official_pred_np[i, - pd['pred_samples']:, 0]
    gt = gt_np[i, -pd['pred_samples']:, 0]
    plt.plot(net_pred[pd['discarded_samples']:], label='fft-4')

    plt.plot(gt[pd['discarded_samples']:], label='ground truth')

    plt.legend()
    plt.ylim([6000, 14000])
    # last week
    plt.xlim([1250, 1450])
    tikz.save('comparison_last_week_60d_fit.tex', standalone=True)
    plt.show()

# day ahead prediction.
# 60 day predictions
if 1:
    restore_step = 8880

    # for i in range(0, 90):
    for i in [7]:
        print('------------------------------', i)
        x = np.linspace(0., 24., num=96)

        path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/\
power_pred_1d_15_min/paper_exp2/2019-05-15 18:38:01_gru_size_64_fft_\
False_fm_True_bs_100_ps_144_dis_48_lr_0.004_dr_0.95_ds_390_sp_1.0_rc_True_pt_12737'
        decout_np, official_pred_np, gt_np, pd = get_pred(path, restore_step)
        net_pred = decout_np[i, :, 0]*pd['power_handler'].std + pd['power_handler'].mean
        plt.plot(x, net_pred[pd['discarded_samples']:], label='time')

        path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/\
power_pred_1d_15_min/paper_exp2/2019-05-15 18:38:24_gru_size_64_fft_False\
_fm_True_bs_100_ps_144_dis_48_lr_0.004_dr_0.95_ds_390_sp_1.0_rc_True_pt_24816_linre'
        decout_np, official_pred_np, gt_np, pd = get_pred(path, restore_step)
        net_pred = decout_np[i, :, 0]*pd['power_handler'].std + pd['power_handler'].mean
        gt = gt_np[i, -pd['pred_samples']:, 0]
        plt.plot(x, net_pred[pd['discarded_samples']:], label='time-window')

        path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/\
power_pred_1d_15_min/paper_exp2/2019-05-16 13:10:25_gru_size_64_fft_True_\
fm_True_bs_100_ps_144_dis_48_lr_0.004_dr_0.95_ds_390_sp_1.0_rc_True_pt_37667\
_wf_learned_gaussian_ws_96_ol_48_ffts_48_fftp_4_fl_None_eps_0.01/'
        decout_np, official_pred_np, gt_np, pd = get_pred(path, restore_step)
        net_pred = decout_np[i, :, 0]*pd['power_handler'].std + pd['power_handler'].mean
        plt.plot(x, net_pred[pd['discarded_samples']:], label='fft')

        official_pred = official_pred_np[i, - pd['pred_samples']:, 0]
        gt = gt_np[i, -pd['pred_samples']:, 0]
        plt.plot(x, official_pred[pd['discarded_samples']:], label='entsoe.eu')
        plt.plot(x, gt[pd['discarded_samples']:], label='ground-truth')
        plt.legend()
        plt.ylabel('power-load [MW]')
        plt.xlabel('time [h]')
        tikz.save('day_ahead plot.tex', standalone=True)
        plt.show()
