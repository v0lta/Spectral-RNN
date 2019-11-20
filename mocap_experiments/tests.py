import io
import time
import pickle
import tensorflow as tf
import numpy as np
import collections
import matplotlib.pyplot as plt
from mocap_experiments.load_h36m import H36MDataSet
from mocap_experiments.prediction_graph import FFTpredictionGraph
from mocap_experiments.write_movie import write_movie, write_figure, write_subplots
from mocap_experiments.util import compute_ent_metrics, organize_into_batches, compute_ent_metrics_splits

PoseData = collections.namedtuple('PoseData', ['f', 'action', 'actor', 'array'])


# experiments_folder = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/mocap/test/'
# experiment_directory = '2019-11-07 17:26:12_gru_size_2048_fft_True_bs_100_ps_64_dis_0_lr_0.001_dr' \
#                       '_0.96_ds_1000_sp_1.0_rc_False_pt_15096114_clw_0.001_csp_64_wf_hann_ws_64_ol_' \
#                       '32_ffts_32_fftp_3_fl_None_eps_0.01_fftcr_10/'
# path = experiments_folder + experiment_directory
# project_folder = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/mocap/test/'
# folder = '2019-11-07 23:39:13_gru_size_4096_fft_True_bs_50_ps_64_dis_0_lr_0.001_dr_0.98_ds' \
#          '_1000_sp_1.0_rc_False_pt_52015206_clw_0_csp_64_wf_hann_ws_32_ol_28_ffts_8_fftp_9_' \
#          'fl_None_eps_0.01_fftcr_16/'
# path = project_folder + folder
# checkpoint_folder = 'soa_kl1_kl2_0.010279945518383045_0.010479976860678938'

# paper figure experiment...
base_path = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/paper3/'
folder = '2019-11-13 16:31:17_gru_size_5120_fft_True_bs_50_ps_48_dis_0_lr_0.001_dr_0.98_ds_1000_sp_1.0_mses_48' \
        '_rc_True_pt_87014808_clw_0.001_csp_48_wf_hann_ws_16_ol_9_ffts_7_fftp_7_fl_None_eps_0.01' \
        '_fftcr_2/'
checkpoint_folder = 'mse_3865.5574'
path = base_path + folder

# fft 2.5 experiment.
# base_path = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/paper5/'
# folder = '2019-11-13 21:12:06_gru_size_3072_fft_True_bs_50_ps_50_dis_0_lr_0.001_dr_0.98_ds_1000_sp_1.0_' \
#          'mses_50_rc_False_pt_30827724_clw_0.001_csp_50_wf_hann_ws_20_ol_16_ffts_4_fftp_13_fl_None_eps_0.01_fftcr_5/'
# path = base_path + folder
# checkpoint_folder = 'mse_4823.359'

# paper 5 experiments
# base_path = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/paper5/'
# folder = '2019-11-13 21:12:06_gru_size_3072_fft_True_bs_50_ps_50_dis_0_lr_0.001_dr_0.98_ds_1000' \
#          '_sp_1.0_mses_50_rc_False_pt_30827724_clw_0.001_csp_50_wf_hann_ws_20_ol_16_ffts_4_fftp_13' \
#          '_fl_None_eps_0.01_fftcr_5/'
# folder = '2019-11-13 21:51:29_gru_size_3072_fft_True_bs_50_ps_50_dis_0_lr_0.001_dr_0.98_ds_1000_sp' \
#          '_1.0_mses_50_rc_False_pt_30827724_clw_0.001_csp_50_wf_hann_ws_20_ol_16_ffts_4_fftp_13_fl' \
#          '_None_eps_0.01_fftcr_5/'
# folder = '2019-11-13 22:23:33_gru_size_3072_fft_True_bs_50_ps_50_dis_0_lr_0.001_dr_0.98_ds_1000_sp' \
#          '_1.0_mses_50_rc_False_pt_30827724_clw_0.001_csp_50_wf_hann_ws_20_ol_16_ffts_4_fftp_13_fl_' \
#          'None_eps_0.01_fftcr_4/'
# folder = '2019-11-14 00:36:16_gru_size_3072_fft_True_bs_50_ps_50_dis_0_lr_0.001_dr_0.98_ds_1000_sp' \
#          '_1.0_mses_50_rc_False_pt_30827724_clw_0.0_csp_50_wf_hann_ws_20_ol_16_ffts_4_fftp_13_fl_' \
#          'None_eps_0.01_fftcr_4/'
# folder = '2019-11-13 22:55:23_gru_size_3072_fft_False_bs_50_ps_50_dis_0_lr_0.001_dr_0.98_ds_1000_sp' \
#          '_1.0_mses_50_rc_False_pt_28947507_clw_0.001_csp_50/'
# folder = '2019-11-14 01:08:12_gru_size_3072_fft_False_bs_50_ps_50_dis_0_lr_0.001_dr_0.98_ds_1000_sp_' \
#          '1.0_mses_50_rc_False_pt_28947507_clw_0.0_csp_50/'

# paper 5 infcuda experiments
# base_path = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/paper5_infcuda/'
# folder = '2019-11-13 22:05:51_gru_size_4096_fft_True_bs_50_ps_50_dis_0_lr_0.001_dr_0.98' \
#         '_ds_1000_sp_1.0_mses_50_rc_False_pt_53686476_clw_0.001_csp_50_wf_hann_ws_20_ol_' \
#         '16_ffts_4_fftp_13_fl_None_eps_0.01_fftcr_5/'
#folder = '2019-11-14 01:16:49_gru_size_4096_fft_False_bs_50_ps_50_dis_0_lr' \
#         '_0.001_dr_0.98_ds_1000_sp_1.0_mses_50_rc_False_pt_51179571_clw_0.001_csp_50/'

# table experiment
# base_path = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/paper_archive/'
# folder = '2019-11-09 12:33:03_gru_size_4096_fft_True_bs_50_ps_224_dis_0_lr_0.001_dr_0.98_ds_1000_sp' \
#          '_1.0_mses_64_rc_False_pt_52015206_clw_0.001_csp_224_wf_hann_ws_64_ol_57_ffts_7_fftp_33_fl_' \
#          'None_eps_0.01_fftcr_32/'


# base_path = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/paper6/'
# folder = '2019-11-15 20:32:47_gru_size_3072_fft_True_bs_50_ps_200_dis_0_lr_0.001_dr_0.97_ds_1000' \
#          '_sp_1.0_mses_50_rc_False_pt_32081203_clw_0.005_csp_200_wf_learned_gaussian_ws_20_ol_16_' \
#          'ffts_4_fftp_51_fl_None_eps_0.01_fftcr_3/'
# checkpoint_folder = 'weights'
#
# path = base_path + folder
#
pd = pickle.load(open(path + 'param.pkl', 'rb'))
#
# pd['chunk_size'] = 250
# pd['pred_samples'] = 200
# pd['mse_samples'] = 200
# pd['input_samples'] = pd['chunk_size']
mocap_handler_test = H36MDataSet(train=False, chunk_size=pd['chunk_size'], dataset_name='h36m')

graph = FFTpredictionGraph(pd)

gpu_options = tf.GPUOptions(visible_device_list=str(pd['GPUs'])[1:-1])
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)
with tf.Session(graph=graph.graph, config=config) as sess:
    test_data = mocap_handler_test.get_batches()
    print('restore weights:.....')
    graph.saver.restore(sess, save_path=path + checkpoint_folder + '/cpk')
    test_mse_lst_net = []
    test_net_lst_out = []
    test_gt_lst_out = []
    test_batch_lst = organize_into_batches(test_data, pd)
    for test_batch in test_batch_lst:
        gt = np.reshape(test_batch, [pd['batch_size'], pd['chunk_size'], 17*3])

        feed_dict = {graph.data_nd: gt}
        if pd['fft']:
            np_loss, np_global_step, \
                test_datenc_np, test_datdec_np, test_decout_np, \
                datand_np, window_np = \
                sess.run([graph.loss, graph.global_step,
                          graph.data_encoder_time,
                          graph.data_decoder_time,
                          graph.decoder_out, graph.data_nd, graph.window],
                         feed_dict=feed_dict)
        else:
            np_loss, np_global_step, \
                test_datenc_np, test_datdec_np, test_decout_np, \
                datand_np = \
                sess.run([graph.loss, graph.global_step,
                          graph.data_encoder_time,
                          graph.data_decoder_time,
                          graph.decoder_out, graph.data_nd],
                         feed_dict=feed_dict)
        net_pred = test_decout_np[:, :, 0]*pd['mocap_handler'].std + pd['mocap_handler'].mean
        gt = gt[:, -pd['pred_samples']:, 0]
        test_mse_lst_net.append(
            np.mean((gt[:, pd['discarded_samples']:]
                     - net_pred[:, pd['discarded_samples']:pd['pred_samples']])
                    ** 2))
        test_net_lst_out.append(test_decout_np)
        test_gt_lst_out.append(test_datdec_np)

        print('.', end='')
    mse_net = np.mean(np.array(test_mse_lst_net))

    net_out = np.concatenate(test_net_lst_out, axis=0)
    gt_out = np.concatenate(test_gt_lst_out, axis=0)
    net_out = np.reshape(net_out, [net_out.shape[0], net_out.shape[1], 17, 3])
    gt_out = np.reshape(gt_out, [gt_out.shape[0], gt_out.shape[1], 17, 3])
    net_out = net_out[:, :pd['pred_samples'], :, :]
    gt_out = gt_out[:, :pd['pred_samples'], :, :]
    euler_ent, euler_kl1, euler_kl2 = compute_ent_metrics(gt_seqs=np.moveaxis(gt_out, [0, 1, 2, 3], [0, 2, 1, 3]),
                                                          seqs=np.moveaxis(net_out, [0, 1, 2, 3], [0, 2, 1, 3]))
    print('euler entropy', euler_ent, 'kl1', euler_kl1, 'kl2', euler_kl2)
    ent, kl1, kl2 = compute_ent_metrics(gt_seqs=np.moveaxis(gt_out, [0, 1, 2, 3], [0, 2, 1, 3]),
                                        seqs=np.moveaxis(net_out, [0, 1, 2, 3], [0, 2, 1, 3]), euler=False)
    print('carthesian entropy', ent, 'carthesian kl1', kl1, 'carthesian kl2', kl2)

    # five_hz = False
    # if five_hz:
    #     gt_out_4s = gt_out[:, :200:10, :, :]
    #     net_out_4s = net_out[:, :200:10, :, :]
    #     _ = compute_ent_metrics_splits(np.moveaxis(gt_out_4s, [0, 1, 2, 3], [0, 2, 1, 3]),
    #                                    np.moveaxis(net_out_4s, [0, 1, 2, 3], [0, 2, 1, 3]), seq_len=20,
    #                                    print_numbers=True)

    test_datenc_np = np.reshape(test_datenc_np, [test_datenc_np.shape[0], test_datenc_np.shape[1], 17, 3])
    test_datdec_np = np.reshape(test_datdec_np, [test_datdec_np.shape[0], test_datdec_np.shape[1], 17, 3])
    test_decout_np = np.reshape(test_decout_np, [test_decout_np.shape[0], test_decout_np.shape[1], 17, 3])
    sel = 8  #11  # 30
    gt_movie = np.concatenate([test_datenc_np, test_datdec_np], axis=1)
    net_movie = np.concatenate([test_datenc_np, test_decout_np], axis=1)
    # write_movie(np.transpose(gt_movie[sel], [1, 2, 0]), r_base=2,
    #             name='test_in.mp4', color_shift_at=pd['chunk_size'] - pd['pred_samples'])
    # write_movie(np.transpose(net_movie[sel], [1, 2, 0]), r_base=1,
    #             name='test_out.mp4', color_shift_at=pd['chunk_size'] - pd['pred_samples'])
    write_figure(np.transpose(net_movie[sel][::12, :, :], [1, 2, 0]),
                              color_shift_at=int(pd['chunk_size'] - pd['pred_samples'])/12, r_base=1,
                 name='test_figure.pdf')
    # write_subplots(np.transpose(net_movie[sel][::12, :, :], [1, 2, 0]),
    #                             color_shift_at=int(pd['chunk_size'] - pd['pred_samples']/12), r_base=1,
    #              name='test_figure.pdf')
    # for i in range(50):
    #     write_movie(np.transpose(gt_movie[i], [1, 2, 0]), r_base=1.5,  name='test_in_vid_' + str(i) + '.mp4',
    #                 color_shift_at=pd['chunk_size'] - pd['pred_samples'])