import io
import time
import pickle
import tensorflow as tf
import numpy as np
import collections
import matplotlib.pyplot as plt
from mocap_experiments.load_h36m import H36MDataSet
from mocap_experiments.prediction_graph import FFTpredictionGraph
from mocap_experiments.write_movie import write_movie
from mocap_experiments.util import compute_ent_metrics, organize_into_batches, compute_ent_metrics_splits

PoseData = collections.namedtuple('PoseData', ['f', 'action', 'actor', 'array'])


experiments_folder = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/mocap/test/'
experiment_directory = '2019-11-07 17:26:12_gru_size_2048_fft_True_bs_100_ps_64_dis_0_lr_0.001_dr' \
                       '_0.96_ds_1000_sp_1.0_rc_False_pt_15096114_clw_0.001_csp_64_wf_hann_ws_64_ol_' \
                       '32_ffts_32_fftp_3_fl_None_eps_0.01_fftcr_10/'
path = experiments_folder + experiment_directory
project_folder = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/mocap/test/'
folder = '2019-11-07 23:39:13_gru_size_4096_fft_True_bs_50_ps_64_dis_0_lr_0.001_dr_0.98_ds' \
         '_1000_sp_1.0_rc_False_pt_52015206_clw_0_csp_64_wf_hann_ws_32_ol_28_ffts_8_fftp_9_' \
         'fl_None_eps_0.01_fftcr_16/'
path = project_folder + folder
checkpoint_folder = 'soa_kl1_kl2_0.010279945518383045_0.010479976860678938'

pd = pickle.load(open(path + 'param.pkl', 'rb'))

pd['chunk_size'] = 512
pd['pred_samples'] = 256
pd['mse_samples'] = 256
pd['input_samples'] = pd['chunk_size']
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
    ent, kl1, kl2 = compute_ent_metrics(gt_seqs=np.moveaxis(gt_out, [0, 1, 2, 3], [0, 2, 1, 3]),
                                        seqs=np.moveaxis(net_out, [0, 1, 2, 3], [0, 2, 1, 3]))
    print('entropy', ent, 'kl1', kl1, 'kl2', kl2)

    gt_out_4s = gt_out[:, :200, :, :]
    net_out_4s = net_out[:, :200, :, :]
    _ = compute_ent_metrics_splits(np.moveaxis(gt_out_4s, [0, 1, 2, 3], [0, 2, 1, 3]),
                                   np.moveaxis(net_out_4s, [0, 1, 2, 3], [0, 2, 1, 3]), seq_len=200)

    test_datenc_np = np.reshape(test_datenc_np, [test_datenc_np.shape[0], test_datenc_np.shape[1], 17, 3])
    test_datdec_np = np.reshape(test_datdec_np, [test_datdec_np.shape[0], test_datdec_np.shape[1], 17, 3])
    test_decout_np = np.reshape(test_decout_np, [test_decout_np.shape[0], test_decout_np.shape[1], 17, 3])
    sel = 5
    write_movie(np.transpose(test_datenc_np[sel], [1, 2, 0]), r_base=1,
                name='test_in.mp4')
    write_movie(np.transpose(test_decout_np[sel], [1, 2, 0]), net=True, r_base=1,
                name='test_out.mp4')
    write_movie(np.transpose(test_datdec_np[sel], [1, 2, 0]), net=True, r_base=1,
                name='test_out_gt.mp4')