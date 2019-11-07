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
from mocap_experiments.util import compute_ent_metrics, organize_into_batches

PoseData = collections.namedtuple('PoseData', ['f', 'action', 'actor', 'array'])


experiments_folder = '/home/moritz/uni/fourier-prediction/mocap_experiments/log/mocap/test/'
experiment_directory = '2019-11-07 13:05:33_gru_size_1024_fft_True_bs_100_ps_64_' \
                       'dis_0_lr_0.001_dr_0.96_ds_1000_sp_1.0_rc_False_pt_3984588_clw_0.01_' \
                       'csp_64_wf_hann_ws_64_ol_32_ffts_32_fftp_3_fl_None_eps_0.01_fftcr_12/'
path = experiments_folder + experiment_directory

pd = pickle.load(open(path + 'param.pkl', 'rb'))
mocap_handler_test = H36MDataSet(train=False, chunk_size=pd['chunk_size'], dataset_name='h36m')

graph = FFTpredictionGraph(pd)

gpu_options = tf.GPUOptions(visible_device_list=str(pd['GPUs'])[1:-1])
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)
with tf.Session(graph=graph.graph, config=config) as sess:
    test_data = mocap_handler_test.get_batches()
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