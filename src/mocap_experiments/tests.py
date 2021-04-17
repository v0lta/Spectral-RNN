import io
import time
import pickle
import tensorflow as tf
import numpy as np
import collections
import matplotlib.pyplot as plt
from mocap_experiments.load_h36m import H36MDataSet
from mocap_experiments.prediction_graph import FFTpredictionGraph
from mocap_experiments.write_movie import write_movie, write_figure
from mocap_experiments.util import compute_ent_metrics, organize_into_batches, compute_ent_metrics_splits

PoseData = collections.namedtuple('PoseData', ['f', 'action', 'actor', 'array'])

# supplementary experiment.
base_path = '/home/moritz/uni/fourier-prediction/log/cvpr_workshop3/'
folder = '2020-03-05 09:07:06_gru_size_3072_fft_True_bs_50_ps_64_dis_0_lr_0.001_dr_0.97_ds_1000_sp_1.0' \
         '_mses_64_rc_False_pt_30827725_clw_0.0_csp_64_wf_learned_gaussian_ws_16_ol_8_ffts_' \
         '8_fftp_9_fl_None_eps_0.01_fftcr_4/'
checkpoint_folder = 'weights'
path = base_path + folder

pd = pickle.load(open(path + 'param.pkl', 'rb'))
mocap_handler_test = H36MDataSet(train=False, chunk_size=pd['chunk_size'], dataset_name='h36m')

graph = FFTpredictionGraph(pd)
gpu_options = tf.GPUOptions(visible_device_list=str(pd['GPUs'])[1:-1])
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False)
                        #gpu_options=gpu_options)
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
        net_pred = test_decout_np[:, :, :]*pd['mocap_handler'].std + pd['mocap_handler'].mean
        gt = gt[:, -pd['pred_samples']:, :]
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
    sel = 30  #11  # 30 # 45
    gt_movie = np.concatenate([test_datenc_np, test_datdec_np], axis=1)
    net_movie = np.concatenate([test_datenc_np, test_decout_np], axis=1)
    # write_movie(np.transpose(gt_movie[sel], [1, 2, 0]), r_base=2,
    #             name='test_in.mp4', color_shift_at=pd['chunk_size'] - pd['pred_samples'])
    # write_movie(np.transpose(net_movie[sel], [1, 2, 0]), r_base=1,
    #             name='test_out.mp4', color_shift_at=pd['chunk_size'] - pd['pred_samples'])
    write_figure(np.transpose(net_movie[sel][::12, :, :], [1, 2, 0]),
                              color_shift_at=int(pd['chunk_size'] - pd['pred_samples'])/12, r_base=1.5,
                  name='test_data/test_figure.pdf')
    gt_movie = gt_movie[:, :, :, :]
    net_movie = net_movie[:, :, :, :]
    for sel in [30]:  # range(pd['batch_size']):
        write_movie(np.transpose(gt_movie[sel], [1, 2, 0]), r_base=1.8,
                    name='test_data/' + str(sel) + '5Hz_in.mp4',
                    color_shift_at=pd['chunk_size'] - pd['pred_samples'] - 1, title='context and ground truth')
        write_movie(np.transpose(net_movie[sel], [1, 2, 0]), r_base=1.8,
                    name='test_data/' + str(sel) + '5Hz_out.mp4',
                    color_shift_at=pd['chunk_size'] - pd['pred_samples'] - 1, title='context and prediction')
    print('done')