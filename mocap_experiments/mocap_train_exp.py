import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import io
import copy
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

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

PoseData = collections.namedtuple('PoseData', ['f', 'action', 'actor', 'array'])


def np_scalar_to_summary(tag: str, scalar: np.array, np_step: np.array,
                         summary_file_writer: tf.summary.FileWriter):
    """
    Adds a numpy scalar to the logfile.
    :param tag: The tensorboard plot title.
    :param scalar: The scalar value to be recordd in that plot.
    :param np_step: The x-Axis step
    :param summary_file_writer: The summary writer used to do the recording.
    """
    mse_net_summary = tf.Summary.Value(tag=tag, simple_value=scalar)
    mse_net_summary = tf.Summary(value=[mse_net_summary])
    summary_file_writer.add_summary(mse_net_summary, global_step=np_step)


# set up a parameter dictionary.
pd = {}
pd['base_dir'] = './log/mocap_cvpr_workshop_5/'
pd['cell_type'] = 'gru'
pd['num_units'] = 1024*3
pd['sample_prob'] = 1.0
pd['init_learning_rate'] = 0.001
pd['decay_rate'] = 0.97
# pd['input_noise_std'] = 0

kl1_target = 0.02
kl2_target = 0.02
mse_target = 5000

pd['iterations'] = 300  # 400
pd['GPUs'] = [0]
pd['batch_size'] = 50
# pd['window_function'] = 'learned_tukey'
# pd['window_function'] = 'learned_plank'
pd['window_function'] = 'learned_gaussian'  # 'learned_gaussian'
pd['freq_loss'] = None
pd['use_residuals'] = True
pd['fft'] = True
pd['window_size'] = 16
pd['fft_compression_rate'] = 1
pd['overlap'] = int(pd['window_size']*0.5)
pd['step_size'] = pd['window_size'] - pd['overlap']
pd['linear_reshape'] = False
pd['downsampling'] = 1
pd['stiefel'] = False
pd['input_noise'] = False

pd['decay_steps'] = 1000
pd['chunk_size'] = 128
pd['input_samples'] = pd['chunk_size']

mocap_handler = H36MDataSet(train=True, chunk_size=pd['chunk_size'], dataset_name='h36mv2')
mocap_handler_test = H36MDataSet(train=False, chunk_size=pd['chunk_size'], dataset_name='h36mv2')
pd['mocap_handler'] = mocap_handler

pd['consistency_loss'] = True
pd['mse_samples'] = 64
pd['pred_samples'] = 64


def fix_pd(pd):
    assert pd['mse_samples'] <= pd['pred_samples']
    if pd['consistency_loss']:
        pd['consistency_samples'] = 64
        assert pd['consistency_samples'] <= pd['pred_samples']
        pd['consistency_loss_weight'] = 0.000
    pd['discarded_samples'] = 0


    if pd['fft']:
        pd['fft_pred_samples'] = pd['pred_samples'] // pd['step_size'] + 1
        if pd['fft_compression_rate']:
            pd['num_proj'] = 17*3*int((pd['window_size']//2 + 1) / pd['fft_compression_rate'])
        else:
            pd['num_proj'] = 17*3*int((pd['window_size']//2 + 1))
    elif pd['linear_reshape']:
        pd['num_proj'] = (pd['step_size']//pd['downsampling'])*17*3
    else:
        pd['num_proj'] = 17*3

    if pd['fft']:
        pd['epsilon'] = 1e-2
    else:
        pd['epsilon'] = None
    return pd


fftc_pd = copy.copy(pd)
fftc_pd['fft_compression_rate'] = 4

fftc_pd2 = copy.copy(pd)
fftc_pd2['fft_compression_rate'] = 8

fftc_pd3 = copy.copy(pd)
fftc_pd3['fft_compression_rate'] = 2

re_pd = copy.copy(pd)
re_pd['fft'] = False
re_pd['linear_reshape'] = True
re_pd['downsampling'] = 1

red_pd = copy.copy(pd)
red_pd['fft'] = False
red_pd['linear_reshape'] = True
red_pd['downsampling'] = 4

red_pd2 = copy.copy(pd)
red_pd2['fft'] = False
red_pd2['linear_reshape'] = True
red_pd2['downsampling'] = 8

red_pd3 = copy.copy(pd)
red_pd3['fft'] = False
red_pd3['linear_reshape'] = True
red_pd3['downsampling'] = 2

time_pd = copy.copy(pd)
time_pd['fft'] = False
time_pd['linear_reshape'] = False


lpd_lst = [fix_pd(pd), fix_pd(fftc_pd), fix_pd(fftc_pd2), fix_pd(re_pd), fix_pd(red_pd), fix_pd(red_pd2),
           fix_pd(time_pd), fix_pd(fftc_pd3), fix_pd(red_pd3)]
# lpd_lst = [fix_pd(time_pd)]
print('number of experiments:', len(lpd_lst))

for exp_no, lpd in enumerate(lpd_lst):
    print('---------- Experiment', exp_no, 'of', len(lpd_lst), '----------')
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    lpd['time_str'] = time_str
    print(lpd)
    pgraph = FFTpredictionGraph(lpd)
    param_str = '_' + lpd['cell_type'] + '_size_' + str(lpd['num_units']) + \
        '_fft_' + str(lpd['fft']) + \
        '_bs_' + str(lpd['batch_size']) + \
        '_ps_' + str(lpd['pred_samples']) + \
        '_dis_' + str(lpd['discarded_samples']) + \
        '_lr_' + str(lpd['init_learning_rate']) + \
        '_dr_' + str(lpd['decay_rate']) + \
        '_ds_' + str(lpd['decay_steps']) + \
        '_sp_' + str(lpd['sample_prob']) + \
        '_mses_' + str(lpd['mse_samples']) + \
        '_rc_' + str(lpd['use_residuals']) + \
        '_pt_' + str(pgraph.total_parameters)

    if lpd['consistency_loss']:
        param_str += '_clw_' + str(lpd['consistency_loss_weight'])
        param_str += '_csp_' + str(lpd['consistency_samples'])

    if lpd['fft']:
        param_str += '_wf_' + str(lpd['window_function'])
        param_str += '_ws_' + str(lpd['window_size'])
        param_str += '_ol_' + str(lpd['overlap'])
        param_str += '_ffts_' + str(lpd['step_size'])
        param_str += '_fftp_' + str(lpd['fft_pred_samples'])
        param_str += '_fl_' + str(lpd['freq_loss'])
        param_str += '_eps_' + str(lpd['epsilon'])
        if lpd['fft_compression_rate']:
            param_str += '_fftcr_' + str(lpd['fft_compression_rate'])

    if lpd['stiefel']:
        param_str += '_stfl'

    if lpd['linear_reshape']:
        param_str += '_linre'

        if lpd['downsampling'] > 1:
            param_str += '_downs_' + str(lpd['downsampling'])


    # do each of the experiments in the parameter dictionary list.
    print(param_str)
    # ilpdb.set_trace()
    summary_writer = tf.summary.FileWriter(lpd['base_dir'] + lpd['time_str'] + param_str,
                                           graph=pgraph.graph)
    # dump the parameters
    with open(lpd['base_dir'] + lpd['time_str'] + param_str + '/param.pkl', 'wb') as file:
        pickle.dump(lpd, file)

    test_data = mocap_handler_test.get_batches()
    # train this.
    gpu_options = tf.GPUOptions(visible_device_list=str(lpd['GPUs'])[1:-1])
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=gpu_options)
    with tf.Session(graph=pgraph.graph, config=config) as sess:
        print('initialize....')
        pgraph.init_op.run()
        for e in range(0, lpd['iterations']):
            training_batches = mocap_handler.get_batches()

            batch_lst = organize_into_batches(training_batches, lpd)

            for it, batch in enumerate(batch_lst):
                start = time.time()
                # array_split add elements here and there, the true data is at 1
                feed_dict = {pgraph.data_nd: np.reshape(
                    batch, [lpd['batch_size'], lpd['chunk_size'], 17*3])}

                np_loss, np_consistency_loss, summary_to_file, np_global_step, _, \
                    _, train_datdec_np, train_decout_np, \
                    _ = \
                    sess.run([pgraph.loss, pgraph.consistency_loss, pgraph.summary_sum, pgraph.global_step,
                              pgraph.training_op, pgraph.data_encoder_time,
                              pgraph.data_decoder_time,
                              pgraph.decoder_out, pgraph.data_nd],
                             feed_dict=feed_dict)
                stop = time.time()
                if it % 10 == 0:
                    print('it: %5d, loss: %5.6f, consist loss: %5.6f, time: %1.2f [s], epoch: %3d of %3d'
                          % (it, np_loss, np_consistency_loss, stop-start, e, lpd['iterations']))
                summary_writer.add_summary(summary_to_file, global_step=np_global_step)

                if it % 1000 == 0:
                    plt.figure()
                    plt.plot(
                        train_decout_np[0, lpd['discarded_samples']:lpd['pred_samples'], 0])
                    plt.plot(
                        train_datdec_np[0, lpd['discarded_samples']:lpd['pred_samples'], 0])
                    plt.plot(
                        np.abs(
                            train_decout_np[0, lpd['discarded_samples']:lpd['pred_samples'], 0]
                            - train_datdec_np[0, lpd['discarded_samples']:lpd['pred_samples'],
                                        0]))
                    plt.title("Prediction vs. ground truth")
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    summary_image = tf.Summary.Image(
                        encoded_image_string=buf.getvalue(),
                        height=int(plt.rcParams["figure.figsize"][0]*100),
                        width=int(plt.rcParams["figure.figsize"][1]*100))
                    summary_image = tf.Summary.Value(tag='prediction_error',
                                                     image=summary_image)
                    summary_image = tf.Summary(value=[summary_image])
                    summary_writer.add_summary(summary_image, global_step=np_global_step)
                    plt.close()
                    buf.close()

            # do a test run.
            if e % 4 == 0:
                test_mse_lst_net = []
                test_ae_net = []
                test_csl_lst_net = []
                test_net_lst_out = []
                test_gt_lst_out = []
                psx_lst = []
                psy_lst = []
                ps_kl_xy_lst = []
                ps_kl_yx_lst = []
                test_runtime_lst = []
                test_batch_lst = organize_into_batches(test_data, lpd)
                for test_batch in test_batch_lst:
                    test_gt = np.reshape(test_batch, [lpd['batch_size'], lpd['chunk_size'], 17*3])

                    test_feed_dict = {pgraph.data_nd: test_gt}
                    test_time_start = time.time()
                    if lpd['fft']:
                        np_loss, np_global_step, \
                            test_datenc_np, test_datdec_np, test_decout_np, \
                            _, window_np, cs_loss_np, mean_psx_np, \
                            mean_psy_np, mean_ps_kl_xy_np, mean_ps_kl_yx_np = \
                            sess.run([pgraph.loss, pgraph.global_step,
                                      pgraph.data_encoder_time,
                                      pgraph.data_decoder_time,
                                      pgraph.decoder_out, pgraph.data_nd, pgraph.window,
                                      pgraph.consistency_loss, pgraph.mean_psx, pgraph.mean_psy,
                                      pgraph.mean_ps_kl_xy, pgraph.mean_ps_kl_yx],
                                     feed_dict=test_feed_dict)
                    else:
                        np_loss, np_global_step, \
                            test_datenc_np, test_datdec_np, test_decout_np, \
                            _, cs_loss_np, mean_psx_np, mean_psy_np, \
                            mean_ps_kl_xy_np, mean_ps_kl_yx_np = \
                            sess.run([pgraph.loss, pgraph.global_step,
                                      pgraph.data_encoder_time,
                                      pgraph.data_decoder_time,
                                      pgraph.decoder_out, pgraph.data_nd, pgraph.consistency_loss,
                                      pgraph.mean_psx, pgraph.mean_psy,
                                      pgraph.mean_ps_kl_xy, pgraph.mean_ps_kl_yx],
                                     feed_dict=test_feed_dict)
                    test_runtime_lst.append(time.time() - test_time_start)
                    net_pred = test_decout_np[:, :, :]*mocap_handler.std + mocap_handler.mean
                    test_gt = test_gt[:, -lpd['pred_samples']:, :]
                    test_mse_lst_net.append(
                        np.mean((test_gt[:, lpd['discarded_samples']:lpd['mse_samples']]
                                 - net_pred[:, lpd['discarded_samples']:lpd['mse_samples']])
                                ** 2))
                    test_ae_net.append(np.mean(np.abs(test_gt[:, lpd['discarded_samples']:lpd['mse_samples']]
                                       - net_pred[:, lpd['discarded_samples']:lpd['mse_samples']])))
                    test_net_lst_out.append(test_decout_np)
                    test_gt_lst_out.append(test_datdec_np)
                    test_csl_lst_net.append(cs_loss_np)
                    psx_lst.append(mean_psx_np)
                    psy_lst.append(mean_psy_np)
                    ps_kl_xy_lst.append(mean_ps_kl_xy_np)
                    ps_kl_yx_lst.append(mean_ps_kl_yx_np)

                    print('.', end='')
                mse_net = np.mean(np.array(test_mse_lst_net))
                ae_net = np.mean(test_ae_net)
                cs_loss_np_mean = np.mean(test_csl_lst_net)
                mean_psx = np.mean(psx_lst)
                mean_psy = np.mean(psy_lst)
                mean_ps_kl_xy = np.mean(ps_kl_xy_lst)
                mean_ps_kl_yx = np.mean(ps_kl_yx_lst)
                print()
                print('epoch: %5d,  test mse_net: %5.2f, test_cs_loss: %5.2f' %
                      (e, mse_net, cs_loss_np_mean))

                if mse_target > mse_net:
                    mse_target = mse_net
                    print('Saving a copy.')
                    ret = pgraph.saver.save(sess, lpd['base_dir'] + time_str +
                                            param_str + '/mse_'+str(mse_net)+'/cpk')
                    print('saved at:', ret)

                # add to tensorboard
                np_scalar_to_summary('test/mse_net_test', mse_net, np_global_step, summary_writer)
                np_scalar_to_summary('test/abse_net_test', ae_net, np_global_step, summary_writer)
                np_scalar_to_summary('test/cl_net_test', cs_loss_np_mean, np_global_step, summary_writer)
                np_scalar_to_summary('test_cartesian_50hz/psx', mean_psx, np_global_step, summary_writer)
                np_scalar_to_summary('test_cartesian_50hz/psy', mean_psy, np_global_step, summary_writer)
                np_scalar_to_summary('test_cartesian_50hz/ps_kl_xy', mean_ps_kl_xy, np_global_step, summary_writer)
                np_scalar_to_summary('test_cartesian_50hz/ps_kl_yx', mean_ps_kl_yx, np_global_step, summary_writer)
                np_scalar_to_summary('test/runtime', np.mean(test_runtime_lst), np_global_step, summary_writer)

                # if lpd['fft']:
                #     # window plot in tensorboard.
                #     plt.figure()
                #     plt.plot(window_np)
                #     plt.title(lpd['window_function'])
                #     buf2 = io.BytesIO()
                #     plt.savefig(buf2, format='png')
                #     buf2.seek(0)
                #     summary_image2 = tf.Summary.Image(
                #         encoded_image_string=buf2.getvalue(),
                #         height=int(plt.rcParams["figure.figsize"][0]*100),
                #         width=int(plt.rcParams["figure.figsize"][1]*100))
                #     summary_image2 = tf.Summary.Value(tag=lpd['window_function'],
                #                                       image=summary_image2)
                #     summary_image2 = tf.Summary(value=[summary_image2])
                #     summary_writer.add_summary(summary_image2, global_step=np_global_step)
                #     plt.close()
                #     buf.close()

            # do the evaluation
            if e % 5 == 0:
                # print('evaluate')
                net_out = np.concatenate(test_net_lst_out, axis=0)
                gt_out = np.concatenate(test_gt_lst_out, axis=0)
                net_out = np.reshape(net_out, [net_out.shape[0], net_out.shape[1], 17, 3])
                gt_out = np.reshape(gt_out, [gt_out.shape[0], gt_out.shape[1], 17, 3])
                ent, kl1, kl2 = compute_ent_metrics(gt_seqs=np.moveaxis(gt_out[:, :pd['pred_samples'], :, :],
                                                                        [0, 1, 2, 3], [0, 2, 1, 3]),
                                                    seqs=np.moveaxis(net_out[:, :pd['pred_samples'], :, :],
                                                                     [0, 1, 2, 3], [0, 2, 1, 3]))
                print('eval at epoch', e, 'Euler entropy', ent, 'Euler kl1', kl1, 'Euler kl2', kl2)

                if kl1 < kl1_target and kl2 < kl2_target:
                    kl1_target = kl1
                    kl2_target = kl2
                    print('Saving a copy.')
                    ret = pgraph.saver.save(sess, lpd['base_dir'] + time_str +
                                            param_str + '/soa_kl1_kl2_'+str(kl1)+'_'+str(kl2)+'/cpk')
                    print('saved at:', ret)

                if lpd['pred_samples'] >= 200:
                    gt_out_4s_5hz = gt_out[:, :200:10, :, :]
                    net_out_4s_5hz = net_out[:, :200:10, :, :]
                    seqs_ent_global_mean, seqs_kl_gen_gt_mean, seqs_kl_gt_gen_mean = \
                        compute_ent_metrics_splits(np.moveaxis(gt_out_4s_5hz, [0, 1, 2, 3], [0, 2, 1, 3]),
                                                   np.moveaxis(net_out_4s_5hz, [0, 1, 2, 3], [0, 2, 1, 3]), seq_len=20)
                    for i in range(5):
                        np_scalar_to_summary('test_euler_5Hz/ent'+str(i),
                                             seqs_ent_global_mean[i],
                                             np_global_step, summary_writer)
                        np_scalar_to_summary('test_euler_5Hz/kl_gen_gt'+str(i),
                                             seqs_kl_gen_gt_mean[i], np_global_step, summary_writer)
                        np_scalar_to_summary('test_euler_5Hz/kl_gt_gen'+str(i),
                                             seqs_kl_gt_gen_mean[i], np_global_step, summary_writer)

                    gt_out_4s_50hz = gt_out[:, :200, :, :]
                    net_out_4s_50hz = net_out[:, :200, :, :]
                    seqs_ent_global_mean, seqs_kl_gen_gt_mean, seqs_kl_gt_gen_mean = \
                        compute_ent_metrics_splits(np.moveaxis(gt_out_4s_50hz, [0, 1, 2, 3], [0, 2, 1, 3]),
                                                   np.moveaxis(net_out_4s_50hz, [0, 1, 2, 3], [0, 2, 1, 3]),
                                                   seq_len=200)
                    for i in range(5):
                        np_scalar_to_summary('test_euler_50Hz/ent'+str(i),
                                             seqs_ent_global_mean[i],
                                             np_global_step, summary_writer)
                        np_scalar_to_summary('test_euler_50Hz/kl_gen_gt'+str(i),
                                             seqs_kl_gen_gt_mean[i], np_global_step, summary_writer)
                        np_scalar_to_summary('test_euler_50Hz/kl_gt_gen'+str(i),
                                             seqs_kl_gt_gen_mean[i], np_global_step, summary_writer)


        print('Saving a copy.')
        ret = pgraph.saver.save(sess, lpd['base_dir'] + time_str +
                                param_str + '/weights/cpk')
        print('saved at:', ret)
        test_datenc_np = np.reshape(test_datenc_np, [test_datenc_np.shape[0], test_datenc_np.shape[1], 17, 3])
        test_datdec_np = np.reshape(test_datdec_np, [test_datdec_np.shape[0], test_datdec_np.shape[1], 17, 3])
        test_decout_np = np.reshape(test_decout_np, [test_decout_np.shape[0], test_decout_np.shape[1], 17, 3])
        sel = 5
        sel2 = 30
        gt_movie = np.concatenate([test_datenc_np, test_datdec_np], axis=1)
        net_movie = np.concatenate([test_datenc_np, test_decout_np], axis=1)
        write_movie(np.transpose(gt_movie[sel], [1, 2, 0]), r_base=1.5,
                    name=lpd['base_dir'] + time_str + param_str + '/in.mp4',
                    color_shift_at=lpd['chunk_size'] - lpd['pred_samples'] - 1)
        write_movie(np.transpose(net_movie[sel], [1, 2, 0]), r_base=1.5,
                    name=lpd['base_dir'] + time_str + param_str + '/out.mp4',
                    color_shift_at=lpd['chunk_size'] - lpd['pred_samples'] - 1)
        # write_movie(np.transpose(gt_movie[sel][:(224+60), :, :], [1, 2, 0]), r_base=1.5,
        #             name=lpd['base_dir'] + time_str + param_str + '/in_4s.mp4',
        #             color_shift_at=lpd['chunk_size'] - lpd['pred_samples'] - 1)
        # write_movie(np.transpose(net_movie[sel][:(224+60), :, :], [1, 2, 0]), r_base=1.5,
        #             name=lpd['base_dir'] + time_str + param_str + '/out_4s.mp4',
        #            color_shift_at=lpd['chunk_size'] - lpd['pred_samples'] - 1)
        write_movie(np.transpose(gt_movie[sel2][:(224+60), :, :], [1, 2, 0]), r_base=1.5,
                    name=lpd['base_dir'] + time_str + param_str + '/in2_4s.mp4',
                    color_shift_at=lpd['chunk_size'] - lpd['pred_samples'] - 1)
        write_movie(np.transpose(net_movie[sel2][:(224+60), :, :], [1, 2, 0]), r_base=1.5,
                    name=lpd['base_dir'] + time_str + param_str + '/out2_4s.mp4',
                    color_shift_at=lpd['chunk_size'] - lpd['pred_samples'] - 1)

        for sel in range(lpd['batch_size']):
            write_movie(np.transpose(gt_movie[sel], [1, 2, 0]), r_base=1.5,
                        name=lpd['base_dir'] + time_str + param_str + '/batch_el' + str(sel) + '_in.mp4',
                        color_shift_at=lpd['chunk_size'] - lpd['pred_samples'] - 1)
            write_movie(np.transpose(net_movie[sel], [1, 2, 0]), r_base=1.5,
                        name=lpd['base_dir'] + time_str + param_str + '/batch_el' + str(sel) + '_out.mp4',
                        color_shift_at=lpd['chunk_size'] - lpd['pred_samples'] - 1)
        print('done')
