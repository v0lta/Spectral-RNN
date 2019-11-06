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
from mocap_experiments.util import compute_ent_metrics

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

pd['base_dir'] = 'log/mocap/test/'
pd['cell_type'] = 'gru'
pd['num_units'] = 1024
pd['sample_prob'] = 1.0
pd['init_learning_rate'] = 0.001
pd['decay_rate'] = 0.98


pd['epochs'] = 1600
pd['GPUs'] = [0]
pd['batch_size'] = 100
# pd['window_function'] = 'learned_tukey'
# pd['window_function'] = 'learned_plank'
pd['window_function'] = 'hann'  # 'learned_gaussian'
pd['freq_loss'] = None
pd['use_residuals'] = False
pd['fft'] = True
pd['linear_reshape'] = False
pd['stiefel'] = False

pd['decay_steps'] = 1000
pd['chunk_size'] = 128

mocap_handler = H36MDataSet(train=True, chunk_size=pd['chunk_size'], dataset_name='h36m')
mocap_handler_test = H36MDataSet(train=False, chunk_size=pd['chunk_size'], dataset_name='h36m')
pd['mocap_handler'] = mocap_handler

pd['consistency_loss'] = True
pd['mse_samples'] = 64
pd['pred_samples'] = 64
assert pd['mse_samples'] <= pd['pred_samples']
if pd['consistency_loss']:
    pd['consistency_samples'] = 50
    assert pd['consistency_samples'] <= pd['pred_samples']
    pd['consistency_loss_weight'] = 0.01
pd['window_size'] = 1
pd['discarded_samples'] = 0


if pd['fft']:
    pd['window_size'] = 64
    pd['fft_compression_rate'] = 16
    pd['overlap'] = int(pd['window_size']*0.5)
    pd['step_size'] = pd['window_size'] - pd['overlap']
    pd['fft_pred_samples'] = pd['pred_samples'] // pd['step_size'] + 1
    pd['input_samples'] = pd['chunk_size']
    if pd['fft_compression_rate']:
        pd['num_proj'] = 17*3*int((pd['window_size']//2 + 1) / pd['fft_compression_rate'])
    else:
        pd['num_proj'] = 17*3*int((pd['window_size']//2 + 1))
elif pd['linear_reshape']:
    pd['num_proj'] = pd['step_size']
else:
    pd['num_proj'] = 17*3

if pd['fft']:
    pd['epsilon'] = 1e-2
else:
    pd['epsilon'] = None

lpd_lst = []
# define a list of experiments.
for consistency_loss_weight in [0.1, 0.01, 0.001]:
    for learning_rate_decay_rate in [0.98, 0.96]:
        for num_units in [1024, 1024*2, 1024*3]:
            for fft_compression_rate in [4, 8, 16, 24, 32]:
                cpd = pd.copy()
                cpd['consistency_loss_weight'] = consistency_loss_weight
                cpd['num_units'] = num_units
                cpd['decay_rate'] = learning_rate_decay_rate
                if cpd['fft']:
                    cpd['fft_compression_rate'] = fft_compression_rate
                    cpd['num_proj'] = 17*3*int((cpd['window_size']//2 + 1) / cpd['fft_compression_rate'])
                lpd_lst.append(cpd)

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
        for e in range(0, lpd['epochs']):
            training_batches = mocap_handler.get_batches()

            def organize_into_batches(batches):
                batch_total = len(batches)
                split_into = int(batch_total/lpd['batch_size'])
                stop_at = lpd['batch_size']*split_into
                batch_lst = np.array_split(np.stack(batches[:stop_at]),
                                           split_into)
                return batch_lst

            batch_lst = organize_into_batches(training_batches)

            for it, batch in enumerate(batch_lst):
                start = time.time()
                # array_split add elements here and there, the true data is at 1
                feed_dict = {pgraph.data_nd: np.reshape(
                    batch, [lpd['batch_size'], lpd['chunk_size'], 17*3])}

                np_loss, np_consistency_loss, summary_to_file, np_global_step, _, \
                    datenc_np, datdec_np, decout_np, \
                    datand_np = \
                    sess.run([pgraph.loss, pgraph.consistency_loss, pgraph.summary_sum, pgraph.global_step,
                              pgraph.training_op, pgraph.data_encoder_time,
                              pgraph.data_decoder_time,
                              pgraph.decoder_out, pgraph.data_nd],
                             feed_dict=feed_dict)
                stop = time.time()
                if it % 10 == 0:
                    print('it: %5d, loss: %5.6f, consist los: %5.6f, time: %1.2f [s], epoch: %3d of %3d'
                          % (it, np_loss, np_consistency_loss, stop-start, e, lpd['epochs']))
                summary_writer.add_summary(summary_to_file, global_step=np_global_step)

                if it % 100 == 0:
                    plt.figure()
                    plt.plot(
                        decout_np[0, lpd['discarded_samples']:lpd['pred_samples'], 0])
                    plt.plot(
                        datdec_np[0, lpd['discarded_samples']:lpd['pred_samples'], 0])
                    plt.plot(
                        np.abs(
                            decout_np[0, lpd['discarded_samples']:lpd['pred_samples'], 0]
                            - datdec_np[0, lpd['discarded_samples']:lpd['pred_samples'],
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

            # epoch done. Save.
            if e % 10 == 0:
                print('Saving a copy.')
                pgraph.saver.save(sess, lpd['base_dir'] + time_str +
                                  param_str + '/weights/cpk',
                                  global_step=np_global_step)
            # do a test run.
            # print('test run ', end='')
            mse_lst_net = []
            net_lst_out = []
            gt_lst_out = []
            test_batch_lst = organize_into_batches(test_data)
            for test_batch in test_batch_lst:
                gt = np.reshape(test_batch, [lpd['batch_size'], lpd['chunk_size'], 17*3])

                feed_dict = {pgraph.data_nd: gt}
                if lpd['fft']:
                    np_loss, np_global_step, \
                        test_datenc_np, test_datdec_np, test_decout_np, \
                        datand_np, window_np = \
                        sess.run([pgraph.loss, pgraph.global_step,
                                  pgraph.data_encoder_time,
                                  pgraph.data_decoder_time,
                                  pgraph.decoder_out, pgraph.data_nd, pgraph.window],
                                 feed_dict=feed_dict)
                else:
                    np_loss, np_global_step, \
                        test_datenc_np, test_datdec_np, test_decout_np, \
                        datand_np = \
                        sess.run([pgraph.loss, pgraph.global_step,
                                  pgraph.data_encoder_time,
                                  pgraph.data_decoder_time,
                                  pgraph.decoder_out, pgraph.data_nd],
                                 feed_dict=feed_dict)
                net_pred = test_decout_np[:, :, 0]*mocap_handler.std + mocap_handler.mean
                gt = gt[:, -lpd['pred_samples']:, 0]
                mse_lst_net.append(
                    np.mean((gt[:, lpd['discarded_samples']:]
                             - net_pred[:, lpd['discarded_samples']:lpd['pred_samples']])
                            ** 2))
                net_lst_out.append(test_decout_np)
                gt_lst_out.append(test_datdec_np)

                print('.', end='')
            mse_net = np.mean(np.array(mse_lst_net))
            print()
            print('epoch: %5d,  test mse_net: %5.2f' %
                  (e, mse_net))

            # add to tensorboard
            np_scalar_to_summary('test/mse_net_test', mse_net, np_global_step, summary_writer)
            if lpd['fft']:
                # window plot in tensorboard.
                plt.figure()
                plt.plot(window_np)
                plt.title(lpd['window_function'])
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png')
                buf2.seek(0)
                summary_image2 = tf.Summary.Image(
                    encoded_image_string=buf2.getvalue(),
                    height=int(plt.rcParams["figure.figsize"][0]*100),
                    width=int(plt.rcParams["figure.figsize"][1]*100))
                summary_image2 = tf.Summary.Value(tag=lpd['window_function'],
                                                  image=summary_image2)
                summary_image2 = tf.Summary(value=[summary_image2])
                summary_writer.add_summary(summary_image2, global_step=np_global_step)
                plt.close()
                buf.close()

            # do the evaluation
            if e % 5 == 0:
                # print('evaluate')
                net_out = np.concatenate(net_lst_out, axis=0)
                gt_out = np.concatenate(gt_lst_out, axis=0)
                net_out = np.reshape(net_out, [net_out.shape[0], net_out.shape[1], 17, 3])
                gt_out = np.reshape(gt_out, [gt_out.shape[0], gt_out.shape[1], 17, 3])
                ent, kl1, kl2 = compute_ent_metrics(gt_seqs=np.moveaxis(gt_out, [0, 1, 2, 3], [0, 2, 1, 3]),
                                                    seqs=np.moveaxis(net_out, [0, 1, 2, 3], [0, 2, 1, 3]),
                                                    seq_len=lpd['pred_samples'])
                print('eval at epoch', e, 'entropy', ent, 'kl1', kl1, 'kl2', kl2)
                np_scalar_to_summary('test/entropy', ent, np_global_step, summary_writer)
                np_scalar_to_summary('test/kl1', kl1, np_global_step, summary_writer)
                np_scalar_to_summary('test/kl2', kl2, np_global_step, summary_writer)

        test_datenc_np = np.reshape(test_datenc_np, [test_datenc_np.shape[0], test_datenc_np.shape[1], 17, 3])
        test_datdec_np = np.reshape(test_datdec_np, [test_datdec_np.shape[0], test_datdec_np.shape[1], 17, 3])
        test_decout_np = np.reshape(test_decout_np, [test_decout_np.shape[0], test_decout_np.shape[1], 17, 3])
        write_movie(np.transpose(test_datenc_np[50], [1, 2, 0]), r_base=1, name='in.mp4')
        write_movie(np.transpose(test_decout_np[50], [1, 2, 0]), net=True, r_base=1, name='out.mp4')
        write_movie(np.transpose(test_datdec_np[50], [1, 2, 0]), net=True, r_base=1, name='out_gt.mp4')
        print('done')
