import sys
import io
import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from power_data_handler import PowerDataHandler, MergePowerHandler
from prediction_graph import FFTpredictionGraph
from IPython.core.debugger import Pdb
debug_here = Pdb().set_trace


fifteen_minute_sampling = False

# set up a parameter dictionary.
pd = {}

pd['prediction_days'] = 60
if pd['prediction_days'] > 1:
    pd['context_days'] = pd['prediction_days']*2
else:
    pd['context_days'] = 15
pd['base_dir'] = 'log/power_pred_60d_1h/definite2/'
pd['cell_type'] = 'gru'
pd['num_units'] = 166
pd['sample_prob'] = 1.0
pd['init_learning_rate'] = 0.004
pd['decay_rate'] = 0.95


pd['epochs'] = 80
pd['GPUs'] = [0]
pd['batch_size'] = 100
# window_function = 'hann'
# pd['window_function'] = 'learned_tukey'
# pd['window_function'] = 'learned_plank'
pd['window_function'] = 'learned_gaussian'
pd['fft_compression_rate'] = None
pd['freq_loss'] = None
pd['use_residuals'] = True
pd['fft'] = False
pd['linear_reshape'] = True
pd['stiefel'] = False

if fifteen_minute_sampling is True:
    pd['decay_steps'] = 390
else:
    pd['decay_steps'] = 455

if fifteen_minute_sampling is True:
    pd['samples_per_day'] = 96
    path = './power_data/15m_by_country_by_company/'
    power_handler = PowerDataHandler(path,
                                     pd['context_days'])
    pd['power_handler'] = power_handler
else:
    pd['samples_per_day'] = 24
    path = './power_data/15m_by_country_by_company/'
    power_handler_min15 = PowerDataHandler(path, pd['context_days'],
                                           samples_per_day=96,
                                           test_keys={})
    path = './power_data/30m_by_country_by_company/'
    power_handler_min30 = PowerDataHandler(path, pd['context_days'],
                                           samples_per_day=48,
                                           test_keys={})
    path = './power_data/1h_by_country_by_company/'
    power_handler_1h = PowerDataHandler(path, pd['context_days'],
                                        samples_per_day=24,
                                        test_keys={})
    testing_keys = [('germany_TenneT_GER', '2015'),
                    ('germany_Amprion', '2018'),
                    ('austria_CTA', '2017'),
                    ('belgium_CTA', '2016'),
                    ('UK_nationalGrid', '2015')]
    power_handler = MergePowerHandler(pd['context_days'],
                                      [power_handler_1h,
                                       power_handler_min30,
                                       power_handler_min15],
                                      testing_keys=testing_keys)
    pd['power_handler'] = power_handler

if pd['prediction_days'] > 1:
    pd['window_size'] = int(pd['samples_per_day']*5)
    pd['pred_samples'] = int(pd['prediction_days']*pd['samples_per_day'])
    pd['discarded_samples'] = 0
else:
    pd['pred_samples'] = int(pd['samples_per_day']*1.5)
    pd['discarded_samples'] = int(pd['samples_per_day']*0.5)
    pd['window_size'] = int(pd['samples_per_day'])

pd['overlap'] = int(pd['window_size']*0.5)
pd['step_size'] = pd['window_size'] - pd['overlap']
pd['fft_pred_samples'] = pd['pred_samples'] // pd['step_size'] + 1
pd['input_samples'] = pd['context_days']*pd['samples_per_day']

if pd['fft']:
    # debug_here()
    if pd['fft_compression_rate']:
        pd['num_proj'] = int((pd['window_size']//2 + 1) / pd['fft_compression_rate'])
    else:
        pd['num_proj'] = int((pd['window_size']//2 + 1))
elif pd['linear_reshape']:
    pd['num_proj'] = pd['step_size']
else:
    pd['num_proj'] = 1

if pd['fft']:
    pd['epsilon'] = 1e-2
else:
    pd['epsilon'] = None

# define a list of experiments.
lpd_lst = []

# cell_size_loop
fft_loop = pd['fft']
if fft_loop:
    assert pd['fft'] is True
    assert pd['linear_reshape'] is False
    # cell_type loop:
    for cell_type in ['gru', 'cgRNN']:
        # size loop.
        for num_units in [8, 32, 128]:
            # window_loop
            for window in ['hann', 'learned_tukey', 'learned_gaussian', 'boxcar']:
                # compression loop:
                for compression in [None, 2, 4]:
                    cpd = pd.copy()
                    cpd['window_function'] = window
                    cpd['fft_compression_rate'] = compression
                    cpd['num_units'] = num_units
                    cpd['cell_type'] = cell_type
                    if cpd['fft_compression_rate']:
                        cpd['num_proj'] = int((cpd['window_size']//2 + 1)
                                              / cpd['fft_compression_rate'])
                    else:
                        cpd['num_proj'] = int((cpd['window_size']//2 + 1))
                    lpd_lst.append(cpd)

reshape_loop = pd['linear_reshape']
if reshape_loop:
    assert pd['fft'] is False
    assert pd['linear_reshape'] is True
    for num_units in [8, 32, 128]:
        # cell_type loop:
        for cell_type in ['gru', 'cgRNN']:
                cpd = pd.copy()
                cpd['num_units'] = num_units
                cpd['cell_type'] = cell_type
                lpd_lst.append(cpd)

time_loop = not (pd['linear_reshape'] or pd['fft'])
if time_loop:
    assert pd['fft'] is False
    assert pd['linear_reshape'] is False
    cpd = pd.copy()
    for cell_type in ['gru', 'cgRNN']:
        for num_units in [8, 32, 128]:
            # cell_type loop:
            cpd['num_units'] = num_units
            cpd['cell_type'] = cell_type
            lpd_lst.append(cpd)

for exp_no, lpd in enumerate(lpd_lst):
    print('---------- Experiment', exp_no, 'of', len(lpd_lst), '----------')
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    lpd['time_str'] = time_str
    print(lpd)
    pgraph = FFTpredictionGraph(lpd)
    param_str = '_' + lpd['cell_type'] + '_size_' + str(lpd['num_units']) + \
        '_fft_' + str(lpd['fft']) + \
        '_fm_' + str(fifteen_minute_sampling) + \
        '_bs_' + str(lpd['batch_size']) + \
        '_ps_' + str(lpd['pred_samples']) + \
        '_dis_' + str(lpd['discarded_samples']) + \
        '_lr_' + str(lpd['init_learning_rate']) + \
        '_dr_' + str(lpd['decay_rate']) + \
        '_ds_' + str(lpd['decay_steps']) + \
        '_sp_' + str(lpd['sample_prob']) + \
        '_rc_' + str(lpd['use_residuals']) + \
        '_pt_' + str(pgraph.total_parameters)

    if lpd['fft']:
        param_str += '_wf_' + str(lpd['window_function'])
        param_str += '_ws_' + str(lpd['window_size'])
        param_str += '_ol_' + str(lpd['overlap'])
        param_str += '_ffts_' + str(lpd['step_size'])
        param_str += '_fftp_' + str(lpd['fft_pred_samples'])
        param_str += '_fl_' + str(lpd['freq_loss'])
        param_str += '_eps_' + str(lpd['epsilon'])
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

    test_data = power_handler.get_test_set()
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
            training_batches = power_handler.get_training_set()

            def organize_into_batches(batches):
                batch_total = len(batches)
                split_into = int(batch_total/lpd['batch_size'])
                batch_lst = np.array_split(np.stack(batches),
                                           split_into)
                return batch_lst

            batch_lst = organize_into_batches(training_batches)
            # ilpdb.set_trace()

            for it, batch in enumerate(batch_lst):
                start = time.time()
                # array_split add elements here and there, the true data is at 1
                feed_dict = {pgraph.data_nd: np.reshape(
                    batch[:lpd['batch_size'], :, :, 1],
                    [lpd['batch_size'], lpd['context_days']*lpd['samples_per_day'], 1])}

                np_loss, summary_to_file, np_global_step, _, \
                    datenc_np, encout_np, datdec_np, decout_np, \
                    datand_np = \
                    sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                              pgraph.training_op, pgraph.data_encoder_gt,
                              pgraph.encoder_out, pgraph.data_decoder_gt,
                              pgraph.decoder_out, pgraph.data_nd],
                             feed_dict=feed_dict)
                stop = time.time()
                if it % 100 == 0:
                    print('it: %5d, loss: %5.6f, time: %1.2f [s], epoch: %3d of %3d'
                          % (it, np_loss, stop-start, e, lpd['epochs']))

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
            print('Saving a copy.')
            pgraph.saver.save(sess, lpd['base_dir'] + time_str +
                              param_str + '/weights/cpk',
                              global_step=np_global_step)
            # do a test run.
            print('test run ', end='')
            mse_lst_net = []
            mse_lst_off = []

            test_batch_lst = organize_into_batches(test_data)
            for test_batch in test_batch_lst:
                gt = np.reshape(test_batch[:lpd['batch_size'], :, :, 1],
                                [lpd['batch_size'],
                                 lpd['context_days']*lpd['samples_per_day'],
                                 1])
                official_pred = np.reshape(test_batch[:lpd['batch_size'], :, :, 0],
                                           [lpd['batch_size'],
                                            lpd['context_days']
                                            * lpd['samples_per_day'], 1])
                feed_dict = {pgraph.data_nd: gt}
                if lpd['fft']:
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
                official_pred = official_pred[0, -lpd['pred_samples']:, 0]
                gt = gt[0, -lpd['pred_samples']:, 0]
                mse_lst_net.append(
                    np.mean((gt[lpd['discarded_samples']:]
                             - net_pred[lpd['discarded_samples']:lpd['pred_samples']])
                            ** 2))
                mse_lst_off.append(
                    np.mean((
                        gt[lpd['discarded_samples']:]
                        - official_pred[lpd['discarded_samples']:lpd['pred_samples']])
                        ** 2))
                print('.', end='')
            mse_net = np.mean(np.array(mse_lst_net))
            mse_off = np.mean(np.array(mse_lst_off))
            print()
            print('epoch: %5d,  test mse_net: %5.2f, test mse_off: %5.2f' %
                  (e, mse_net, mse_off))
            print('baseline difference: %5.2f' % (mse_off-mse_net))

            # add to tensorboard
            mse_net_summary = tf.Summary.Value(tag='mse_net_test', simple_value=mse_net)
            mse_net_summary = tf.Summary(value=[mse_net_summary])
            mse_off_summary = tf.Summary.Value(tag='mse_off_test', simple_value=mse_off)
            mse_off_summary = tf.Summary(value=[mse_off_summary])
            summary_writer.add_summary(mse_net_summary, global_step=np_global_step)
            summary_writer.add_summary(mse_off_summary, global_step=np_global_step)
            mse_diff_summary = tf.Summary.Value(tag='mse_net_off_diff',
                                                simple_value=mse_off-mse_net)
            mse_diff_summary = tf.Summary(value=[mse_diff_summary])
            summary_writer.add_summary(mse_diff_summary, global_step=np_global_step)
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
