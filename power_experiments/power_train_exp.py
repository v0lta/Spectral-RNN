import io
import time
import copy
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from power_experiments.power_data_handler import PowerDataHandler, MergePowerHandler
from power_experiments.prediction_graph import FFTpredictionGraph


fifteen_minute_sampling = True

# set up a parameter dictionary.
pd = {}
pd['base_dir'] = 'log/cvpr_workshop_power_pred4/'

pd['prediction_days'] = 60 #,  1
if pd['prediction_days'] > 1:
    pd['context_days'] = pd['prediction_days']*2
else:
    pd['context_days'] = 15

pd['cell_type'] = 'gru'
pd['num_units'] = 64
pd['sample_prob'] = 1.0
pd['init_learning_rate'] = 0.004
pd['decay_rate'] = 0.96


pd['epochs'] = 320
pd['GPUs'] = [0]
pd['batch_size'] = 50
# window_function = 'hann'
# pd['window_function'] = 'learned_tukey'
# pd['window_function'] = 'learned_plank'
pd['window_function'] = 'learned_gaussian'
pd['fft_compression_rate'] = None
pd['conv_fft_bins'] = None
pd['fully_fft_comp'] = None  #TODO: fixme
pd['freq_loss'] = None
pd['use_residuals'] = True
pd['fft'] = True
pd['linear_reshape'] = False
pd['downsampling'] = 1  # set to 1 to turn this off.
pd['stiefel'] = False


def fix_parameters(pd):
    if fifteen_minute_sampling is True:
        pd['decay_steps'] = 390
    else:
        pd['decay_steps'] = 455

    if fifteen_minute_sampling is True:
        pd['samples_per_day'] = 96
        path = './power_experiments/power_data/15m_by_country_by_company/'
        power_handler = PowerDataHandler(path,
                                         pd['context_days'])
        pd['power_handler'] = power_handler
    else:
        pd['samples_per_day'] = 24
        path = './power_experiments/power_data/15m_by_country_by_company/'
        power_handler_min15 = PowerDataHandler(path, pd['context_days'],
                                               samples_per_day=96,
                                               test_keys={})
        path = './power_experiments/power_data/30m_by_country_by_company/'
        power_handler_min30 = PowerDataHandler(path, pd['context_days'],
                                               samples_per_day=48,
                                               test_keys={})
        path = './power_experiments/power_data/1h_by_country_by_company/'
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
        if pd['fft_compression_rate']:
            pd['num_proj'] = int((pd['window_size']//2 + 1) / pd['fft_compression_rate'])
        elif pd['conv_fft_bins']:
            pd['num_proj'] = int((pd['window_size']//2 + 1) / pd['conv_fft_bins'])
        elif pd['fully_fft_comp']:
            pd['num_proj'] = int((pd['window_size']//2 + 1) / pd['fully_fft_comp'])
        else:
            pd['num_proj'] = int((pd['window_size']//2 + 1))
    elif pd['linear_reshape']:
        pd['num_proj'] = pd['step_size']/pd['downsampling']
    else:
        pd['num_proj'] = 1

    if pd['fft']:
        pd['epsilon'] = 1e-2
    else:
        pd['epsilon'] = None
    return pd


# set up the experiment list.
pd_lst = [fix_parameters(pd)]
lp_pd = copy.copy(pd)
lp_pd['fft_compression_rate'] = 4
pd_lst.append(fix_parameters(lp_pd))

lr_pd = copy.copy(pd)
lr_pd['fft'] = False
lr_pd['linear_reshape'] = True
lr_pd['downsampling'] = 1
pd_lst.append(fix_parameters(lr_pd))

lr_pd2 = copy.copy(pd)
lr_pd2['fft'] = False
lr_pd2['linear_reshape'] = True
lr_pd2['downsampling'] = 4
pd_lst.append(fix_parameters(lr_pd2))

time_pd = copy.copy(pd)
time_pd['fft'] = False
time_pd['linear_reshape'] = False
pd_lst.append(fix_parameters(time_pd))

for exp_no, cpd in enumerate(pd_lst):
    print('---------- Experiment', exp_no, 'of', len(pd_lst), '----------')
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    cpd['time_str'] = time_str
    print(cpd)
    pgraph = FFTpredictionGraph(cpd)
    param_str = '_' + cpd['cell_type'] + '_size_' + str(cpd['num_units']) + \
        '_fft_' + str(cpd['fft']) + \
        '_fm_' + str(fifteen_minute_sampling) + \
        '_bs_' + str(cpd['batch_size']) + \
        '_ps_' + str(cpd['pred_samples']) + \
        '_dis_' + str(cpd['discarded_samples']) + \
        '_lr_' + str(cpd['init_learning_rate']) + \
        '_dr_' + str(cpd['decay_rate']) + \
        '_ds_' + str(cpd['decay_steps']) + \
        '_sp_' + str(cpd['sample_prob']) + \
        '_rc_' + str(cpd['use_residuals']) + \
        '_pt_' + str(pgraph.total_parameters)

    if cpd['fft']:
        param_str += '_wf_' + str(cpd['window_function'])
        param_str += '_ws_' + str(cpd['window_size'])
        param_str += '_ol_' + str(cpd['overlap'])
        param_str += '_ffts_' + str(cpd['step_size'])
        param_str += '_fftp_' + str(cpd['fft_pred_samples'])
        param_str += '_fl_' + str(cpd['freq_loss'])
        param_str += '_eps_' + str(cpd['epsilon'])
        if cpd['fft_compression_rate']:
            param_str += '_fftcr_' + str(cpd['fft_compression_rate'])

    if cpd['stiefel']:
        param_str += '_stfl'

    if cpd['linear_reshape']:
        param_str += '_linre'
        assert cpd['conv_fft_bins'] is None

    if cpd['downsampling'] > 1:
            param_str += '_downs_' + str(cpd['downsampling'])

    if cpd['conv_fft_bins']:
        param_str += '_conv_fft_bins_' + str(cpd['conv_fft_bins'])
        assert cpd['linear_reshape'] is False

    # do each of the experiments in the parameter dictionary list.
    print(param_str)
    # icpdb.set_trace()
    summary_writer = tf.summary.FileWriter(cpd['base_dir'] + cpd['time_str'] + param_str,
                                           graph=pgraph.graph)
    # dump the parameters
    with open(cpd['base_dir'] + cpd['time_str'] + param_str + '/param.pkl', 'wb') as file:
        pickle.dump(cpd, file)

    param_summary = tf.Summary.Value(tag='network_weights', simple_value=pgraph.total_parameters)
    param_summary = tf.Summary(value=[param_summary])
    summary_writer.add_summary(param_summary, global_step=0)

    test_data = cpd['power_handler'].get_test_set()
    # train this.
    gpu_options = tf.GPUOptions(visible_device_list=str(cpd['GPUs'])[1:-1])
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=gpu_options)
    with tf.Session(graph=pgraph.graph, config=config) as sess:
        print('initialize....')
        pgraph.init_op.run()
        for e in range(0, cpd['epochs']):
            training_batches = cpd['power_handler'].get_training_set()

            def organize_into_batches(batches):
                batch_total = len(batches)
                split_into = int(batch_total/cpd['batch_size'])
                batch_lst = np.array_split(np.stack(batches),
                                           split_into)
                return batch_lst

            batch_lst = organize_into_batches(training_batches)

            for it, batch in enumerate(batch_lst):
                start = time.time()
                # array_split add elements here and there, the true data is at 1
                feed_dict = {pgraph.data_nd: np.reshape(
                    batch[:cpd['batch_size'], :, :, 1],
                    [cpd['batch_size'], cpd['context_days']*cpd['samples_per_day'], 1])}

                np_loss, summary_to_file, np_global_step, _, \
                    datenc_np, datdec_np, decout_np, \
                    datand_np = \
                    sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                              pgraph.training_op, pgraph.data_encoder_time,
                              pgraph.data_decoder_time,
                              pgraph.decoder_out, pgraph.data_nd],
                             feed_dict=feed_dict)
                stop = time.time()
                if it % 100 == 0:
                    print('it: %5d, loss: %5.6f, time: %1.2f [s], epoch: %3d of %3d'
                          % (it, np_loss, stop-start, e, cpd['epochs']))

                summary_writer.add_summary(summary_to_file, global_step=np_global_step)

                if it % 100 == 0:
                    plt.figure()
                    plt.plot(
                        decout_np[0, cpd['discarded_samples']:cpd['pred_samples'], 0])
                    plt.plot(
                        datdec_np[0, cpd['discarded_samples']:cpd['pred_samples'], 0])
                    plt.plot(
                        np.abs(
                            decout_np[0, cpd['discarded_samples']:cpd['pred_samples'], 0]
                            - datdec_np[0, cpd['discarded_samples']:cpd['pred_samples'],
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
            pgraph.saver.save(sess, cpd['base_dir'] + time_str +
                              param_str + '/weights/cpk',
                              global_step=np_global_step)
            # do a test run.
            print('test run ', end='')
            mse_lst_net = []
            mse_lst_off = []
            time_lst = []

            test_batch_lst = organize_into_batches(test_data)
            for test_batch in test_batch_lst:
                gt = np.reshape(test_batch[:cpd['batch_size'], :, :, 1],
                                [cpd['batch_size'],
                                 cpd['context_days']*cpd['samples_per_day'],
                                 1])
                official_pred = np.reshape(test_batch[:cpd['batch_size'], :, :, 0],
                                           [cpd['batch_size'],
                                            cpd['context_days']
                                            * cpd['samples_per_day'], 1])
                feed_dict = {pgraph.data_nd: gt}
                test_start = time.time()
                if cpd['fft']:
                    np_loss, np_global_step, \
                        datenc_np, datdec_np, decout_np, \
                        datand_np, window_np = \
                        sess.run([pgraph.loss, pgraph.global_step,
                                  pgraph.data_encoder_time,
                                  pgraph.data_decoder_time,
                                  pgraph.decoder_out, pgraph.data_nd, pgraph.window],
                                 feed_dict=feed_dict)
                else:
                    np_loss, np_global_step, \
                        datenc_np, datdec_np, decout_np, \
                        datand_np = \
                        sess.run([pgraph.loss, pgraph.global_step,
                                  pgraph.data_encoder_time,
                                  pgraph.data_decoder_time,
                                  pgraph.decoder_out, pgraph.data_nd],
                                 feed_dict=feed_dict)
                test_stop = time.time() - test_start
                net_pred = decout_np[:, :, 0]*cpd['power_handler'].std + cpd['power_handler'].mean
                official_pred = official_pred[:, -cpd['pred_samples']:, 0]
                gt = gt[:, -cpd['pred_samples']:, 0]
                mse_lst_net.append(
                    np.mean((gt[:, cpd['discarded_samples']:]
                             - net_pred[:, cpd['discarded_samples']:cpd['pred_samples']])
                            ** 2))
                mse_lst_off.append(
                    np.mean((
                        gt[:, cpd['discarded_samples']:]
                        - official_pred[:, cpd['discarded_samples']:cpd['pred_samples']])
                        ** 2))
                time_lst.append(test_stop)
                print('.', end='')
            mse_net = np.mean(np.array(mse_lst_net))
            mse_off = np.mean(np.array(mse_lst_off))
            print()
            print('epoch: %5d,  test mse_net: %5.2f, test mse_off: %5.2f' %
                  (e, mse_net, mse_off))
            print('baseline difference: %5.2f' % (mse_off-mse_net))
            print('test_runtime', np.mean(time_lst))

            # add to tensorboard
            time_test_summary = tf.Summary.Value(tag='runtime_net_test', simple_value=np.mean(time_lst))
            time_test_summary = tf.Summary(value=[time_test_summary])
            mse_net_summary = tf.Summary.Value(tag='mse_net_test', simple_value=mse_net)
            mse_net_summary = tf.Summary(value=[mse_net_summary])
            mse_off_summary = tf.Summary.Value(tag='mse_off_test', simple_value=mse_off)
            mse_off_summary = tf.Summary(value=[mse_off_summary])
            summary_writer.add_summary(mse_net_summary, global_step=np_global_step)
            summary_writer.add_summary(mse_off_summary, global_step=np_global_step)
            summary_writer.add_summary(time_test_summary, global_step=np_global_step)
            mse_diff_summary = tf.Summary.Value(tag='mse_net_off_diff',
                                                simple_value=mse_off-mse_net)
            mse_diff_summary = tf.Summary(value=[mse_diff_summary])
            summary_writer.add_summary(mse_diff_summary, global_step=np_global_step)
            if cpd['fft']:
                # window plot in tensorboard.
                plt.figure()
                plt.plot(window_np)
                plt.title(cpd['window_function'])
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png')
                buf2.seek(0)
                summary_image2 = tf.Summary.Image(
                    encoded_image_string=buf2.getvalue(),
                    height=int(plt.rcParams["figure.figsize"][0]*100),
                    width=int(plt.rcParams["figure.figsize"][1]*100))
                summary_image2 = tf.Summary.Value(tag=cpd['window_function'],
                                                  image=summary_image2)
                summary_image2 = tf.Summary(value=[summary_image2])
                summary_writer.add_summary(summary_image2, global_step=np_global_step)
                plt.close()
                buf.close()
