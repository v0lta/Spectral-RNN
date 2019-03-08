import sys
import io
import time
import pickle
import tensorflow as tf
import numpy as np
import scipy.signal as scisig
import matplotlib.pyplot as plt
from power_data_handler import PowerDataHandler, MergePowerHandler
from prediction_graph import FFTpredictionGraph
from IPython.core.debugger import Tracer
debug_here = Tracer()


fifteen_minute_sampling = False
pd = {}

pd['prediction_days'] = 30
if pd['prediction_days'] > 1:
    pd['context_days'] = pd['prediction_days']*2
else:
    pd['context_days'] = 15
pd['base_dir'] = 'log/test/'
pd['cell_type'] = 'gru'
pd['num_units'] = 222
pd['sample_prob'] = 1.0
pd['init_learning_rate'] = 0.004
pd['decay_rate'] = 0.95


pd['epochs'] = 1
pd['GPUs'] = [7]
pd['batch_size'] = 100
# window_function = 'hann'
pd['window_function'] = 'learned_tukey'
pd['freq_loss'] = None
pd['use_residuals'] = True
pd['fft'] = False
pd['stiefel'] = False

if fifteen_minute_sampling is True:
    pd['decay_steps'] = 390
else:
    pd['decay_steps'] = 455

if fifteen_minute_sampling is True:
    pd['samples_per_day'] = 96
    path = './power_data/15m_by_country_by_company/'
    pd['power_handler'] = PowerDataHandler(path,
                                           pd['context_days'])
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
    pd['window_size'] = int(pd['samples_per_day']*4)
    pd['pred_samples'] = int(pd['prediction_days']*pd['samples_per_day'])
    pd['discarded_samples'] = 0
else:
    pd['pred_samples'] = int(pd['samples_per_day']*1.5)
    pd['discarded_samples'] = int(pd['samples_per_day']*0.5)
    pd['window_size'] = int(pd['samples_per_day'])

pd['overlap'] = int(pd['window_size']*0.75)
pd['step_size'] = pd['window_size'] - pd['overlap']
pd['fft_pred_samples'] = pd['pred_samples'] // pd['step_size'] + 1
pd['input_samples'] = pd['context_days']*pd['samples_per_day']

if pd['fft']:
    pd['num_proj'] = int(pd['window_size']//2 + 1)
else:
    pd['num_proj'] = 1

if pd['fft']:
    pd['epsilon'] = 1e-2
else:
    pd['epsilon'] = None


pgraph = FFTpredictionGraph(pd)

print(pgraph.total_parameters)
# ipdb.set_trace()
time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
pd['time_str'] = time_str
param_str = '_' + pd['cell_type'] + '_size_' + str(pd['num_units']) + \
    '_fft_' + str(pd['fft']) + \
    '_fm_' + str(fifteen_minute_sampling) + \
    '_bs_' + str(pd['batch_size']) + \
    '_ps_' + str(pd['pred_samples']) + \
    '_dis_' + str(pd['discarded_samples']) + \
    '_lr_' + str(pd['init_learning_rate']) + \
    '_dr_' + str(pd['decay_rate']) + \
    '_ds_' + str(pd['decay_steps']) + \
    '_sp_' + str(pd['sample_prob']) + \
    '_rc_' + str(pd['use_residuals']) + \
    '_pt_' + str(pgraph.total_parameters)

if pd['fft']:
    param_str += '_wf_' + str(pd['window_function'])
    param_str += '_ws_' + str(pd['window_size'])
    param_str += '_ol_' + str(pd['overlap'])
    param_str += '_ffts_' + str(pd['step_size'])
    param_str += '_fftp_' + str(pd['fft_pred_samples'])
    param_str += '_fl_' + str(pd['freq_loss'])
    param_str += '_eps_' + str(pd['epsilon'])

if pd['stiefel']:
    param_str += '_stfl'

print(param_str)
# ipdb.set_trace()
summary_writer = tf.summary.FileWriter(pd['base_dir'] + pd['time_str'] + param_str,
                                       graph=pgraph.graph)
# dump the parameters
with open(pd['base_dir'] + pd['time_str'] + param_str + '/param.pkl', 'wb') as file:
    pickle.dump(pd, file)


test_data = power_handler.get_test_set()
# train this.
gpu_options = tf.GPUOptions(visible_device_list=str(pd['GPUs'])[1:-1])
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)
with tf.Session(graph=pgraph.graph, config=config) as sess:
    print('initialize....')
    pgraph.init_op.run()
    for e in range(0, pd['epochs']):
        training_batches = power_handler.get_training_set()

        def organize_into_batches(batches):
            batch_total = len(batches)
            split_into = int(batch_total/pd['batch_size'])
            batch_lst = np.array_split(np.stack(batches),
                                       split_into)
            return batch_lst

        batch_lst = organize_into_batches(training_batches)
        # ipdb.set_trace()

        for it, batch in enumerate(batch_lst):
            start = time.time()
            # array_split add elements here and there, the true data is at 1
            feed_dict = {pgraph.data_nd: np.reshape(
                batch[:pd['batch_size'], :, :, 1],
                [pd['batch_size'], pd['context_days']*pd['samples_per_day'], 1])}

            np_loss, summary_to_file, np_global_step, _, \
                datenc_np, encout_np, datdec_np, decout_np, \
                datand_np = \
                sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                          pgraph.training_op, pgraph.data_encoder_gt, pgraph.encoder_out,
                          pgraph.data_decoder_gt, pgraph.decoder_out, pgraph.data_nd],
                         feed_dict=feed_dict)
            stop = time.time()
            if it % 5 == 0:
                print('it: %5d, loss: %5.6f, time: %1.2f [s], epoch: %3d of %3d'
                      % (it, np_loss, stop-start, e, pd['epochs']))

            summary_writer.add_summary(summary_to_file, global_step=np_global_step)

            if it % 100 == 0:
                plt.figure()
                plt.plot(decout_np[0, pd['discarded_samples']:pd['pred_samples'], 0])
                plt.plot(datdec_np[0, pd['discarded_samples']:pd['pred_samples'], 0])
                plt.plot(
                    np.abs(decout_np[0, pd['discarded_samples']:pd['pred_samples'], 0]
                           - datdec_np[0, pd['discarded_samples']:pd['pred_samples'], 0]))
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
        pgraph.saver.save(sess, pd['base_dir'] + time_str + param_str + '/weights/cpk',
                          global_step=np_global_step)
        # do a test run.
        print('test run ', end='')
        mse_lst_net = []
        mse_lst_off = []

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
        if pd['fft']:
            # window plot in tensorboard.
            plt.figure()
            plt.plot(window_np)
            plt.title(pd['window_function'])
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png')
            buf2.seek(0)
            summary_image2 = tf.Summary.Image(
                encoded_image_string=buf2.getvalue(),
                height=int(plt.rcParams["figure.figsize"][0]*100),
                width=int(plt.rcParams["figure.figsize"][1]*100))
            summary_image2 = tf.Summary.Value(tag=pd['window_function'],
                                              image=summary_image2)
            summary_image2 = tf.Summary(value=[summary_image2])
            summary_writer.add_summary(summary_image2, global_step=np_global_step)
            plt.close()
            buf.close()
