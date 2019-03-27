import io
import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mackey_glass_generator import MackeyGenerator
from power_experiments.prediction_graph import FFTpredictionGraph
from IPython.core.debugger import Tracer
debug_here = Tracer()

pd = {}

pd['base_dir'] = 'logs/mackey2/'
pd['cell_type'] = 'cgRNN'
pd['num_units'] = 128
pd['sample_prob'] = 1.0
pd['init_learning_rate'] = 0.004
pd['decay_rate'] = 0.95
pd['decay_steps'] = 390

pd['iterations'] = 8000
pd['GPUs'] = [6]
pd['batch_size'] = 100
# pd['window_function'] = 'hann'
# pd['window_function'] = 'learned_tukey'
# pd['window_function'] = 'learned_plank'
# pd['window_function'] = 'learned_gaussian'
# pd['window_function'] = 'learned_gauss_plank'
pd['freq_loss'] = None
pd['use_residuals'] = True
pd['fft'] = True
pd['linear_reshape'] = False
pd['stiefel'] = False

# data parameters
pd['tmax'] = 1024
pd['delta_t'] = 1.0
pd['input_samples'] = int(pd['tmax']/pd['delta_t'])
pd['generator'] = MackeyGenerator(pd['batch_size'],
                                  pd['tmax'], pd['delta_t'],
                                  restore_and_plot=False)

pd['window_size'] = 128
pd['pred_samples'] = 512
pd['discarded_samples'] = 0
pd['overlap'] = int(pd['window_size']*0.5)
pd['step_size'] = pd['window_size'] - pd['overlap']
pd['fft_pred_samples'] = pd['pred_samples'] // pd['step_size'] + 1

if pd['fft']:
    pd['num_proj'] = int(pd['window_size']//2 + 1)
elif pd['linear_reshape']:
    pd['num_proj'] = pd['step_size']
else:
    pd['num_proj'] = 1

if pd['fft']:
    if pd['window_function'] == 'boxcar':
        pd['epsilon'] = 0.0
    else:
        pd['epsilon'] = 1e-2
else:
    pd['epsilon'] = None

pgraph = FFTpredictionGraph(pd, generator=pd['generator'])

print(pgraph.total_parameters)
# ipdb.set_trace()
time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
pd['time_str'] = time_str
param_str = '_' + pd['cell_type'] + '_size_' + str(pd['num_units']) + \
    '_fft_' + str(pd['fft']) + \
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

if pd['linear_reshape']:
    param_str += '_linre'

print(param_str)
# ipdb.set_trace()
summary_writer = tf.summary.FileWriter(pd['base_dir'] + pd['time_str'] + param_str,
                                       graph=pgraph.graph)
# dump the parameters
with open(pd['base_dir'] + pd['time_str'] + param_str + '/param.pkl', 'wb') as file:
    pickle.dump(pd, file)


# train this.
gpu_options = tf.GPUOptions(visible_device_list=str(pd['GPUs'])[1:-1])
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)
with tf.Session(graph=pgraph.graph, config=config) as sess:
    print('initialize....')
    pgraph.init_op.run()
    for it in range(pd['iterations']):
        start = time.time()
        # array_split add elements here and there, the true data is at 1
        if not pd['fft']:
            np_loss, summary_to_file, np_global_step, _, \
                datenc_np, encout_np, datdec_np, decout_np, \
                datand_np = \
                sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                          pgraph.training_op, pgraph.data_encoder_gt, pgraph.encoder_out,
                          pgraph.data_decoder_gt, pgraph.decoder_out, pgraph.data_nd])
        else:
            np_loss, summary_to_file, np_global_step, _, \
                datenc_np, encout_np, datdec_np, decout_np, \
                datand_np, window_np = \
                sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                          pgraph.training_op, pgraph.data_encoder_gt, pgraph.encoder_out,
                          pgraph.data_decoder_gt, pgraph.decoder_out, pgraph.data_nd,
                          pgraph.window])

        stop = time.time()
        if it % 5 == 0:
            print('it: %5d, loss: %5.6f, time: %1.2f [s]'
                  % (it, np_loss, stop-start))

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
            summary_writer.add_summary(
                summary_image, global_step=np_global_step)
            plt.close()
            buf.close()
            # add to tensorboard
            if pd['fft']:
                # add fft window plot in tensorboard.
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
                summary_writer.add_summary(
                    summary_image2, global_step=np_global_step)
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
