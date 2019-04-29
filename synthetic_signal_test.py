import io
import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mackey_glass_generator import MackeyGenerator
from power_experiments.prediction_graph import FFTpredictionGraph
from IPython.core.debugger import Pdb
debug_here = Pdb().set_trace

pd = {}

pd['base_dir'] = 'logs/mackey2/explore4/'
pd['cell_type'] = 'cgRNN'
pd['num_units'] = 128
pd['sample_prob'] = 1.0
pd['init_learning_rate'] = 0.001
pd['decay_rate'] = 0.9
pd['decay_steps'] = 1000

pd['iterations'] = 20000
pd['GPUs'] = [0]
pd['batch_size'] = 100
pd['window_function'] = 'hann'
# pd['window_function'] = 'learned_tukey'
# pd['window_function'] = 'learned_plank'
# pd['window_function'] = 'learned_gaussian'
# pd['window_function'] = 'learned_gauss_plank'
pd['freq_loss'] = None
pd['use_residuals'] = True
pd['fft'] = False
pd['linear_reshape'] = True
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
pd['fft_compression_rate'] = None

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
        pd['epsilon'] = 1e-3
else:
    pd['epsilon'] = None

# define a list of experiments.
lpd_lst = []

# cell_size_loop
fft_loop = pd['fft']
if fft_loop:
    assert pd['fft'] is True
    assert pd['linear_reshape'] is False
    for num_units in [4, 8, 16, 32, 64, 128]:
        # window_loop
        for window in ['hann', 'learned_tukey', 'learned_gaussian', 'boxcar']:
            # compression loop:
            for compression in [None, 2, 4, 8, 16, 32]:
                # cell_type loop:
                for cell_type in ['gru', 'cgRNN']:
                    # residual loop
                    for use_residuals in [True, False]:
                        cpd = pd.copy()
                        cpd['window_function'] = window
                        cpd['fft_compression_rate'] = compression
                        cpd['num_units'] = num_units
                        cpd['cell_type'] = cell_type
                        cpd['use_residuals'] = use_residuals
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
    for num_units in [4, 8, 16, 32, 64, 128]:
        # cell_type loop:
        for cell_type in ['gru', 'cgRNN']:
            # residual loop
            for use_residuals in [True, False]:
                cpd = pd.copy()
                cpd['num_units'] = num_units
                cpd['cell_type'] = cell_type
                cpd['use_residuals'] = use_residuals
                lpd_lst.append(cpd)

time_loop = not (pd['linear_reshape'] or pd['fft'])
if time_loop:
    assert pd['fft'] is False
    assert pd['linear_reshape'] is False
    for num_units in [4, 6, 8, 16, 32, 64, 128]:
        # cell_type loop:
        for cell_type in ['gru', 'cgRNN']:
            # residual loop
            for use_residuals in [True, False]:
                cpd = pd.copy()
                cpd['num_units'] = num_units
                cpd['cell_type'] = cell_type
                cpd['use_residuals'] = use_residuals
                lpd_lst.append(cpd)


for exp_no, lpd in enumerate(lpd_lst):
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    lpd['time_str'] = time_str
    pgraph = FFTpredictionGraph(lpd, generator=lpd['generator'])
    param_str = time_str + '_' + lpd['cell_type'] + '_size_' + str(lpd['num_units']) + \
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

    print('---------- Experiment', exp_no, 'of', len(lpd_lst), '----------')
    print(param_str)
    # print(lpd)
    summary_writer = tf.summary.FileWriter(lpd['base_dir'] + lpd['time_str'] + param_str,
                                           graph=pgraph.graph)
    # dump the parameters
    with open(lpd['base_dir'] + lpd['time_str'] + param_str + '/param.pkl', 'wb') as file:
        pickle.dump(lpd, file)

    gpu_options = tf.GPUOptions(visible_device_list=str(lpd['GPUs'])[1:-1])
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=gpu_options)
    with tf.Session(graph=pgraph.graph, config=config) as sess:
        print('initialize....')
        pgraph.init_op.run()
        for it in range(lpd['iterations']):
            start = time.time()
            # array_split add elements here and there, the true data is at 1
            if not lpd['fft']:
                np_loss, summary_to_file, np_global_step, _, \
                    datenc_np, encout_np, datdec_np, decout_np, \
                    datand_np = \
                    sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                              pgraph.training_op, pgraph.data_encoder_gt,
                              pgraph.encoder_out, pgraph.data_decoder_gt,
                              pgraph.decoder_out, pgraph.data_nd])
            else:
                np_loss, summary_to_file, np_global_step, _, \
                    datenc_np, encout_np, datdec_np, decout_np, \
                    datand_np, window_np = \
                    sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                              pgraph.training_op, pgraph.data_encoder_gt,
                              pgraph.encoder_out, pgraph.data_decoder_gt,
                              pgraph.decoder_out, pgraph.data_nd,
                              pgraph.window])

            stop = time.time()
            if it % 100 == 0:
                print('it: %5d, loss: %5.6f, time: %1.2f [s]'
                      % (it, np_loss, stop-start))

            summary_writer.add_summary(summary_to_file, global_step=np_global_step)

            if it % 100 == 0:
                plt.figure()
                plt.plot(decout_np[0, lpd['discarded_samples']:lpd['pred_samples'], 0])
                plt.plot(datdec_np[0, lpd['discarded_samples']:lpd['pred_samples'], 0])
                plt.plot(
                    np.abs(decout_np[0, lpd['discarded_samples']:lpd['pred_samples'], 0]
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
                summary_writer.add_summary(
                    summary_image, global_step=np_global_step)
                plt.close()
                buf.close()
                # add to tensorboard
                if lpd['fft']:
                    # add fft window plot in tensorboard.
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
                    summary_writer.add_summary(
                        summary_image2, global_step=np_global_step)
                    plt.close()
                    buf.close()

        # epoch done. Save.
        print('Saving a copy.')
        pgraph.saver.save(sess, lpd['base_dir'] + time_str + param_str + '/weights/cpk',
                          global_step=np_global_step)

