import copy
from src.mackey_glass_generator import MackeyGenerator
from src.run_synthetics import run_experiemtns

pd = {}
pd['base_dir'] = 'log/cvpr_workshop_synthetic_5/'
pd['cell_type'] = 'gru'
pd['num_units'] = 64
pd['sample_prob'] = 1.0
pd['init_learning_rate'] = 0.001
pd['decay_rate'] = 0.9
pd['decay_steps'] = 1000
pd['iterations'] = 30000
# pd['iterations'] = 2
pd['GPUs'] = [0]
pd['batch_size'] = 12
pd['window_function'] = 'learned_gaussian'
pd['freq_loss'] = None
pd['use_residuals'] = True
pd['linear_reshape'] = False
pd['downsampling'] = 1  # set to 1 to turn this off.
pd['stiefel'] = False
# data parameters
pd['tmax'] = 512
pd['delta_t'] = 0.1
pd['input_samples'] = int(pd['tmax']/pd['delta_t'])
pd['generator'] = MackeyGenerator(pd['batch_size'],
                                  pd['tmax'], pd['delta_t'],
                                  restore_and_plot=False)
pd['window_size'] = 128
pd['pred_samples'] = 2560
pd['discarded_samples'] = 0

pd['fft'] = False
pd['overlap'] = int(pd['window_size']*0.5)
pd['step_size'] = pd['window_size'] - pd['overlap']
pd['fft_pred_samples'] = pd['pred_samples'] // pd['step_size'] + 1
pd['fft_compression_rate'] = None
# don't touch!
pd['conv_fft_bins'] = None
pd['fully_fft_comp'] = None


def adjust_parameters(pd):
    if pd['fft']:
        if pd['fft_compression_rate']:
            pd['num_proj'] = int((pd['window_size']//2 + 1) / pd['fft_compression_rate'])
        else:
            pd['num_proj'] = int((pd['window_size']//2 + 1))
    elif pd['linear_reshape']:
        pd['num_proj'] = pd['step_size']/pd['downsampling']
    else:
        pd['num_proj'] = 1

    if pd['fft']:
        if pd['window_function'] == 'boxcar':
            pd['epsilon'] = 0.0
        else:
            pd['epsilon'] = 1e-3
    else:
        pd['epsilon'] = None
    return pd


pd = adjust_parameters(pd)
pd2 = copy.copy(pd)
pd2['linear_reshape'] = True
pd2['downsampling'] = 1
pd2 = adjust_parameters(pd2)

pd3 = copy.copy(pd)
pd3['linear_reshape'] = True
pd3['downsampling'] = 8
pd3 = adjust_parameters(pd3)

pd4 = copy.copy(pd)
pd4['linear_reshape'] = True
pd4['downsampling'] = 16
pd4 = adjust_parameters(pd4)

pd5 = copy.copy(pd)
pd5['fft'] = True
pd5['fft_compression_rate'] = 1
pd5 = adjust_parameters(pd5)

pd6 = copy.copy(pd)
pd6['fft'] = True
pd6['fft_compression_rate'] = 8
pd6 = adjust_parameters(pd6)

pd7 = copy.copy(pd)
pd7['fft'] = True
pd7['fft_compression_rate'] = 16
pd7 = adjust_parameters(pd7)

pd8 = copy.copy(pd)
pd8['fft'] = True
pd8['cell_type'] = 'cgRNN'
pd8 = adjust_parameters(pd8)

pd9 = copy.copy(pd)
pd9['fft'] = True
pd9['cell_type'] = 'cgRNN'
pd9['num_units'] = 54
pd9 = adjust_parameters(pd9)

pd10 = copy.copy(pd)
pd10['fft'] = True
pd10['cell_type'] = 'cgRNN'
pd10['num_units'] = 32
pd10 = adjust_parameters(pd10)

pd11 = copy.copy(pd)
pd11['fft'] = True
pd11['cell_type'] = 'cgRNN'
pd11['num_units'] = 32
pd11 = adjust_parameters(pd11)

pd12 = copy.copy(pd)
pd12['fft'] = True
pd12['cell_type'] = 'cgRNN'
pd12['freq_loss'] = 'complex_square'
pd12 = adjust_parameters(pd12)

pd13 = copy.copy(pd)
pd13['fft'] = True
pd13['fft_compression_rate'] = 32
pd13 = adjust_parameters(pd13)

pd14 = copy.copy(pd)
pd14['linear_reshape'] = True
pd14['downsampling'] = 32
pd14 = adjust_parameters(pd14)


# define a list of experiments.
lpd_lst = [pd, pd2, pd3, pd4, pd5, pd6, pd7, pd8, pd9, pd10, pd11, pd12, pd13, pd14]
# lpd_lst = [pd13, pd14]
run_experiemtns(lpd_lst)
