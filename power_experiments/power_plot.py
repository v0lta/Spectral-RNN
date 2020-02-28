import pickle
import numpy as np
import scipy.signal as scisig
import matplotlib.pyplot as plt
import tikzplotlib as tikz

import sys
sys.path.insert(0, "../")
from tensorboard_plot_helper_module import plot_logs, return_logs
window_size = 0

# 60 1h pred plots 
if 1:
    plot_path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/power_pred_60d_1h'
    logs = return_logs(plot_path, window_size, vtag='mse_net_test')
    stop_at = len(logs[0][0][0])
    plt.plot(logs[2][0][0][:stop_at], logs[2][0][1][:stop_at], label='fft-gru-tukey')
    plt.plot(logs[9][0][0][:stop_at], logs[9][0][1][:stop_at], label='reshape-gru')
    plt.plot(logs[3][0][0][:stop_at], logs[3][0][1][:stop_at], label='time-gru')
    plt.ylabel('time-domain mean squared error [mw]')
    plt.xlabel('weight-updates')
    plt.legend()
    plt.ylim(0, 19000000)
    plt.savefig('power_pred_60d_steps.pdf')
    # tikz.save('power_pred_60d_steps.tex')
    plt.show()

    plt.plot(logs[2][0][0][:stop_at], logs[2][0][1][:stop_at], label='fft-gru-tukey')
    plt.plot(logs[9][0][0][:stop_at], logs[9][0][1][:stop_at], label='reshape-gru')
    plt.ylabel('time-domain mean squared error [mw]')
    plt.xlabel('weight-updates')
    plt.legend()
    plt.ylim(0, 1800000)
    # plt.savefig('power_pred_60d_steps_notime.pdf')
    tikz.save('power_pred_60d_steps_notime.tex')
    plt.show()

    # time duration plot.
    plot_path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/power_pred_60d_1h'
    logs = return_logs(plot_path, window_size, vtag='mse_net_test')
    stop_at = len(logs[0][0][0])
    plt.plot(logs[2][0][0][:stop_at]/logs[2][0][0][-1]*34,
             logs[2][0][1][:stop_at], label='fft-gru-tukey')
    plt.plot(logs[9][0][0][:stop_at]/logs[9][0][0][-1]*44,
             logs[9][0][1][:stop_at], label='reshape-gru')
    plt.plot(logs[3][0][0][:stop_at]/logs[3][0][0][-1]*(13*60)+13,
             logs[3][0][1][:stop_at], label='time-gru')
    plt.ylabel('time-domain mean squared error [mw]')
    plt.xlabel('time [min]')
    plt.legend()
    plt.ylim(0, 19000000)
    plt.savefig('power_pred_60d_time.pdf')
    # tikz.save('power_pred_60d_time.tex')
    plt.show()


    # plot_path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/power_pred_60d_1h'
    # logs = return_logs(plot_path, window_size, vtag='mse_net_test')
    # stop_at = len(logs[0][0][0])
    # plt.plot(logs[9][0][0][:stop_at], logs[9][0][1][:stop_at], label='fft-cgRNN-tukey')
    # plt.plot(logs[6][0][0][:stop_at], logs[6][0][1][:stop_at], label='fft-gru-tukey')
    # plt.plot(logs[22][0][0][:stop_at], logs[22][0][1][:stop_at], label='reshape-gru')
    # plt.plot(logs[5][0][0][:stop_at], logs[5][0][1][:stop_at], label='time-gru')
    # plt.ylabel('time-domain mean squared error')
    # plt.xlabel('weight-updates')
    # plt.legend()
    # plt.ylim(0, 3000000)
    # plt.show()
    # # plt.savefig('power_pred_steps.pdf')
    # # tikz.save('power_pred_steps.tex')
    # plt.clf()
    # plt.cla()
    # plt.close()

