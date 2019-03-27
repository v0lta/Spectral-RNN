import pickle
import numpy as np
import scipy.signal as scisig
import matplotlib.pyplot as plt
import matplotlib2tikz as tikz

import sys
sys.path.insert(0, "../")
from tensorboard_plot_helper_module import plot_logs, return_logs
window_size = 0

# 1d 15min pred plots test loss
plot_path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/power_pred_1d_15_min'
logs = return_logs(plot_path, window_size, vtag='mse_net_test')
stop_at = len(logs[0][0][0])
plt.plot(logs[9][0][0][:stop_at], logs[9][0][1][:stop_at], label='fft-cgRNN-tukey')
plt.plot(logs[6][0][0][:stop_at], logs[6][0][1][:stop_at], label='fft-gru-tukey')
plt.plot(logs[22][0][0][:stop_at], logs[22][0][1][:stop_at], label='reshape-gru')
plt.plot(logs[5][0][0][:stop_at], logs[5][0][1][:stop_at], label='time-gru')
plt.ylabel('time-domain mean squared error')
plt.xlabel('weight-updates')
plt.legend()
plt.ylim(0, 3000000)
# plt.show()
# plt.savefig('power_pred_steps.pdf')
tikz.save('power_pred_15min_1d_test.tex')
plt.clf()
plt.cla()
plt.close()

# 1d 15 min day-ahead vs. official.
window_size = 3
logs = return_logs(plot_path, window_size, vtag='mse_net_off_diff')
stop_at = len(logs[0][0][0])
plt.plot(logs[9][0][0][:stop_at], logs[9][0][1][:stop_at], label='fft-cgRNN-tukey')
plt.plot(logs[6][0][0][:stop_at], logs[6][0][1][:stop_at], label='fft-gru-tukey')
plt.plot(logs[22][0][0][:stop_at], logs[22][0][1][:stop_at], label='reshape-gru')
plt.plot(logs[5][0][0][:stop_at], logs[5][0][1][:stop_at], label='time-gru')
plt.ylabel('time-domain mean squared error')
plt.xlabel('weight-updates')
plt.grid()
plt.ylim(-200000, 200000)
plt.legend()
# plt.show()
# plt.savefig('power_pred_steps.pdf')
tikz.save('power_pred_15min_1d_offcomp.tex')
plt.clf()
plt.cla()
plt.close()

# plt.plot((logs[0][0][0]/logs[0][0][0][-1]*(25)),
#          logs[0][0][1], label='fft-gru-tukey')
# plt.plot((logs[1][0][0]/logs[0][0][0][-1])*(60+20),
#          logs[1][0][1], label='fft-cgRNN-tukey')
# plt.plot((logs[3][0][0]/logs[3][0][0][-1])*(60*11+4),
#          logs[2][0][1], label='GRU')

# plt.ylim([0.0, 0.025])
# plt.ylabel('time-domain mean squared error')
# plt.xlabel('time [min]')
# plt.legend()
# plt.show()
# # plt.savefig('power_pred_time.pdf')
# # tikz.save('power_pred_time.tex')
# plt.close()
