import pickle
import numpy as np
import scipy.signal as scisig
import matplotlib.pyplot as plt

import matplotlib2tikz as tikz

import sys
sys.path.insert(0, "../")
from tensorboard_plot_helper_module import plot_logs, return_logs
window_size = 156

# 960 plot
plot_path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/log/power_pred_logs_explore/960'
logs = return_logs(plot_path, window_size, vtag='time_loss')
stop_at = len(logs[0][0][0])
plt.plot(logs[0][0][0][:stop_at], logs[0][0][1][:stop_at], label='fft-gru-tukey')
plt.plot(logs[1][0][0][:stop_at], logs[1][0][1][:stop_at], label='fft-cgRNN-tukey')
plt.plot(logs[3][0][0][:stop_at], logs[3][0][1][:stop_at], label='gru')
plt.ylabel('time-domain mean squared error')
plt.xlabel('weight-updates')
plt.legend()
plt.ylim([0.0, 0.2])
plt.show()
# plt.savefig('power_pred_steps.pdf')
# tikz.save('power_pred_steps.tex')
plt.clf()
plt.cla()
plt.close()


plt.plot((logs[0][0][0]/logs[0][0][0][-1]*(25)),
         logs[0][0][1], label='fft-gru-tukey')
plt.plot((logs[1][0][0]/logs[0][0][0][-1])*(60+20),
         logs[1][0][1], label='fft-cgRNN-tukey')
plt.plot((logs[3][0][0]/logs[3][0][0][-1])*(60*11+4),
         logs[2][0][1], label='GRU')

plt.ylim([0.0, 0.025])
plt.ylabel('time-domain mean squared error')
plt.xlabel('time [min]')
plt.legend()
plt.show()
# plt.savefig('power_pred_time.pdf')
# tikz.save('power_pred_time.tex')
plt.close()
