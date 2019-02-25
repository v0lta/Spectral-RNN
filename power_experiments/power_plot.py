import pickle
import numpy as np
import scipy.signal as scisig
import matplotlib.pyplot as plt

import matplotlib2tikz as tikz

import sys
sys.path.insert(0, "../")
from tensorboard_plot_helper_module import plot_logs, return_logs

window_size = 36
stop = 3128000
# extract a logfile plot.
plot_path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/power_pred_logs_test3/'
logs = return_logs(plot_path, window_size, vtag='time_loss')
stop_at = len(logs[0][0][0])
plt.plot(logs[0][0][0][:stop_at], logs[0][0][1][:stop_at], label='fft-cgRNN')
plt.plot(logs[2][0][0][:stop_at], logs[2][0][1][:stop_at], label='GRU')

plt.ylim([0.0, 0.025])
plt.ylabel('time-domain mean squared error')
plt.xlabel('weight-updates')
plt.legend()
# plt.savefig('power_pred_steps.pdf')
tikz.save('power_pred_steps.tex')
plt.clf()
plt.cla()
plt.close()


plt.plot((logs[0][0][0][:stop_at]/logs[0][0][0][stop_at - 1])*(60+30),
         logs[0][0][1][:stop_at], label='fft-cgRNN')
plt.plot((logs[2][0][0][:stop_at]/logs[0][0][0][stop_at - 1])*(60*9),
         logs[2][0][1][:stop_at], label='GRU')
plt.ylim([0.0, 0.025])
plt.ylabel('time-domain mean squared error')
plt.xlabel('time [min]')
plt.legend()
# plt.savefig('power_pred_time.pdf')
tikz.save('power_pred_time.tex')
plt.close()
