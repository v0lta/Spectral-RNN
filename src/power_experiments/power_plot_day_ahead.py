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
plot_path = '/home/moritz/infcuda/fft_pred_networks/power_experiments/\
log/power_pred_1d_15_min/paper_exp2/'
logs = return_logs(plot_path, window_size, vtag='mse_net_test')
stop_at = len(logs[0][0][0])
plt.semilogy(logs[0][0][0][:stop_at], logs[0][0][1][:stop_at], label='time')
plt.semilogy(logs[2][0][0][:stop_at], logs[2][0][1][:stop_at], label='reshape')
plt.semilogy(logs[4][0][0][:stop_at], logs[4][0][1][:stop_at], label='fft')

# plot the entensoe.eu error bar.
eu = np.ones([stop_at])*981186.28
plt.semilogy(eu, label='entsoe.eu')

plt.ylabel('mean squared error [MW]')
plt.xlabel('epochs')
plt.legend()
plt.ylim(0, 3000000)

# plt.savefig('power_pred_steps.pdf')
# tikz.save('power_pred_15min_1d_test.tex', standalone=True)
plt.show()
plt.clf()
plt.cla()
plt.close()
