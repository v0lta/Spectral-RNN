# fourier-prediction
Source code for the paper: 
https://arxiv.org/pdf/1812.05645.pdf

To reproduce the synthetic experiments adjust the parameters in `synthetic_signal_test.py` and run it. The results may be plotted by using `synthetic_signal_plot.py` after adjusting the logfile path in that file.

To run the power prediction experiments download:

http://www.wolter.tech/wordpress/wp-content/uploads/2019/02/power_data.zip

And extract it in the power experiments folder. Then run `power_load_pred_exp.py`
with the desired parameters.

The experiments in music_exp worked and outperformed https://arxiv.org/abs/1705.09792 but not  https://arxiv.org/pdf/1711.04845.pdf. The experiments in human motion exp did work well numerically, but the resulting motions did look jumpy and choppy.

The code has been tested using Tensorflow 1.10.0 but should also run in 1.13.

To recreate the synthetic experiments run synthetic_signal_test.py