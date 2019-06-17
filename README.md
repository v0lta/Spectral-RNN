FOURIER RNNS FOR SEQUENCE PREDICTION
------------------------------------
Source code for the paper: 
https://arxiv.org/pdf/1812.05645.pdf

To reproduce the synthetic experiments adjust the parameters in `synthetic_signal_test.py` and run it. The results may be plotted by using `synthetic_signal_plot.py` after adjusting the logfile path in that file.

To run the power prediction experiments download:

http://www.wolter.tech/wordpress/wp-content/uploads/2019/02/power_data.zip

And extract it in the `power_experiments` folder. Then run `power_load_pred_exp.py`
with the desired parameters.

The code has been tested using Tensorflow 1.10.0 but should also run in 1.13.

To recreate the synthetic experiments run `synthetic_signal_test.py` after adjusting the hyperparameters as described in the paper.
