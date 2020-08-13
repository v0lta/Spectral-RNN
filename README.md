Sequence Prediction using Spectral RNNs
------------------------------------
Source code for the paper: 
https://arxiv.org/pdf/1812.05645.pdf if you find this work useful please consider citing the paper.
The code has been tested using Tensorflow 1.10.0 but should also run in 1.13.

Experiments:
------------

 - To recreate the synthetic experiments run `synthetic_signal_test.py` after adjusting the hyperparameters as described in the paper.
    The results may be plotted by using `synthetic_signal_plot.py` after adjusting the logfile path in that file.

 - To run the power prediction experiments download:
   http://www.wolter.tech/wordpress/wp-content/uploads/2019/02/power_data.zip
   And extract it in the `power_experiments` folder. Then run `power_load_pred_exp.py`
   with the desired parameters.

 - The mocap experiments use the human3.6m data set avaialble at
   http://vision.imar.ro/human3.6m/
   After downloading and pickeling the data run mocap_train_exp.py to repeat the experiments
   in the paper.
 
 
Demo Video:
-----------

https://www.wolter.tech/wordpress/wp-content/uploads/2019/12/all_in_one.mp4
