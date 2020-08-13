Sequence Prediction using Spectral RNNs
------------------------------------
Source code for the paper: 
https://arxiv.org/pdf/1812.05645.pdf if you find this work useful please consider citing the paper.
The code has been tested using Tensorflow 1.10.0 but should also run in 1.13.

###### Experiments:

 - To recreate the synthetic experiments run `synthetic_signal_test.py` after adjusting the   
   hyperparameters as described in the paper.
   The results may be plotted by using `synthetic_signal_plot.py` after adjusting the logfile path in that file.

 - To run the power prediction experiments download:
   http://www.wolter.tech/wordpress/wp-content/uploads/2019/02/power_data.zip
   And extract it in the `power_experiments` folder. Then run `power_load_pred_exp.py`
   with the desired parameters.

 - The mocap experiments use the human3.6m data set available at
   http://vision.imar.ro/human3.6m/
   After downloading and pickling the data run `mocap_train_exp.py` to repeat the experiments
   in the paper.
 

###### Citation:
```
@inproceedings{wolter2018spectral,
  title={Sequence Prediction using Spectral RNNs},
  author={Wolter, Moritz and Gall, Juergen and Yao, Angela},
  booktitle={29th International Conference on Artificial Neural Networks},
  year={2020}
}
```

###### Demo Video:
![Alt Text](demo.gif)

In the demo the desired behaviour is shown on the left, while the right side depicts the network predictions.
The red and blue coloured stick figures are context, the green and yellow figures show the ground truth and 
network output.

###### Funding:
This work has been partly funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) YA 447/2-1 and GA 1927/4-1 (FOR2535 Anticipating Human Behavior) as well as by the National Research Foundation of Singapore under its NRF Fellowship Programme [NRF-NRFFAI1-2019-0001].
