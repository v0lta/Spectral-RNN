"""
Based on https://github.com/magnux/MotionGAN/blob/master/utils/human36_skels_to_h5.py
Download the D3 Position files from the by subject category in http://vision.imar.ro/human3.6m/filebrowser.php
extract them and run this code to create pickled versions of the dataset.
"""

import os
import re
from glob import glob
import pickle
import collections
from mocap_experiments.write_movie import read_pose
os.environ["CDF_LIB"] = '/home/moritz/CDF/cdf37_0-dist/lib'
from spacepy import pycdf
PoseData = collections.namedtuple('PoseData', ['f', 'action', 'actor', 'array'])
subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
cameras = ['54138969', '55011271', '58860488', '60457274']
actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
           'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
           'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']


if __name__ == "__main__":
    data_path = '/home/moritz/uni/fourier-prediction/mocap_experiments/data/un_zipped/by_actor'
    dataset = 'h36m'

    train_file_path_lst = []
    val_file_path_lst = []
    for (path, dirs, files) in os.walk(data_path):
        for file_path in files:
            split_point = file_path.split('.')
            if split_point[-1] == 'cdf':
                path_folders = path.split('/')
                if path_folders[-3] == 'S5':
                    val_file_path_lst.append(path + '/' + file_path)
                else:
                    train_file_path_lst.append(path + '/' + file_path)

    # create the training_file:
    training_lst = []
    for training_file_path in train_file_path_lst:
        array = pycdf.CDF(training_file_path)
        array = read_pose(array['Pose'])
        action = training_file_path.split('/')[-1].split('.')[-2]
        actor = training_file_path.split('/')[-4]
        assert actor != 'S5'
        training_lst.append(PoseData(training_file_path, action, actor, array))
    pickle.dump(training_lst, open(data_path + '/train_' + dataset + 'v2.pkl', 'wb'))
    print('wrote to:', data_path + '/train_' + dataset + 'v2.pkl')

    val_lst = []
    for val_file_path in val_file_path_lst:
        array = pycdf.CDF(val_file_path)
        array = read_pose(array['Pose'])
        action = val_file_path.split('/')[-1].split('.')[-2]
        actor = val_file_path.split('/')[-4]
        assert actor == 'S5'
        val_lst.append(PoseData(val_file_path, action, actor, array))
    pickle.dump(val_lst, open(data_path + '/val_' + dataset + 'v2.pkl', 'wb'))
    print('wrote to:', data_path + '/val_' + dataset + 'v2.pkl')