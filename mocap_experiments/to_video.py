# Created by moritz (wolter@cs.uni-bonn.de) at 13/02/2020

import os
import re
from glob import glob
import pickle
import collections
from mocap_experiments.write_movie import read_pose, write_movie
import mocap_experiments.viz as viz
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
        array = array[viz.H36M_USED_JOINTS, :, :]
        write_movie(array, name='./test_vids_2/train_' + action + actor + '.mp4')

    val_lst = []
    for val_file_path in val_file_path_lst:
        array = pycdf.CDF(val_file_path)
        array = read_pose(array['Pose'])
        action = val_file_path.split('/')[-1].split('.')[-2]
        actor = val_file_path.split('/')[-4]
        assert actor == 'S5'
        array = array[viz.H36M_USED_JOINTS, :, :]
        write_movie(array, name='./test_vids_2/val_' + action + actor + '.mp4')
    # pickle.dump(val_lst, open(data_path + '/val_' + dataset + 'v2.pkl', 'wb'))
    # print('wrote to:', data_path + '/val_' + dataset + 'v2.pkl')
