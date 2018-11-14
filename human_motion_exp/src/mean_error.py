import data_utils
import numpy as np
import scipy as sci
import tensorflow as tf

act_lst = ['walking', 'eating', 'smoking', 'discussion', 
           'directions', 'greeting', 'phoning', 'posing',
           'purchases', 'sitting', 'sittingdown', 'takingphoto',
           'waiting', 'walkingdog', 'walkingtogether']
walking_lst = ['walking']


# figure out what the test error of the mean pose would be.
def read_all_data(actions=walking_lst, seq_length_in=50, seq_length_out=25,
                  data_dir="./data/h3.6m/dataset", one_hot=True):
    """
    Loads data for training/testing and normalizes it.

    Args
    actions: list of strings (actions) to load
    seq_length_in: number of frames to use in the burn-in sequence
    seq_length_out: number of frames to use in the output sequence
    data_dir: directory to load the data from
    one_hot: whether to use one-hot encoding per action
    Returns
    train_set: dictionary with normalized training data
    test_set: dictionary with test data
    data_mean: d-long vector with the mean of the training data
    data_std: d-long vector with the standard dev of the training data
    dim_to_ignore: dimensions that are not used becaused stdev is too small
    dim_to_use: dimensions that we are actually using in the model
    """

    # === Read training data ===
    print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
           seq_length_in, seq_length_out))
    train_subject_ids = [1, 6, 7, 8, 9, 11]
    test_subject_ids = [5]

    train_set, complete_train = data_utils.load_data(data_dir, train_subject_ids,
                                                     actions, one_hot)
    test_set, complete_test = data_utils.load_data(data_dir, test_subject_ids,
                                                   actions, one_hot)

    # Compute normalization stats
    data_mean, data_std, dim_to_ignore, dim_to_use = \
        data_utils.normalization_stats(complete_train)

    # Normalize -- subtract mean, divide by stdev
    train_set = data_utils.normalize_data(train_set, data_mean, data_std,
                                          dim_to_use, actions, one_hot)
    test_set = data_utils.normalize_data(test_set, data_mean, data_std,
                                         dim_to_use, actions, one_hot)
    print("done reading data.")

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = \
    read_all_data(actions=walking_lst, seq_length_in=50, seq_length_out=25,
                  data_dir="./data/h3.6m/dataset", one_hot=False)

train_keys = list(train_set.keys())
pose_lst = []
for key in train_keys:
    pose_lst.append(train_set[key])
pose_array = np.concatenate(pose_lst, axis=0)
mean_pose = np.mean(pose_array, axis=0)

test_pose_lst = []
test_keys = list(test_set.keys())
for key in test_keys:
    test_pose_lst.append(test_set[key])
test_pose_array = np.concatenate(test_pose_lst, axis=0)
mean_array = np.concatenate(test_pose_array.shape[0]*[np.expand_dims(mean_pose, 0)],
                            axis=0)

mean_pose_mse = np.mean(np.square(mean_array - test_pose_array).flatten())
print(mean_pose_mse)
actions = walking_lst

# include a batch dimensions
test_pose_array = np.expand_dims(test_pose_array, axis=0)
mean_array = np.expand_dims(mean_array, axis=0)


test_euler = []
mean_pose_euler = []
# expmap -> rotmat -> euler
test_pose_denormed = data_utils.unNormalizeData(
    test_pose_array[0, :, :], data_mean, data_std, dim_to_ignore, actions, False)
mean_pose_denormed = data_utils.unNormalizeData(
    mean_array[0, :, :], data_mean, data_std, dim_to_ignore, actions, False)

for j in np.arange(test_pose_denormed.shape[0]):
    for k in np.arange(3, 97, 3):
        test_pose_denormed[j, k:k+3] = \
            data_utils.rotmat2euler(data_utils.expmap2rotmat(test_pose_denormed[j,
                                                                                k:k+3]))

for j in np.arange(mean_pose_denormed.shape[0]):
    for k in np.arange(3, 97, 3):
        mean_pose_denormed[j, k:k+3] = \
            data_utils.rotmat2euler(data_utils.expmap2rotmat(mean_pose_denormed[j,
                                                                                k:k+3]))

# The global translation (first 3 entries) and global rotation
# (next 3 entries) are also not considered in the error, so the_key
# are set to zero.
# See https://github.com/asheshjain399/RNNexp/issues/6#issuecomment-249404882
# gt_i = np.copy(srnn_gts_euler[action][i])
test_pose_denormed[:, 0:6] = 0
mean_pose_denormed[:, 0:6] = 0

# Now compute the l2 error. The following is numpy port of the error
# function provided by Ashesh Jain (in matlab), available at
# https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/motionGenerationError.m#L40-L54
idx_to_use = np.where(np.std(test_pose_denormed, 0) > 1e-4)[0]

euc_error = np.power(test_pose_denormed[:, idx_to_use]
                     - mean_pose_denormed[:, idx_to_use], 2)
euc_error = np.sum(euc_error, 1)
euc_error = np.sqrt(euc_error)
mean_euc_error = np.mean(euc_error)
print(mean_euc_error)

# # This is simply the mean error over the N_SEQUENCE_TEST examples
# mean_mean_errors = np.mean(mean_errors, 0)