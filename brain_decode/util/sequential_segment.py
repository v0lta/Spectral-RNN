import random
import ipdb
import numpy as np
from util.trial_segment import _create_cnt_y_and_trial_bounds_from_start_and_ival
from util.util import SignalAndTarget
# from util.trial_segment import create_signal_target_from_raw_mne
debug_here = ipdb.set_trace


def get_seq_signal_and_target(raw, name_to_start_codes, epoch_ival_ms):
    '''
    Return full data array as well as the labels.
    Returns:
        data: [sensors, time]
        cnt_y: [time, 4]
    '''
    data = raw.get_data()
    events = np.array([raw.info["events"][:, 0], raw.info["events"][:, 2]]).T
    fs = raw.info["sfreq"]
    debug_here()
    cnt_y, i_start_stops = _create_cnt_y_and_trial_bounds_from_start_and_ival(
        data.shape[1], events, fs, name_to_start_codes, epoch_ival_ms)
    return data, cnt_y


def get_sequential_batches(raw, name_to_start_codes, epoch_ival_ms, length):
    '''
    Return full data array as well as the labels.
    Returns:
        data: [sensors, time]
        cnt_y: [time, 4]
    '''
    data, cnt_y = get_seq_signal_and_target(raw, name_to_start_codes, epoch_ival_ms)
    steps = data.shape[1]//length
    offset_max = data.shape[1] % length
    start = int(np.random.uniform(0, offset_max))
    batches = []
    for step in range(0, steps):
        current_start = start + step*length
        batches.append({'data': data[:, current_start:(current_start + length)],
                        'labels': cnt_y[current_start:(current_start + length), :]})
    random.shuffle(batches)

    X = []
    y = []
    for batch in batches:
        X.append(batch['data'])
        y.append(batch['labels'])
    return SignalAndTarget(X=np.stack(X), y=np.stack(y))
